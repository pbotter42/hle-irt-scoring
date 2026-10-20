"""Full-data scoring and separate repeated masked-response CV.

The author's estimator is imported unchanged. Validation responses never enter
initialization, fitting, restart selection, or quadrature checks.
"""
from run_analysis import ROOT, DOMAINS, load_data, save, state_from, ExtendedTransposed, Settings
import argparse
import gc
import fcntl
import hashlib
import json
import time
import numpy as np

RUN = ROOT / 'runs20'
Q = {'unidimensional': 15, 'testlet': 15, 'bifactor': 11}

def make_splits():
    frame, meta, group = load_data()
    observed = np.isfinite(frame.to_numpy(float))
    path = ROOT / 'cv_splits.npz'
    if path.exists():
        folds = np.load(path)['folds']
    else:
        folds = np.full((3,) + observed.shape, -1, dtype=np.int8)
        for repeat in range(3):
            rng = np.random.default_rng(20260920 + repeat)
            for model in range(len(frame)):
                for domain in range(len(DOMAINS)):
                    ids = np.flatnonzero(observed[model] & (group == domain))
                    shuffled = rng.permutation(ids)
                    # Randomize which folds receive any remainder, independently
                    # of responses. Every observed cell is held out once/repeat.
                    offset = rng.integers(5)
                    folds[repeat, model, shuffled] = (np.arange(len(ids)) + offset) % 5
        np.savez_compressed(path, folds=folds, models=frame.index.to_numpy(str),
                            items=frame.columns.to_numpy(str))
    assert folds.shape == (3,) + observed.shape
    assert np.all(folds[:, ~observed] == -1)
    assert np.all((folds[:, observed] >= 0) & (folds[:, observed] < 5))
    return frame, group, folds

def data_for(task):
    frame, group, folds = make_splits()
    y = frame.to_numpy(float)
    if task == 'full':
        valid = np.zeros_like(y, dtype=bool)
    else:
        repeat, fold = map(int, task.split('_'))
        valid = folds[repeat] == fold
    training = y.copy()
    training[valid] = np.nan
    assert not np.isfinite(training[valid]).any()
    return y, training, group, valid

def losses(y, prediction):
    p = np.clip(prediction, 1e-12, 1-1e-12)
    return {'n': int(len(y)), 'brier': float(np.mean((prediction-y)**2)),
            'log_loss': float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))}

def baseline(task):
    y, train, group, valid = data_for(task)
    row = (np.nansum(train, axis=1)+.5)/(np.isfinite(train).sum(1)+1)
    pred_row = np.broadcast_to(row[:, None], y.shape)
    pred_domain = np.zeros_like(y)
    for d in range(len(DOMAINS)):
        sub = train[:, group == d]
        p = (np.nansum(sub, axis=1)+.5)/(np.isfinite(sub).sum(1)+1)
        pred_domain[:, group == d] = p[:, None]
    folder = RUN / task
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'baseline_predictions.npz',
                        row=pred_row[valid], domain=pred_domain[valid])
    result = {name: losses(y[valid], p[valid])
              for name, p in [('percent_correct', pred_row), ('domain_percent', pred_domain)]}
    (folder/'baselines.json').write_text(json.dumps(result, indent=2))

def coarse(task, family, seeds):
    _, train, group, _ = data_for(task)
    folder = RUN / task / family
    folder.mkdir(parents=True, exist_ok=True)
    # The coarse grid itself leaves a score residual near 0.001--0.003.
    # Search at a tolerance consistent with that grid, then impose the original
    # 0.001 criterion at high order. Earlier saved starts used the stricter
    # criterion and can be retained: the fitted objective is the same.
    est = ExtendedTransposed(train, group, family, Settings(points=5, max_iterations=240,
                                                          score_tolerance=5e-3))
    for seed in seeds:
        stem = folder / f'start_{seed:02d}_q5'
        if not stem.with_suffix('.json').exists():
            result = est.fit(seed, verbose=False)
            save(result, stem, compact=True)
            del result
        info = json.loads(stem.with_suffix('.json').read_text())
        print(json.dumps({'stage': 'restart', 'task': task, 'family': family, 'seed': seed,
                          'objective': info['objective'], 'converged': info['converged'],
                          'seconds': info['elapsed_seconds']}), flush=True)

def refine(task, family):
    folder=RUN/task/family
    folder.mkdir(parents=True,exist_ok=True)
    # Multiple local process queues may help different folds. A per-fit lock
    # prevents duplicate refinements when the queues meet.
    with (folder/'.refinement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        audit_path=folder/'audit.json'
        if audit_path.exists():
            a=json.loads(audit_path.read_text())
            if a['converged'] and a['max_probability_change']<.002 and a['higher_order_score_residual']<.001:
                print(json.dumps({'stage':'already_complete','task':task,'family':family}),flush=True)
                return
        return refine_impl(task,family)

def refine_impl(task, family):
    y, train, group, valid = data_for(task)
    folder = RUN / task / family
    candidates = []
    for seed in range(1, 21):
        stem = folder / f'start_{seed:02d}_q5'
        info = json.loads(stem.with_suffix('.json').read_text())
        candidates.append((info['objective'], seed, stem))
    objective, seed, best = max(candidates)
    initial = state_from(best)
    selected = None
    for points in [9, Q[family]]:
        stem = folder / f'selected_q{points}'
        est = ExtendedTransposed(train, group, family, Settings(points=points, max_iterations=240))
        if not stem.with_suffix('.json').exists():
            result = est.fit(seed, initial=initial, verbose=True)
            save(result, stem)
            del result
        initial = state_from(stem)
        selected = stem
        print(json.dumps({'stage':'refine','task':task,'family':family,'points':points,'seed':seed}),flush=True)
    # Both integrations condition only on the training mask. No held-out labels.
    e = est.estep(initial)
    check_points = Q[family] + 6
    check = est.estep(initial, points=check_points)
    fitinfo = json.loads(selected.with_suffix('.json').read_text())
    while (not fitinfo['converged'] or
           np.max(abs(check['prediction']-e['prediction'])) >= .002 or
           est.normalized_score(initial, check) >= .001):
        points = est.settings.points + (2 if family == 'bifactor' else 6)
        if points > (17 if family == 'bifactor' else 33):
            raise RuntimeError(f'Numerical accuracy requires further investigation: {task} {family}')
        del e, check, est
        gc.collect()
        est = ExtendedTransposed(train, group, family, Settings(points=points, max_iterations=240))
        selected = folder / f'selected_q{points}'
        if not selected.with_suffix('.json').exists():
            result = est.fit(seed, initial=initial, verbose=True)
            save(result, selected)
            del result
        initial = state_from(selected)
        fitinfo = json.loads(selected.with_suffix('.json').read_text())
        e = est.estep(initial)
        check_points = (17 if points < 17 else 21) if family == 'bifactor' else points+6
        check = est.estep(initial, points=check_points)
        print(json.dumps({'stage':'accuracy_refinement','task':task,'family':family,'points':points}),flush=True)
    audit = {'task': task, 'family': family, 'restarts': 20, 'selected_seed': seed,
             'restart_objectives': [x[0] for x in candidates],
             'points_fit': est.settings.points, 'points_check': check_points,
             'converged': fitinfo['converged'],
             'normalized_score_residual': est.normalized_score(initial, e),
             'higher_order_score_residual': est.normalized_score(initial, check),
             'max_probability_change': float(np.max(abs(check['prediction']-e['prediction']))),
             'mean_probability_change': float(np.mean(abs(check['prediction']-e['prediction']))),
             'objective_change_per_item': abs(check['objective']-e['objective'])/y.shape[1],
             'training_observations': int(np.isfinite(train).sum()),
             'validation_observations': int(valid.sum()),
             'no_active_score_bound': bool(np.max(abs(initial['scores'])) < 7.99),
             'no_active_domain_upper_bound': bool(family=='unidimensional' or np.max(initial['tau']) < (3.99 if family=='bifactor' else 19.99)),
             'no_active_domain_bound': bool(family=='unidimensional' or (np.min(initial['tau'])>1e-7 and np.max(initial['tau']) < (3.99 if family=='bifactor' else 19.99))),
             'domain_lower_bound_count': 0 if family=='unidimensional' else int(np.sum(initial['tau']<=1e-7)),
             'domain_lower_bound_domains': [] if family=='unidimensional' else [DOMAINS[i] for i in np.flatnonzero(initial['tau']<=1e-7)],
             'finite_probabilities': bool(np.isfinite(e['prediction']).all())}
    if family == 'testlet':
        from transposed_extended import higher_order
        lam, h = higher_order(initial)
        im = np.einsum('jk,jkd->jd', e['weights'], e['nodes'])
        a, b = np.exp(im[:,1]), im[:,0]
        z1 = a[None,:]*(initial['scores'][:,0,None]-b[None,:]) + a[None,:]*initial['tau'][group][None,:]*initial['scores'][:,1:][:,group]
        z2 = (a/lam[group])[None,:]*(h[:,group]-(b*lam[group])[None,:])
        audit['testlet_ho_max_predictor_discrepancy'] = float(np.max(abs(z1-z2)))
    if task != 'full':
        prediction = e['prediction'][valid]
        np.savez_compressed(folder/'validation_predictions.npz', prediction=prediction)
        audit['losses'] = losses(y[valid], prediction)
        audit['losses_higher_quadrature'] = losses(y[valid], check['prediction'][valid])
    (folder/'audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({'stage':'complete', **audit}), flush=True)
    del e, check, est
    gc.collect()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('family', choices=['unidimensional','bifactor','testlet','baseline'])
    ap.add_argument('--tasks', nargs='+', default=['full'])
    ap.add_argument('--stage', choices=['coarse','refine','all'], default='all')
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(1,21)))
    args = ap.parse_args()
    for task in args.tasks:
        if args.family == 'baseline':
            baseline(task)
            continue
        if args.stage in ('coarse','all'): coarse(task, args.family, args.seeds)
        if args.stage in ('refine','all'): refine(task, args.family)
