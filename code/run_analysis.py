"""HLE scoring comparison using an unchanged copy of the author's estimator."""
import os
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import sys, argparse, json, time, hashlib
sys.path.insert(0, str(Path(__file__).resolve().parent / 'private_code'))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from transposed_extended import ExtendedTransposed, Settings, higher_order
from numerical_audit import MemoryBoundedTransposed as ExtendedTransposed

ROOT = Path(__file__).resolve().parent
DOMAINS = ['Biology/Medicine', 'Math', 'Humanities/Social Science', 'Computer Science/AI',
           'Other', 'Physics', 'Chemistry', 'Engineering']

def load_data():
    df = pd.read_csv(ROOT/'primary_responses.csv', index_col=0)
    meta = pd.read_csv(ROOT/'item_metadata.csv', index_col=0).loc[df.columns]
    group = np.array([DOMAINS.index(x) for x in meta.category])
    return df, meta, group

def save(result, stem, compact=False):
    stem.parent.mkdir(parents=True, exist_ok=True)
    arrays = {'state_'+k: v for k,v in result['state'].items()}
    if not compact:
        arrays['prediction'] = result['prediction']
        arrays['posterior_item_mean'] = np.einsum('jk,jkd->jd', result['weights'], result['nodes'])
    if result['family'] == 'testlet' and not compact:
        arrays['ho_loading'], arrays['ho_scores'] = higher_order(result['state'])
    small = {k:v for k,v in result.items() if k not in ['state','modes','hessians','nodes','weights','prediction']}
    small['max_score_abs'] = float(np.max(abs(result['state']['scores'])))
    small['tau'] = result['state']['tau'].tolist()
    array_tmp=stem.with_suffix('.npz.part');json_tmp=stem.with_suffix('.json.part')
    with array_tmp.open('wb') as fh:np.savez_compressed(fh,**arrays)
    json_tmp.write_text(json.dumps(small,indent=2))
    os.replace(array_tmp,stem.with_suffix('.npz'))
    os.replace(json_tmp,stem.with_suffix('.json'))

def state_from(stem):
    with np.load(stem.with_suffix('.npz')) as data:
        return {k:data['state_'+k].copy() for k in ['scores','tau','means','modes']}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('family', choices=['unidimensional','bifactor','testlet'])
    ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--variant',default='primary', choices=['primary','missing_wrong','bound20','penalty_half','penalty_double','common_items'])
    ap.add_argument('--points',type=int,default=9)
    ap.add_argument('--score-tolerance',type=float,default=.001)
    args=ap.parse_args()
    frame,meta,group=load_data(); y=frame.to_numpy(float)
    settings_kwargs={'score_tolerance':args.score_tolerance}
    if args.variant=='missing_wrong': y=np.nan_to_num(y)
    if args.variant=='bound20': settings_kwargs['bifactor_mean_upper']=20.
    if args.variant=='penalty_half': settings_kwargs['score_penalty']=.5
    if args.variant=='penalty_double': settings_kwargs['score_penalty']=2.
    common=np.isfinite(y).all(axis=0)
    if args.variant=='common_items': y=y[:,common];group=group[common]
    folder=ROOT/'fits'/args.variant/args.family;folder.mkdir(parents=True,exist_ok=True)
    candidates=[]
    if args.variant=='primary':
        est=ExtendedTransposed(y,group,args.family,Settings(points=5,max_iterations=240,**settings_kwargs))
        for seed in range(args.starts):
            stem=folder/f'start_{seed:02d}_q5'
            if not stem.with_suffix('.json').exists():
                result=est.fit(seed,verbose=True);save(result,stem)
            small=json.loads(stem.with_suffix('.json').read_text())
            candidates.append((small['objective'],stem,seed))
            print('START_COMPLETE',args.family,seed,small['converged'],small['objective'],flush=True)
        _,best,seed=max(candidates)
        initial=state_from(best)
        previous=folder/'selected_q9'
        if args.points>9 and previous.with_suffix('.npz').exists():
            initial=state_from(previous)
    else:
        full_folder=ROOT/'runs20/full'/args.family
        full_audit=json.loads((full_folder/'audit.json').read_text())
        initial=state_from(full_folder/f"selected_q{full_audit['points_fit']}");seed=0
        if args.variant=='common_items': initial['modes']=initial['modes'][common]
        previous=[p for p in folder.glob('selected_q*.json') if int(p.stem.split('q')[-1])<args.points]
        if previous:
            initial=state_from(max(previous,key=lambda p:int(p.stem.split('q')[-1])).with_suffix(''))
    final=folder/f'selected_q{args.points}'
    est=ExtendedTransposed(y,group,args.family,Settings(points=args.points,max_iterations=240,**settings_kwargs))
    if not final.with_suffix('.json').exists():
        result=est.fit(seed,initial=initial,verbose=True);save(result,final)
        del result
    state=state_from(final); e=est.estep(state)
    check_points=max(args.points+6,15)
    check=est.estep(state,points=check_points)
    audit={'variant':args.variant,'family':args.family,'shape':list(y.shape),'points_fit':args.points,
      'points_check':check_points,'normalized_score_residual':est.normalized_score(state,e),
      'higher_order_score_residual':est.normalized_score(state,check),
      'objective_change_per_item':abs(check['objective']-e['objective'])/y.shape[1],
      'max_probability_change':float(np.max(abs(check['prediction']-e['prediction']))),
      'mean_probability_change':float(np.mean(abs(check['prediction']-e['prediction']))),
      'finite_probabilities':bool(np.isfinite(e['prediction']).all()),
      'probability_range':[float(e['prediction'].min()),float(e['prediction'].max())],
      'score_rank_correlation_with_accuracy':float(spearmanr(state['scores'][:,0],np.nanmean(y,axis=1)).statistic),
      'q5_start_objectives':[float(v[0]) for v in candidates],
      'tau':state['tau'].tolist(),'domain_order':DOMAINS}
    if args.family=='testlet':
        lam,h=higher_order(state)
        # Check the exact pointwise predictor transformation on posterior item means.
        im=np.einsum('jk,jkd->jd',e['weights'],e['nodes'])
        a=np.exp(im[:,1]);b=im[:,0]
        z1=a[None,:]*(state['scores'][:,0,None]-b[None,:])+a[None,:]*state['tau'][group][None,:]*state['scores'][:,1:][:,group]
        z2=(a/lam[group])[None,:]*(h[:,group]-(b*lam[group])[None,:])
        audit['testlet_ho_max_predictor_discrepancy']=float(np.max(abs(z1-z2)))
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))
    print('AUDIT',json.dumps(audit),flush=True)

if __name__=='__main__':main()
