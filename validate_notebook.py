"""External HLE smoke test plus execution of every teaching-notebook code cell."""
from pathlib import Path
import os
import sys
import json
import hashlib
import tempfile
import time
import importlib.metadata

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
PORTABLE = (HERE / "code/private_code/transposed_extended.py").exists()
SOURCE = HERE / "code" if PORTABLE else HERE
RESULTS = HERE / "results" if PORTABLE else HERE.parents[1] / "output/hle_article"
sys.path[:0] = [str(SOURCE), str(SOURCE / "private_code")]

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr, pearsonr
import nbformat
from nbclient import NotebookClient
from transposed_extended import ExtendedTransposed, Settings, higher_order
from numerical_audit import MemoryBoundedTransposed

EXPECTED_SOURCE_SHA = "368100c9c04aa2dc16ae9caf5b02a3c740825bdfb79a4a6d575480e79506f611"


def smoke_test():
    """Fit small synthetic data; test the interface, not substantive convergence."""
    rng = np.random.default_rng(20261019)
    n, domains = 12, 8
    groups = np.repeat(np.arange(domains), 4)
    g = rng.normal(size=n)
    s = rng.normal(size=(n, domains))
    b = rng.normal(1, 1.2, size=len(groups))
    a = rng.lognormal(0, 0.4, size=len(groups))
    y = rng.binomial(1, expit(a * (g[:, None] - b) + .5 * s[:, groups])).astype(float)
    y[rng.random(y.shape) < .03] = np.nan
    settings = Settings(points=3, max_iterations=12)
    records = []
    for family in ["unidimensional", "bifactor", "testlet"]:
        started = time.monotonic()
        estimator = MemoryBoundedTransposed(y, groups, family, settings)
        result = estimator.fit(seed=1)
        scores = result["state"]["scores"]
        assert scores.shape == (n, 1 if family == "unidimensional" else domains + 1)
        assert np.isfinite(scores).all() and np.isfinite(result["objective"])
        p = result["prediction"]
        assert p.shape == y.shape and np.isfinite(p).all()
        assert ((p >= 0) & (p <= 1)).all()
        np.testing.assert_allclose(result["weights"].sum(axis=1), 1, atol=1e-12)
        assert len(result["history"]) == result["iterations"] <= 12
        assert estimator.mask.sum() == np.isfinite(y).sum()

        # Compare the original dense update with the derivative-batching wrapper.
        original = ExtendedTransposed(y, groups, family, settings)
        state = original.start(1)
        e = original.estep(state)
        old_update, _ = original.mstep(state, e)
        new_update, _ = estimator.mstep(state, e)
        update_error = 0.
        for key in ["scores", "tau", "means", "modes"]:
            np.testing.assert_allclose(old_update[key], new_update[key], atol=1e-8, rtol=1e-8)
            update_error = max(update_error, float(np.max(abs(old_update[key] - new_update[key]))))
        record = {
            "family": family, "interface_passed": True, "iterations": result["iterations"],
            "converged": bool(result["converged"]), "stopping_reason": result["stopping_reason"],
            "wrapper_update_max_difference": update_error,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        if family == "testlet":
            lam, h = higher_order(result["state"])
            tau = result["state"]["tau"]
            before = a * (scores[:, 0, None] - b) + a * tau[groups] * scores[:, 1:][:, groups]
            after = (a / lam[groups]) * (h[:, groups] - b * lam[groups])
            np.testing.assert_allclose(before, after, atol=1e-12, rtol=1e-12)
            record["higher_order_predictor_max_difference"] = float(np.max(abs(before - after)))
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def check_saved_results():
    y = pd.read_csv(SOURCE / "primary_responses.csv", index_col=0)
    metadata = pd.read_csv(SOURCE / "item_metadata.csv", index_col=0).loc[y.columns]
    arr = y.to_numpy(float)
    observed = np.isfinite(arr)
    assert arr.shape == (29, 513) and observed.sum() == 14760
    assert (~observed).sum() == 117 and (np.nansum(arr, axis=0) == 0).sum() == 85
    assert metadata["category"].nunique() == 8
    assert np.isin(arr[observed], [0, 1]).all()
    scores = pd.read_csv(RESULTS / "model_scores.csv").set_index("model_id").loc[y.index]
    np.testing.assert_allclose(scores["percent_correct"], 100*np.nanmean(arr, axis=1), atol=1e-12)
    np.testing.assert_allclose(scores["percent_rank"], scores["percent_correct"].rank(ascending=False))
    comparison = pd.read_csv(RESULTS / "score_comparisons.csv").set_index("family")
    correlations = {}
    for family in ["unidimensional", "bifactor", "testlet"]:
        g = scores[family + "_score"]
        np.testing.assert_allclose(scores[family + "_rank"], g.rank(ascending=False))
        rho = spearmanr(scores["percent_correct"], g).statistic
        r = pearsonr(scores["percent_correct"], g).statistic
        np.testing.assert_allclose(rho, comparison.loc[family, "spearman_general"], atol=1e-12)
        correlations[family] = {"pearson": float(r), "spearman": float(rho)}
    logs = pd.read_csv(RESULTS / "restart_search.csv")
    counts = logs.groupby(["analysis", "repeat", "fold", "family"], dropna=False).size()
    assert len(counts) == 48 and counts.eq(20).all()
    selections = logs.groupby(["analysis", "repeat", "fold", "family"], dropna=False)["selected_for_refinement"].sum()
    assert selections.eq(1).all()
    with np.load(SOURCE / "cv_splits.npz", allow_pickle=False) as data:
        folds = data["folds"]
    assert folds.shape == (3, 29, 513)
    assert np.all(folds[:, ~observed] == -1)
    assert np.all((folds[:, observed] >= 0) & (folds[:, observed] < 5))
    summary = pd.read_csv(RESULTS / "cv_summary.csv").set_index("family")
    repeats = pd.read_csv(RESULTS / "cv_repeat_losses.csv")
    for metric in ["brier", "log_loss"]:
        means = repeats.groupby("family")[metric].mean().reindex(summary.index)
        np.testing.assert_allclose(summary[metric], means, atol=1e-12)
        assert summary[metric].idxmin() == "bifactor"
    assert set(summary.index) == {"unidimensional", "bifactor", "testlet"}
    return {"sample_counts_passed": True, "percentages_and_ranks_passed": True,
            "all_48_fits_have_20_starts": True, "saved_masks_passed": True,
            "predictive_means_passed": True, "correlations": correlations}


def execute_notebook():
    path = HERE / "hle_scoring_walkthrough.ipynb"
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    assert not any("smoke" in cell.source.lower() for cell in nb.cells)
    with tempfile.TemporaryDirectory(prefix="hle-notebook-kernel-") as tmp:
        spec = Path(tmp) / "kernels/hle-validation"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "HLE validation", "language": "python",
        }))
        old = os.environ.get("JUPYTER_PATH", "")
        os.environ["JUPYTER_PATH"] = tmp + (os.pathsep + old if old else "")
        try:
            NotebookClient(nb, timeout=300, kernel_name="hle-validation",
                           resources={"metadata": {"path": str(HERE)}},
                           record_timing=True).execute()
        finally:
            if old:
                os.environ["JUPYTER_PATH"] = old
            else:
                os.environ.pop("JUPYTER_PATH", None)
    nb.metadata.kernelspec = {"display_name": "Python 3 (HLE)", "language": "python", "name": "python3"}
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    figures = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
    assert figures == 2, f"Expected both notebook figures to render; found {figures}."
    stderr = [o.get("text", "") for c in code_cells for o in c.outputs
              if o.output_type == "stream" and o.get("name") == "stderr"]
    assert not stderr, f"Unexpected notebook warnings: {stderr}"
    nbformat.validate(nb)
    nbformat.write(nb, path)
    return {"code_cells": len(code_cells), "all_code_cells_executed": True, "rendered_figures": figures,
            "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    estimator = SOURCE / "private_code/transposed_extended.py"
    digest = hashlib.sha256(estimator.read_bytes()).hexdigest()
    assert digest == EXPECTED_SOURCE_SHA, "The manuscript estimator changed."
    records = smoke_test()
    saved = check_saved_results()
    print("Saved-result checks passed. Executing the notebook in a fresh kernel.", flush=True)
    execution = execute_notebook()
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "scipy", "pandas", "matplotlib", "jax", "jaxlib",
        "nbformat", "nbclient", "ipykernel", "markdown", "latex2mathml", "Pygments"]}
    report = {
        "passed": True, "python": sys.version.split()[0], "versions": versions,
        "estimator_sha256": digest, "smoke_test_is_external_to_notebook": True,
        "smoke_test": records, "saved_results": saved, **execution,
        "scope": "Short synthetic fits for three families, unchanged-source and wrapper checks, "
                 "saved-result consistency, and complete notebook execution. "
                 "No rerun of the manuscript's 20-start full-data or predictive fits.",
    }
    (HERE / "notebook_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, **execution}), flush=True)


if __name__ == "__main__":
    main()
