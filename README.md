# HLE scoring: accessible Python companion

Open `hle_scoring_walkthrough.ipynb` in Jupyter or VS Code. The executed `notebook.html`
is a standalone reading copy with embedded figures and native MathML equations.

The notebook explains the manuscript's code with short text, equations, canonical
citations, one synthetic EM update, and saved full-data score and predictive comparisons.
It imports Preston Botter's estimator unchanged. It does not refit the full HLE analysis.

## Setup

Use Python 3.12. In this companion folder:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --user --name hle-notebook --display-name "HLE notebook"
```

On Windows, use `.venv\Scripts\python.exe`. Select the HLE kernel and run all cells.
Keep the `code/` and `results/` folders beside the notebook. No API keys are required.
First execution compiles JAX functions and can take longer than later executions.
Version ranges express compatibility intent; `notebook_verification.json` records
the actual tested versions.

## Separate validation

The smoke test is outside the notebook:

```sh
.venv/bin/python validate_notebook.py
.venv/bin/python render_notebook.py
```

The validator fits all three families briefly on synthetic data, checks the testlet
to higher-order predictor identity, checks the original and memory-bounded implementations,
and executes every notebook code cell in a fresh kernel. It verifies saved HLE counts,
percentages, ranks, restart coverage, and loss summaries. A smoke test is not evidence
of convergence or scientific validity; the full manuscript fits have their own
numerical audits.

## Files and provenance

- `hle_scoring_walkthrough.ipynb`: the executed teaching notebook.
- `notebook.html`: standalone reading copy.
- `code/`: unchanged estimator, memory wrapper, orchestration, response matrix,
  question category metadata, and saved response masks.
- `results/`: selected manuscript tables.
- `provenance.json`: source release and estimator checksum.
- `notebook_verification.json`: execution scope, versions, and checksums.

The response data are from Sharma, Nadela, and Matteson (2026),
[HLE Psychometrics](https://github.com/matrix-mayank/hle-psychometrics).
The notebook contains no HLE question texts. Upstream data and code retain their
respective rights; this folder does not assign them a new license.
The source revision is recorded in `provenance.json`.

The [HLE IRT scoring repository](https://github.com/pbotter42/hle-irt-scoring) accompanies
[I Re-scored Humanity’s Last Exam. Llama Fell 14 Places.](https://benchmarkanatomy.substack.com/p/i-re-scored-humanitys-last-exam-llama)
The mathematical formulation and implementation may change as the methodological manuscripts develop.
Contact Preston Botter at pdbotter@gmail.com for discussion.

The original orchestration contains an unused historical percentage-baseline helper;
neither this notebook nor its validation runs it. The reported predictive comparison
contains only the three fitted IRT models.
