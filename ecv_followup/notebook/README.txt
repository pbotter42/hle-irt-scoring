HLE ECV follow-up: accessible Python companion

Open hle_ecv_walkthrough.ipynb in Jupyter or VS Code and run all cells with
Python 3.12. Keep data/, fit/, and code/ next to it. The notebook working
directory should be this notebook/ folder. notebook.html is the executed
reading copy with embedded images and equations.

Setup:
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python -m ipykernel install --user --name hle-ecv --display-name "HLE ECV"

Choose the HLE ECV kernel in your notebook editor.
Windows: use .venv\Scripts\python.exe in place of .venv/bin/python.
The first item E-step can take about a minute while JAX compiles. It uses
the saved full-data fit, not a new parameter search or any train/test split.
The source research estimator is included unchanged in code/.

The notebook covers score/rank correlations, recovery of posterior-mean
slopes, standardized logistic factor loadings, overall/subject/item ECV,
PUC, empirical score correlations, negative specific slopes, and two CFA/IRT-style
path diagrams that connect the loading pattern to ECV. It exports
five figures to ../figures/ and reproduced tables to results/.

ECV is a loading-based diagnostic under an explicitly stated orthogonal,
unit-variance reference. The fixed-score estimator does not impose empirical
orthogonality. Conventional thresholds and additive variance interpretations
therefore require caution. No full-model uncertainty intervals are supplied.

Validation remains separate from the notebook:
.venv/bin/python validate_notebook.py
.venv/bin/python render_notebook.py

The validation executes every cell in a fresh kernel and checks the reproduced
results against the saved analysis. notebook_verification.json records its
scope and runtime versions. This validation is not evidence of model validity.
The requirements specify compatibility ranges, not a lockfile.

Original responses/category metadata: Sharma, Nadela, & Matteson (2026),
HLE Psychometrics, https://github.com/matrix-mayank/hle-psychometrics
provenance.json records the source. No HLE question texts are included.
Upstream data and code retain their respective rights; no new license is assigned.
The notebook and follow-up materials are available in the repository:
https://github.com/pbotter42/hle-irt-scoring
Contact Preston Botter at pdbotter@gmail.com for research discussion.

Reported ECV now excludes entire items with negative posterior-mean slopes.
There are 484 retained items; the 513-item fitted model is unchanged. See the
notebook footnote for the inclusion rule and full-item comparison.
