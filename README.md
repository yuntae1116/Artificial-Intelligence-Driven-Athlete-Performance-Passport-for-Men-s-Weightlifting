# Athlete Performance Passport — Scientific Reports analysis

Python implementation adapted from `AI-driven_APP_workflow.ipynb`.

## Install
Python 3.11 is recommended.

```bash
pip install -r requirements_APP.txt
```

## Run

```bash
python APP_revision_workflow.py --lifting_sample.CSV --out results
```

Audit input and athlete linkage without training:

```bash
python APP_revision_workflow.py --lifting_sample.CSV --out results --audit-only
```

Run software tests with synthetic data:

```bash
python APP_revision_workflow.py --self-test --out software_test
```

For a quick pipeline check (not manuscript estimates):

```bash
python APP_revision_workflow.py --lifting_sample.CSV --out smoke_results --smoke
```

## Scope

Includes preprocessing, profile-level partitions, RNN/LSTM prediction,
random-intercept/random-slope mixed-effects benchmark, cross-fitted residuals,
XGBoost classification, probability calibration, ROC/PR evaluation, SHAP,
feature ablations, and sensitivity analyses.

The script uses saved hyperparameter settings from the supplied workflow.
**It does not run the original randomized hyperparameter search.**
The raw IWF-derived dataset is not included. Results cannot be verified
without running the script on the study data. Source-row order within a year
is not independently verified as chronological.

## Reproducibility notes

The notebook's own documentation describes this as a revised analysis workflow,
not proof of exact reproduction of every manuscript estimate. Compare generated
cohort sizes, metrics, and figures with the submitted manuscript before
describing it as the exact publication pipeline.

Do not commit athlete-level raw data, private files, or generated results
without checking disclosure and data-sharing permissions.
