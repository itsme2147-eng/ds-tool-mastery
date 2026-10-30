# Data Science Tool Mastery Project

Task 4 of the AI/ML Research Specialist virtual internship (Yuva Intern, Thiruvananthapuram).
Demonstrates proficiency with industry-standard data science tools through **verified practical exercises**
and **two portfolio pieces**: a tested, installable Python package and an end-to-end ML project.

## Objectives -> evidence
| Objective | Evidence |
|---|---|
| 1. Master industry-standard tools for an AI/ML specialisation | NumPy, Pandas, SQL (sqlite3), Matplotlib/Seaborn, SciPy, statsmodels, scikit-learn, LightGBM, Git, pytest, joblib |
| 2. Demonstrate proficiency through practical exercises | `notebooks/01_Tool_Mastery_Exercises.ipynb`: 31 automated checks, all passing |
| 3. Portfolio pieces | `src/toolkit` + 21 tests; `notebooks/02_Breast_Cancer_ML_Portfolio.ipynb`; `models/MODEL_CARD.md` |

## Repository layout
```
src/toolkit/      transformers.py (CorrelationFilter, Winsorizer), evaluation.py, data.py, train.py (CLI)
tests/            21 pytest tests
notebooks/        01_Tool_Mastery_Exercises.ipynb, 02_Breast_Cancer_ML_Portfolio.ipynb (both executed)
models/           final_model.joblib, MODEL_CARD.md, CLI metrics
data/             penguins.csv, flights.csv, SOURCE.md (provenance)
outputs/          figures/, exercise_results.json, ml_results.json
docs/             Task4_Tool_Mastery_Report.docx
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest -q                                                 # 21 passed
python -m toolkit.train --model logreg --seed 42          # reproducible training CLI
jupyter notebook notebooks/                               # run both notebooks top to bottom
```
CLI output with the default seed (note: the CLI's default pipeline keeps the 0.95 correlation filter, so its numbers differ slightly from the tuned notebook model, which keeps all 30 features):
```
logreg: test ROC-AUC 0.995 [0.985, 1.000] | recall 0.929 @0.5, 0.976 @0.240 (FN 1, FP 2)
```

## Headline results
- 31/31 exercise checks pass (for example a from-scratch SVD PCA matches scikit-learn; SQL and pandas aggregates agree; a leakage demo shows 0.86 fake accuracy on pure noise vs 0.58 when done correctly).
- Cancer classifier: test ROC-AUC 0.996 [0.986, 1.000]; recall 0.976 and specificity 0.958 at a threshold chosen without touching the test set; all pre-declared success criteria met on average, but not in every one of 20 random splits.
- The custom CorrelationFilter was *not* selected by tuning for logistic regression (all 30 features kept); reported as a result.

## Scope and honesty notes
Deep-learning frameworks (PyTorch/TensorFlow) are **not** demonstrated here; they are listed as the next learning step.
Data are public research datasets; the cancer model is an educational exercise, not a clinical tool.
