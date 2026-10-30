# Model card: malignant vs benign classifier (educational)

**Not a medical device.** Learning project on a public research dataset; no clinical validation.

## Model
- Pipeline: `CorrelationFilter` (tuned threshold = 1.0, i.e. keeps all 30 features) -> `StandardScaler` -> `LogisticRegression` (C = 1.0, L2).
- Positive class: **malignant (1)**. Output: probability of malignancy.
- Operating threshold: **0.207**, chosen from out-of-fold training predictions to reach >= 98% recall (stored in `final_model.joblib` next to the pipeline).

## Data
UCI Wisconsin Diagnostic Breast Cancer (569 cases: 212 malignant, 357 benign), 30 numeric features computed from digitised fine-needle-aspirate images. Public, bundled with scikit-learn. Split 80/20 stratified (455 train / 114 test, seed 42).

## Evaluation (held-out test set, n = 114, 42 malignant; bootstrap 95% CIs)
| Metric | Value |
|---|---|
| ROC-AUC | 0.996 [0.986, 1.000] |
| Recall (malignant) @ 0.207 | 0.976 [0.921, 1.000] (41 of 42) |
| Specificity @ 0.207 | 0.958 [0.908, 1.000] (69 of 72) |
| Brier score | 0.021 |

Nested CV ROC-AUC (training set): logistic regression 0.994 +/- 0.004. Across 20 random splits: mean ROC-AUC 0.995, mean recall 0.974 (min 0.929), mean specificity 0.928 (min 0.847).

## Limitations
- Only 569 cases from one source; no external or prospective validation.
- Intervals are wide because the test set is small; one confident false negative (p = 0.06) occurred.
- Features come from the dataset authors' image-analysis pipeline; other imaging setups may shift distributions.
- Correlated features make individual coefficients unstable; read importances loosely.

## Ethical considerations
The recall target trades missed cancers against unnecessary follow-up. That trade-off is a clinical decision and must involve domain experts. Do not use this model for patient care.

## Reproduce
`pip install -r requirements.txt` then run `notebooks/02_Breast_Cancer_ML_Portfolio.ipynb`, or `python -m toolkit.train --model logreg` (needs `pip install -e .` or `PYTHONPATH=src`). The CLI uses a default pipeline (correlation filter 0.95) and therefore gives slightly different numbers from the tuned notebook model above.
