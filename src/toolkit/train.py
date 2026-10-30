"""Reproducible training CLI for the breast-cancer classifier.

Usage:  python -m toolkit.train --model logreg --seed 42 --out models/
Writes: <out>/<model>.joblib (fitted pipeline) and <out>/<model>_metrics.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import load_cancer
from .evaluation import bootstrap_ci, summarise_binary, threshold_for_recall
from sklearn.metrics import roc_auc_score


def make_estimator(name: str, seed: int):
    return {
        "logreg": lambda: LogisticRegression(C=1.0, max_iter=2000, random_state=seed),
        "svm": lambda: SVC(C=1.0, probability=True, random_state=seed),
        "rf": lambda: RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=seed),
        "lgbm": lambda: LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=1),
    }[name]()


def build_pipeline(name: str, seed: int = 42, corr_threshold: float = 0.95) -> Pipeline:
    from .transformers import CorrelationFilter
    return Pipeline([("corr", CorrelationFilter(corr_threshold)), ("scale", StandardScaler()),
                     ("clf", make_estimator(name, seed))])


def run(model: str = "logreg", seed: int = 42, test_size: float = 0.2, target_recall: float = 0.98,
        out: str | Path = "models") -> dict:
    X, y = load_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    pipe = build_pipeline(model, seed)

    # threshold chosen from OUT-OF-FOLD training predictions only (no test-set peeking)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = threshold_for_recall(y_tr, oof, target_recall)

    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    auc, lo, hi = bootstrap_ci(roc_auc_score, y_te, proba, n_boot=1000, seed=seed)
    result = {
        "model": model, "seed": seed, "n_train": len(X_tr), "n_test": len(X_te),
        "cv_oof_roc_auc": float(roc_auc_score(y_tr, oof)), "test_roc_auc_ci": [auc, lo, hi],
        "at_default_threshold": summarise_binary(y_te, proba, 0.5),
        "at_recall_threshold": summarise_binary(y_te, proba, thr),
        "features_kept": len(pipe.named_steps["corr"].keep_),
    }
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / f"{model}.joblib")
    (out / f"{model}_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["logreg", "svm", "rf", "hgb", "lgbm"], default="logreg")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--target-recall", type=float, default=0.98)
    p.add_argument("--out", default="models")
    a = p.parse_args(argv)
    r = run(a.model, a.seed, a.test_size, a.target_recall, a.out)
    d, t = r["at_default_threshold"], r["at_recall_threshold"]
    print(f"{r['model']}: test ROC-AUC {r['test_roc_auc_ci'][0]:.3f} "
          f"[{r['test_roc_auc_ci'][1]:.3f}, {r['test_roc_auc_ci'][2]:.3f}] | "
          f"recall {d['recall']:.3f} @0.5, {t['recall']:.3f} @{t['threshold']:.3f} (FN {t['fn']}, FP {t['fp']})")


if __name__ == "__main__":
    main()
