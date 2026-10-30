"""Evaluation helpers for binary classifiers (positive class = the one we must not miss)."""
from __future__ import annotations
from typing import Callable
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score)


def bootstrap_ci(metric: Callable, y_true, y_pred, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap confidence interval. Returns (point estimate, lower, upper).

    Resamples with replacement; resamples that contain a single class are skipped because
    ranking metrics such as ROC-AUC are undefined for them.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n, stats = len(y_true), []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_pred)), float(lo), float(hi)


def threshold_for_recall(y_true, score, target_recall: float = 0.98) -> float:
    """Largest decision threshold whose recall on (y_true, score) is at least ``target_recall``.

    Use out-of-fold training predictions, never the test set, to choose the threshold.
    """
    y_true, score = np.asarray(y_true), np.asarray(score)
    positives = y_true == 1
    if positives.sum() == 0:
        raise ValueError("no positive cases")
    for t in np.sort(np.unique(score))[::-1]:
        if (score[positives] >= t).mean() >= target_recall:
            return float(t)
    return float(score.min())


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed frequency across bins."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.minimum((proba * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - proba[m].mean())
    return float(ece)


def summarise_binary(y_true, proba, threshold: float = 0.5) -> dict:
    """Confusion-matrix metrics at ``threshold`` plus threshold-free metrics."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    return {
        "threshold": float(threshold), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "specificity": div(tn, tn + fp), "precision": precision,
        "accuracy": div(tp + tn, len(y_true)),
        "f1": div(2 * precision * recall, precision + recall) if precision == precision and recall == recall else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)), "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }
