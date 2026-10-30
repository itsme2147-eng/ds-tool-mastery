import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from toolkit import bootstrap_ci, expected_calibration_error, summarise_binary, threshold_for_recall


def test_summarise_binary_counts_and_rates():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .2, .6, .1, .1, .2, .3])
    s = summarise_binary(y, p, 0.5)
    assert (s["tp"], s["fn"], s["fp"], s["tn"]) == (2, 1, 1, 4)
    assert s["recall"] == pytest.approx(2 / 3) and s["specificity"] == pytest.approx(4 / 5)
    assert s["precision"] == pytest.approx(2 / 3) and s["accuracy"] == pytest.approx(6 / 8)


def test_threshold_for_recall_hits_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([.9, .8, .7, .3, .6, .2, .1, .05])
    t = threshold_for_recall(y, p, 0.75)
    assert t == 0.7 and (p[y == 1] >= t).mean() >= 0.75
    assert threshold_for_recall(y, p, 1.0) == 0.3


def test_threshold_for_recall_requires_positives():
    with pytest.raises(ValueError):
        threshold_for_recall(np.zeros(5), np.linspace(0, 1, 5))


def test_ece_is_zero_for_perfectly_calibrated_extremes_and_large_when_wrong():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_bootstrap_ci_is_reproducible_and_brackets_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300); s = y * 0.6 + rng.normal(0, 0.5, 300)
    a = bootstrap_ci(roc_auc_score, y, s, n_boot=300, seed=7)
    b = bootstrap_ci(roc_auc_score, y, s, n_boot=300, seed=7)
    assert a == b and a[1] <= a[0] <= a[2]


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(2)
    def width(n):
        y = rng.integers(0, 2, n); s = y * 0.6 + rng.normal(0, 0.5, n)
        _, lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=300, seed=0)
        return hi - lo
    assert width(2000) < width(100)
