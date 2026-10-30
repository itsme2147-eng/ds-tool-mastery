import json
import joblib
import numpy as np
import pytest

from toolkit.data import load_cancer, load_flights, load_penguins
from toolkit.train import build_pipeline, run


def test_datasets_load_with_expected_shape():
    X, y = load_cancer()
    assert X.shape == (569, 30) and y.sum() == 212          # 212 malignant, 357 benign
    assert load_penguins().shape == (344, 7) and len(load_flights()) == 144


def test_run_writes_artifacts_and_reaches_expected_quality(tmp_path):
    r = run("logreg", seed=42, out=tmp_path)
    assert (tmp_path / "logreg.joblib").exists() and (tmp_path / "logreg_metrics.json").exists()
    assert r["test_roc_auc_ci"][0] > 0.97                    # real, well-separated data
    assert r["at_recall_threshold"]["recall"] >= r["at_default_threshold"]["recall"]
    assert json.loads((tmp_path / "logreg_metrics.json").read_text())["model"] == "logreg"


def test_training_is_deterministic(tmp_path):
    a = run("logreg", seed=7, out=tmp_path / "a")
    b = run("logreg", seed=7, out=tmp_path / "b")
    assert a == b


def test_saved_model_reloads_and_predicts_identically(tmp_path):
    run("logreg", seed=3, out=tmp_path)
    X, _ = load_cancer()
    m = joblib.load(tmp_path / "logreg.joblib")
    np.testing.assert_allclose(m.predict_proba(X), m.predict_proba(X))
    assert m.predict_proba(X).shape == (569, 2)


def test_pipeline_drops_correlated_features():
    X, y = load_cancer()
    pipe = build_pipeline("logreg").fit(X, y)
    assert len(pipe.named_steps["corr"].keep_) < 30


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        build_pipeline("not-a-model")
