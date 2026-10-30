import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from toolkit import CorrelationFilter, Winsorizer


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    return pd.DataFrame({"a": a, "a_copy": a * 3 + 1, "b": rng.normal(size=200), "c": rng.normal(size=200)})


def test_correlation_filter_drops_redundant_feature(frame):
    f = CorrelationFilter(0.95).fit(frame)
    assert f.to_drop_ == ["a_copy"]            # later column of the correlated pair is dropped
    assert list(f.transform(frame).columns) == ["a", "b", "c"]


def test_correlation_filter_keeps_all_when_threshold_is_one(frame):
    out = CorrelationFilter(1.0).fit_transform(frame.drop(columns="a_copy"))
    assert out.shape[1] == 3


def test_correlation_filter_learns_only_from_fit_data(frame):
    f = CorrelationFilter(0.95).fit(frame)
    other = frame.copy(); other["b"] = other["c"]            # new data is perfectly correlated...
    assert list(f.transform(other).columns) == ["a", "b", "c"]  # ...but the learned columns do not change


def test_correlation_filter_works_with_numpy_and_clone(frame):
    f = clone(CorrelationFilter(0.9)).fit(frame.to_numpy())
    assert f.transform(frame.to_numpy()).shape == (200, 3)


def test_correlation_filter_validates_threshold(frame):
    with pytest.raises(ValueError):
        CorrelationFilter(0).fit(frame)


def test_get_feature_names_out(frame):
    f = CorrelationFilter().fit(frame)
    assert list(f.get_feature_names_out()) == ["a", "b", "c"]


def test_winsorizer_clips_to_training_quantiles():
    train = np.arange(101, dtype=float).reshape(-1, 1)
    w = Winsorizer(0.05, 0.95).fit(train)
    out = w.transform(np.array([[-1000.0], [50.0], [1000.0]]))
    assert out.ravel().tolist() == [5.0, 50.0, 95.0]


def test_winsorizer_preserves_dataframe_and_index():
    df = pd.DataFrame({"x": [1.0, 2.0, 100.0]}, index=[10, 11, 12])
    out = Winsorizer(0.0, 0.5).fit_transform(df)
    assert isinstance(out, pd.DataFrame) and list(out.index) == [10, 11, 12]


def test_transformers_compose_in_a_pipeline(frame):
    pipe = make_pipeline(CorrelationFilter(), Winsorizer(0.01, 0.99), StandardScaler())
    out = pipe.fit_transform(frame)
    assert out.shape == (200, 3) and abs(out.mean()) < 1e-9
