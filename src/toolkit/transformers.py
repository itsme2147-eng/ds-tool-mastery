"""Leakage-safe scikit-learn transformers.

Both transformers learn their parameters in ``fit`` (on training data only) and apply them
unchanged in ``transform``, so they are safe inside cross-validation pipelines.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature of every pair whose absolute Pearson correlation exceeds ``threshold``.

    Of each correlated pair, the feature appearing later in the column order is dropped.
    Highly redundant features (e.g. radius, perimeter and area) add variance without information.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        if not 0 < self.threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        df = pd.DataFrame(X)
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        self.columns_in_ = list(df.columns)
        self.to_drop_ = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        self.keep_ = [c for c in self.columns_in_ if c not in self.to_drop_]
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_")
        if isinstance(X, pd.DataFrame):
            return X[self.keep_]
        arr = np.asarray(X)
        idx = [self.columns_in_.index(c) for c in self.keep_]
        return arr[:, idx]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "keep_")
        return np.asarray([str(c) for c in self.keep_], dtype=object)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to the [lower, upper] quantiles learned from the training data."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("need 0 <= lower < upper <= 1")
        arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(arr, self.lower, axis=0)
        self.hi_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "lo_")
        arr = np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(arr, index=X.index, columns=X.columns)
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None
                          else [f"x{i}" for i in range(self.n_features_in_)], dtype=object)
