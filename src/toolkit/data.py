"""Dataset loaders (real, public data). See data/SOURCE.md for provenance."""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_penguins() -> pd.DataFrame:
    """Palmer penguins (344 rows, a few missing values)."""
    return pd.read_csv(DATA_DIR / "penguins.csv")


def load_flights() -> pd.DataFrame:
    """Monthly airline passengers 1949-1960 in long format (year, month, passengers)."""
    return pd.read_csv(DATA_DIR / "flights.csv")


def load_cancer() -> tuple[pd.DataFrame, pd.Series]:
    """UCI Wisconsin Diagnostic Breast Cancer data.

    Returns X (569 x 30) and y where **1 = malignant** (the positive class we must not miss)
    and 0 = benign. scikit-learn encodes it the other way round, so we flip it here.
    """
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer(as_frame=True)
    y = (raw.target == 0).astype(int).rename("malignant")
    return raw.data.copy(), y
