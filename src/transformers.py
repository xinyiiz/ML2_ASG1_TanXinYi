# src/transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeaturesAdder(BaseEstimator, TransformerMixin):
    """
    Adds dayofweek, dayofyear, weekofyear from dteday (dd/mm/YYYY).
    Drops dteday afterwards so models don't see raw date strings.
    """

    def __init__(self, date_col: str = "dteday"):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # If date column exists, compute date features
        if self.date_col in X.columns:
            dt = pd.to_datetime(X[self.date_col], format="%d/%m/%Y", errors="coerce")

            if "dayofweek" not in X.columns:
                X["dayofweek"] = dt.dt.dayofweek

            if "dayofyear" not in X.columns:
                X["dayofyear"] = dt.dt.dayofyear

            if "weekofyear" not in X.columns:
                # ISO week number (1-53)
                X["weekofyear"] = dt.dt.isocalendar().week.astype("int64")

            # drop raw date col
            X = X.drop(columns=[self.date_col])

        else:
            # If tests ever provide engineered columns only, just ensure they exist
            missing = [c for c in ["dayofweek", "dayofyear", "weekofyear"] if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing {missing}. Expected either '{self.date_col}' or all engineered date columns."
                )

        return X
