# src/transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, date_col="dteday", drop=True):
        self.date_col = date_col
        self.drop = drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if X is None:
            raise ValueError("DateFeatureAdder received None input")

        df = X.copy()

        # d is only available if date_col exists
        d = None
        if self.date_col in df.columns:
            d = pd.to_datetime(df[self.date_col], errors="coerce")

        # ---- dayofweek ----
        if "dayofweek" not in df.columns:
            if d is not None:
                df["dayofweek"] = d.dt.dayofweek.fillna(0).astype(int)
            elif "weekday" in df.columns:
                df["dayofweek"] = df["weekday"]
            else:
                df["dayofweek"] = 0

        # ---- dayofyear ----
        if "dayofyear" not in df.columns:
            if d is not None:
                df["dayofyear"] = d.dt.dayofyear.fillna(0).astype(int)
            else:
                df["dayofyear"] = 0

        # ---- weekofyear ----
        if "weekofyear" not in df.columns:
            if d is not None:
                week = d.dt.isocalendar().week
                df["weekofyear"] = week.astype("Int64").fillna(0).astype(int)
            else:
                df["weekofyear"] = 0

        # Drop raw date column if present
        if self.drop and self.date_col in df.columns:
            df = df.drop(columns=[self.date_col])

        return df
