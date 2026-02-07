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
        df = X.copy()

        if self.date_col in df.columns:
            d = pd.to_datetime(df[self.date_col], errors="coerce")

            df["dayofweek"] = d.dt.dayofweek
            df["dayofyear"] = d.dt.dayofyear

            # ✅ SAFE handling for weekofyear
            week = d.dt.isocalendar().week
            df["weekofyear"] = week.astype("Int64").fillna(0).astype("int64")

            if self.drop:
                df = df.drop(columns=[self.date_col])

        else:
            # hard fallback (should not normally trigger)
            df["dayofweek"] = 0
            df["dayofyear"] = 0
            df["weekofyear"] = 0

        return df
