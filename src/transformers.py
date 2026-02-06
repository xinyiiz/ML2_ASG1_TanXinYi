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
        # Expect a pandas DataFrame
        df = X.copy()

        # Convert date column safely
        if self.date_col in df.columns:
            d = pd.to_datetime(df[self.date_col], errors="coerce")
            df["dayofweek"] = d.dt.dayofweek
            df["dayofyear"] = d.dt.dayofyear
            df["weekofyear"] = d.dt.isocalendar().week.astype("int64")

            if self.drop:
                df = df.drop(columns=[self.date_col])
        else:
            # If dteday is missing, still ensure required cols exist
            for col in ["dayofweek", "dayofyear", "weekofyear"]:
                if col not in df.columns:
                    df[col] = 0

        return df
