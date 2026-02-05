import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DayOfWeekAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "dayofweek" in X.columns:
            if "dteday" in X.columns:
                X = X.drop(columns=["dteday"])
            return X

        if "dteday" in X.columns:
            X["dteday"] = pd.to_datetime(
                X["dteday"], format="%d/%m/%Y", errors="coerce"
            )
            X["dayofweek"] = X["dteday"].dt.dayofweek
            X = X.drop(columns=["dteday"])
            return X

        if "weekday" in X.columns:
            X["dayofweek"] = X["weekday"]
            return X

        raise ValueError("Cannot create 'dayofweek'")
