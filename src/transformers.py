import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "dteday" in X.columns:
            X["dteday"] = pd.to_datetime(X["dteday"])
            X["dayofweek"] = X["dteday"].dt.dayofweek
            X["dayofyear"] = X["dteday"].dt.dayofyear
            X["weekofyear"] = X["dteday"].dt.isocalendar().week.astype(int)
            X = X.drop(columns=["dteday"])

        return X
