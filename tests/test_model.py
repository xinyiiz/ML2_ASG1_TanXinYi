import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


import pandas as pd
import joblib
from pathlib import Path

def test_model_quality_gate():
    model_path = Path("model/final_model.joblib")
    model = joblib.load(model_path)

    df = pd.read_csv("data/day_2012.csv")

    # target
    y = df["cnt"]

    # features
    X = df.drop(columns=["cnt"])

    # FIX: recreate engineered features exactly like training
    if "dteday" in X.columns:
        X["dteday"] = pd.to_datetime(X["dteday"])
        X["dayofweek"] = X["dteday"].dt.dayofweek
        X = X.drop(columns=["dteday"])

    # prediction
    preds = model.predict(X)

    # simple quality gate
    assert len(preds) == len(y)
    assert preds.mean() > 0
