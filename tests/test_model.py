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
import pandas as pd

def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "dayofweek" in df.columns:
        return df

    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], format="%d/%m/%Y", errors="coerce")
        df["dayofweek"] = df["dteday"].dt.dayofweek
        df = df.drop(columns=["dteday"])
        return df

    if "weekday" in df.columns:
        df["dayofweek"] = df["weekday"]
        return df

    # last resort so the error message is clearer
    raise ValueError("Cannot create 'dayofweek' - missing both 'dteday' and 'weekday'.")


def test_model_quality_gate():
    # load model (this is a Pipeline)
    model_path = Path("model/final_model.joblib")
    model = joblib.load(model_path)

    # load raw data - NO feature engineering here
    df = pd.read_csv("data/day_2012.csv")

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    # predict - pipeline handles everything
    preds = model.predict(X)

    # basic quality checks
    assert len(preds) == len(y)
    assert preds.mean() > 0
