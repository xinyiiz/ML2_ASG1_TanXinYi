# tests/test_model.py

import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has 'dayofweek' before sending into the saved pipeline.
    Priority:
      1) keep existing dayofweek
      2) derive from dteday (dd/mm/YYYY) and drop dteday
      3) fallback: copy from weekday
    """
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

    raise ValueError("Cannot create 'dayofweek' - missing both 'dteday' and 'weekday'.")


def test_model_quality_gate():
    # load model (this is a Pipeline)
    model_path = Path("model/final_model.joblib")
    assert model_path.exists(), f"Missing model file: {model_path}"
    model = joblib.load(model_path)

    # load raw data - NO feature engineering here
    df = pd.read_csv("data/day_2012.csv")
    assert "cnt" in df.columns, "Target column 'cnt' missing from data/day_2012.csv"

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    # ✅ IMPORTANT: make sure 'dayofweek' exists before predict()
    X = ensure_dayofweek(X)

    # predict - pipeline handles everything else
    preds = model.predict(X)

    # basic quality checks
    assert len(preds) == len(y)
    assert float(np.mean(preds)) > 0
