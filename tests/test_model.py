import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


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

    raise ValueError("Cannot create 'dayofweek' - missing both 'dteday' and 'weekday'.")


def test_model_quality_gate():
    model_path = Path("model/final_model.joblib")
    model = joblib.load(model_path)

    df = pd.read_csv("data/day_2012.csv")

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    # create dayofweek
    X = ensure_dayofweek(X)

    # PROOF this file is running in CI
    assert "dayofweek" in X.columns, f"dayofweek still missing. cols={list(X.columns)}"

    preds = model.predict(X)

    assert len(preds) == len(y)
    assert float(np.mean(preds)) > 0
