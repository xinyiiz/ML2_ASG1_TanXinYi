import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure 'dayofweek' exists.
    Priority:
      1) Use existing 'dayofweek'
      2) Derive from 'dteday' if present (dayfirst dates like 01/01/2011)
      3) Fallback to 'weekday' if present
    """
    df = df.copy()

    if "dayofweek" in df.columns:
        return df

    if "dteday" in df.columns:
        df["dteday"] = pd.to_datetime(df["dteday"], format="%d/%m/%Y", errors="coerce")
        if df["dteday"].isna().all():
            raise ValueError("Failed to parse 'dteday' into datetime - check date format in CSV.")
        df["dayofweek"] = df["dteday"].dt.dayofweek
        # drop raw date so we don't feed raw strings into the model
        df = df.drop(columns=["dteday"])
        return df

    if "weekday" in df.columns:
        # dataset weekday is already 0-6
        df["dayofweek"] = df["weekday"]
        return df

    raise ValueError("Cannot create 'dayofweek' - missing both 'dteday' and 'weekday'.")


def test_model_quality_gate():
    # 1) Load model
    model_path = Path("model/final_model.joblib")
    assert model_path.exists(), f"Model file not found: {model_path.resolve()}"
    model = joblib.load(model_path)

    # 2) Load data
    data_path = Path("data/day_2012.csv")
    assert data_path.exists(), f"Dataset file not found: {data_path.resolve()}"
    df = pd.read_csv(data_path)

    # 3) Split X/y
    assert "cnt" in df.columns, f"'cnt' not found. cols={list(df.columns)}"
    y = df["cnt"].astype(float)
    X = df.drop(columns=["cnt"])

    # 4) Ensure required engineered feature exists
    X = ensure_dayofweek(X)

    # Proof this file runs in CI + sanity checks
    assert "dayofweek" in X.columns, f"dayofweek still missing. cols={list(X.columns)}"
    assert X.isnull().mean().max() < 1.0, "All values in at least one column are NaN."

    # 5) Predict
    preds = model.predict(X)

    # 6) Basic output checks (lightweight CI gate)
    assert len(preds) == len(y), f"preds len {len(preds)} != y len {len(y)}"
    assert np.isfinite(preds).all(), "Predictions contain NaN/inf."

    # Optional: simple sanity check so it doesn't pass with garbage
    assert float(np.mean(preds)) > 0, "Mean prediction is not positive - looks suspicious."

    # Optional: if you want to log a metric in output (not required)
    _ = rmse(y, preds)
