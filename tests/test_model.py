import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """
    CI-safe feature fix:
    - If dayofweek exists, keep it.
    - Else derive from dteday (if present), else map from weekday (if present).
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

    raise ValueError("Cannot create dayofweek - missing both dteday and weekday.")


def test_model_quality_gate():
    # --- Load model ---
    model_path = Path("model/final_model.joblib")
    assert model_path.exists(), f"Missing model file: {model_path}"
    model = joblib.load(model_path)

    # --- Load evaluation data (use 2011 so the gate reflects expected performance) ---
    df = pd.read_csv("data/day_2011.csv")
    assert "cnt" in df.columns, "Target column cnt missing from evaluation data."

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    # Fix feature mismatch
    X = ensure_dayofweek(X)
    assert "dayofweek" in X.columns, f"dayofweek still missing. cols={list(X.columns)}"

    # Consistent holdout split for fair, repeatable CI
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = model.predict(X_test)
    assert len(preds) == len(y_test)
    assert float(np.mean(preds)) > 0  # basic sanity

    # --- Quality Gate (threshold) ---
    # Baseline RMSE from your Task 1 Linear Regression run
    BASELINE_RMSE = 630.16
    THRESHOLD = 0.95 * BASELINE_RMSE  # example suggested by brief

    model_rmse = rmse(y_test, preds)

    # The actual pass/fail gate for GitHub Actions
    assert model_rmse <= THRESHOLD, (
        f"QUALITY GATE FAILED: rmse={model_rmse:.2f} > threshold={THRESHOLD:.2f}"
    )
