import json
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def test_model_quality_gate():
    # Paths inside the repo
    MODEL_PATH = "model/final_model.joblib"
    DATA_PATH = "data/day_2011.csv"

    # Baseline RMSE - use YOUR baseline from notebook (example: 637.80)
    BASELINE_RMSE = 637.80

    # Gate rule (stricter = lower ratio)
    GATE_RATIO = 0.95
    THRESHOLD = GATE_RATIO * BASELINE_RMSE

    assert os.path.exists(MODEL_PATH), f"Missing model: {MODEL_PATH}"
    assert os.path.exists(DATA_PATH), f"Missing data: {DATA_PATH}"

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    assert "cnt" in df.columns, "Target column 'cnt' not found"

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    # If your csv still has dteday, drop it (your pipeline usually expects no raw date)
    if "dteday" in X.columns:
        X = X.drop(columns=["dteday"])

    preds = model.predict(X)
    score = rmse(y, preds)

    # Print shows up in GitHub Actions logs
    print(f"RMSE={score:.2f} | Threshold={THRESHOLD:.2f} (baseline={BASELINE_RMSE:.2f})")

    # Quality gate
    assert score <= THRESHOLD, f"FAILED quality gate: RMSE {score:.2f} > {THRESHOLD:.2f}"
