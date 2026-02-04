import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_baseline_rmse(path="metrics/baseline_metrics.json"):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj["baseline_rmse"])


def main():
    # -------- Paths (repo-relative) --------
    MODEL_PATH = os.getenv("MODEL_PATH", "model/final_model.joblib")
    DATA_PATH = os.getenv("EVAL_DATA_PATH", "data/day_2011.csv")  # you can switch to 2012 too
    BASELINE_PATH = os.getenv("BASELINE_METRICS_PATH", "metrics/baseline_metrics.json")

    # Gate settings
    # Example from brief: assert rmse <= 0.95 * rmse_baseline :contentReference[oaicite:2]{index=2}
    GATE_RATIO = float(os.getenv("GATE_RATIO", "0.95"))  # stricter if 0.95, looser if >1.0

    assert os.path.exists(MODEL_PATH), f"Missing model file: {MODEL_PATH}"
    assert os.path.exists(DATA_PATH), f"Missing eval data file: {DATA_PATH}"
    assert os.path.exists(BASELINE_PATH), f"Missing baseline metrics: {BASELINE_PATH}"

    # -------- Load --------
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    # -------- Basic cleaning checks --------
    assert "cnt" in df.columns, "Target column cnt not found"
    assert df.isna().sum().sum() == 0, "Dataset contains missing values - clean/impute before testing"

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    # If you still have raw date column, drop it here (align with your pipeline expectations)
    if "dteday" in X.columns:
        X = X.drop(columns=["dteday"])

    # -------- Predict --------
    y_pred = model.predict(X)
    test_rmse = rmse(y, y_pred)

    # -------- Quality Gate --------
    baseline_rmse = load_baseline_rmse(BASELINE_PATH)
    threshold = GATE_RATIO * baseline_rmse

    # Extra: log artifacts locally (GitHub Actions can upload)
    os.makedirs("outputs", exist_ok=True)

    out_metrics = {
        "eval_data": DATA_PATH,
        "rmse": test_rmse,
        "baseline_rmse": baseline_rmse,
        "gate_ratio": GATE_RATIO,
        "threshold": threshold,
        "passed": bool(test_rmse <= threshold),
    }
    with open("outputs/metrics.json", "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    # Residual plot artifact (nice evidence)
    residuals = y - y_pred
    plt.figure()
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted cnt")
    plt.ylabel("Residuals (y - y_pred)")
    plt.title("Residual Plot - Quality Gate Run")
    plt.savefig("outputs/residuals.png", bbox_inches="tight")
    plt.close()

    print(f"[Quality Gate] RMSE={test_rmse:.2f} | Baseline RMSE={baseline_rmse:.2f} | Threshold={threshold:.2f}")

    # Fail the workflow if gate not met
    assert test_rmse <= threshold, (
        f"FAILED quality gate: rmse {test_rmse:.2f} > threshold {threshold:.2f} "
        f"(gate_ratio={GATE_RATIO}, baseline_rmse={baseline_rmse:.2f})"
    )


if __name__ == "__main__":
    main()
