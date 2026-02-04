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
