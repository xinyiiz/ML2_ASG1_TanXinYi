import os
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------
# 1) Custom transformer
# -----------------------------
class DayOfWeekAdder(BaseEstimator, TransformerMixin):
    """
    Makes sure 'dayofweek' exists for the model.

    Input can contain:
    - dayofweek already
    - or dteday (dd/mm/YYYY)
    - or weekday (0-6)

    Output:
    - always has dayofweek
    - drops dteday (so the rest of the pipeline won't choke on a raw date string)
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "dayofweek" in X.columns:
            # If dteday exists too, drop it to keep features consistent
            if "dteday" in X.columns:
                X = X.drop(columns=["dteday"])
            return X

        if "dteday" in X.columns:
            X["dteday"] = pd.to_datetime(X["dteday"], format="%d/%m/%Y", errors="coerce")
            X["dayofweek"] = X["dteday"].dt.dayofweek
            X = X.drop(columns=["dteday"])
            return X

        if "weekday" in X.columns:
            X["dayofweek"] = X["weekday"]
            return X

        raise ValueError("Cannot create 'dayofweek' - missing both 'dteday' and 'weekday'.")


# -----------------------------
# 2) Feature lists (matches your notebook intent)
# -----------------------------
CAT_FEATURES = ["season", "mnth", "weekday", "weathersit"]
NUM_FEATURES = ["holiday", "workingday", "temp", "atemp", "hum", "windspeed", "dayofweek"]
TARGET = "cnt"


# -----------------------------
# 3) Build pipeline
# -----------------------------
def build_pipeline():
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_FEATURES),
            ("num", num_pipe, NUM_FEATURES),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("add_dayofweek", DayOfWeekAdder()),
        ("preprocess", preprocessor),
        ("model", model),
    ])

    return pipe


# -----------------------------
# 4) Train and save
# -----------------------------
def main():
    data_path = Path("data/day_2012.csv")  # train on 2012 like your test uses
    df = pd.read_csv(data_path)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    pipe = build_pipeline()
    pipe.fit(X, y)

    out_dir = Path("model")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "final_model.joblib"
    joblib.dump(pipe, out_path)

    print("Saved model to:", out_path.resolve())
    print("Columns in training X:", list(X.columns))


if __name__ == "__main__":
    main()
