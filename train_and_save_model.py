import joblib
import pandas as pd
from pathlib import Path

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
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        if "dayofweek" in X.columns:
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

        raise ValueError("Cannot create 'dayofweek'")


# -----------------------------
# 2) Feature lists
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

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


# -----------------------------
# 4) Train and save
# -----------------------------
def main():
    df = pd.read_csv("data/day_2012.csv")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # 🔑 CRITICAL FIX: add dayofweek BEFORE fitting
    X = DayOfWeekAdder().transform(X)

    pipe = build_pipeline()
    pipe.fit(X, y)

    out_dir = Path("model")
    out_dir.mkdir(exist_ok=True)

    joblib.dump(pipe, out_dir / "final_model.joblib")
    print("Model saved successfully with dayofweek in feature_names_in_")
    print("Training columns:", list(X.columns))


if __name__ == "__main__":
    main()
