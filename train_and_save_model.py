# train_and_save_model.py

import joblib
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ✅ IMPORTANT: custom transformer must be in an importable module
from src.transformers import DayOfWeekAdder


# -----------------------------
# Feature lists
# -----------------------------
CAT_FEATURES = ["season", "mnth", "weekday", "weathersit"]
NUM_FEATURES = ["holiday", "workingday", "temp", "atemp", "hum", "windspeed", "dayofweek"]
TARGET = "cnt"


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
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    return pipe


def main():
    df = pd.read_csv("data/day_2012.csv")

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # 🔑 CRITICAL: create dayofweek BEFORE fit so sklearn feature check passes
    X = DayOfWeekAdder().transform(X)

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
