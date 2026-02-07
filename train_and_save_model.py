# train_and_save_model.py
from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from src.transformers import DateFeatureAdder


def build_pipeline():
    numeric_features = [
        "holiday",
        "workingday",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "dayofweek",
        "dayofyear",
        "weekofyear",
    ]

    categorical_features = [
        "season",
        "mnth",
        "weekday",
        "weathersit",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # ✅ ensure ColumnTransformer returns pandas (keeps feature names)
    preprocessor.set_output(transform="pandas")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    date_step = DateFeatureAdder(date_col="dteday", drop=True)
    # ✅ ensure DateFeatureAdder outputs pandas (keeps column names)
    date_step.set_output(transform="pandas")

    pipe = Pipeline(
        steps=[
            ("date_features", date_step),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe


def main():
    data_path = Path("data/day_2011.csv")  # change to your real file if needed
    model_path = Path("model/final_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    if "cnt" not in df.columns:
        raise ValueError("Target column 'cnt' not found in dataset.")

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    pipe = build_pipeline()
    pipe.fit(X, y)

    joblib.dump(pipe, model_path)
    print(f"Saved model to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
