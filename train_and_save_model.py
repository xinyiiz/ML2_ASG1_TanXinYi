# train_and_save_model.py
from pathlib import Path
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

from src.transformers import DateFeaturesAdder


def build_pipeline():
    # After DateFeaturesAdder runs, dteday is dropped and these exist:
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

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline(
        steps=[
            # FIRST - make engineered cols available for the ColumnTransformer
            ("date_features", DateFeaturesAdder(date_col="dteday")),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipe


def main():
    data_path = Path("data/day_2011.csv")
    model_path = Path("model/final_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    y = df["cnt"]
    X = df.drop(columns=["cnt"])

    pipe = build_pipeline()
    pipe.fit(X, y)

    joblib.dump(pipe, model_path)
    print(f"Saved model to: {model_path.resolve()}")


if __name__ == "__main__":
    main()
