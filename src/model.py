# =========================
# File: src/model.py
# =========================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.pipeline import build_pipeline
from src.features import TARGET, all_features
from src.preprocess import add_date_features


def build_model() -> Pipeline:
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    return build_pipeline(model)


def make_X_y(df: pd.DataFrame):
    df = add_date_features(df)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    feats = all_features()
    X = df[feats].copy()
    y = df[TARGET].copy()
    return X, y


def make_X(df: pd.DataFrame):
    df = add_date_features(df)
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

    feats = all_features()
    X = df[feats].copy()
    return X


def train(df: pd.DataFrame) -> Pipeline:
    X, y = make_X_y(df)
    pipe = build_model()
    pipe.fit(X, y)
    return pipe


def predict(pipe: Pipeline, df: pd.DataFrame):
    X = make_X(df)
    return pipe.predict(X)
