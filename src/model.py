# =========================
# File: src/model.py
# =========================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.preprocess import add_dayofweek
from src.pipeline import build_preprocessor
from src.features import TARGET, all_features

def build_model() -> Pipeline:
    """
    End-to-end sklearn pipeline:
    preprocessing + RandomForestRegressor
    """
    preprocessor = build_preprocessor()
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

def make_X_y(df: pd.DataFrame):
    """
    For training/evaluation.
    Ensures dayofweek exists before selecting features.
    """
    df = add_dayofweek(df)

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    feats = all_features()
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {set(missing)}")

    X = df[feats].copy()
    y = df[TARGET].copy()
    return X, y

def make_X(df: pd.DataFrame):
    """
    For inference.
    Creates dayofweek and returns feature-only dataframe.
    Drops target if present.
    """
    df = add_dayofweek(df)

    feats = all_features()
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {set(missing)}")

    X = df[feats].copy()
    return X

def train(df: pd.DataFrame) -> Pipeline:
    """
    Convenience wrapper if your tests expect a train() function.
    """
    X, y = make_X_y(df)
    pipe = build_model()
    pipe.fit(X, y)
    return pipe

def predict(pipe: Pipeline, df: pd.DataFrame):
    """
    Convenience wrapper if your tests expect a predict() function.
    """
    X = make_X(df)
    return pipe.predict(X)
