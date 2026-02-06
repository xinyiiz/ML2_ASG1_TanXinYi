# =========================
# File: src/model.py
# =========================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from src.pipeline import build_pipeline
from src.features import TARGET, all_features


def build_model() -> Pipeline:
    """
    End-to-end sklearn pipeline:
    DateFeatureAdder -> preprocessing -> RandomForestRegressor
    """
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    return build_pipeline(model)


def make_X_y(df: pd.DataFrame):
    """
    For training/evaluation.

    IMPORTANT:
    - Do NOT create engineered date columns here.
    - all_features() should return RAW input features that exist in the CSV,
      including 'dteday' (so DateFeatureAdder can derive dayofyear/weekofyear).
    """
    df = df.copy()

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

    Returns RAW input features only.
    Date features will be created inside the pipeline by DateFeatureAdder.
    """
    df = df.copy()

    # If target exists in inference input, ignore it safely
    if TARGET in df.columns:
        df = df.drop(columns=[TARGET])

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
