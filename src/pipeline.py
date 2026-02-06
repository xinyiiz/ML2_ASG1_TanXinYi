# =========================
# File: src/pipeline.py
# =========================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.transformers import DateFeatureAdder


def build_preprocessor() -> ColumnTransformer:
    """
    Mirrors your Task 1 preprocessing idea:
    - OHE for categoricals
    - StandardScaler for numerics
    - Imputation for safety
    """
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", num_pipe, NUMERIC_FEATURES),
        ]
    )


def build_pipeline(model) -> Pipeline:
    """
    Full end-to-end pipeline:
    1) Create dayofweek/dayofyear/weekofyear from dteday (and drop dteday)
    2) Preprocess columns (OHE + scaling + imputation)
    3) Train/predict with model
    """
    preprocessor = build_preprocessor()

    return Pipeline([
        ("date_features", DateFeatureAdder()),
        ("preprocessor", preprocessor),
        ("model", model),
    ])
