# =========================
# File: src/pipeline.py
# =========================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

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
