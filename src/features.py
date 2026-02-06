# =========================
# File: src/features.py
# =========================

CATEGORICAL_FEATURES = ["season", "mnth", "weekday", "weathersit"]

BASE_NUMERIC_FEATURES = [
    "holiday",
    "workingday",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

DERIVED_DATE_FEATURES = ["dayofweek", "dayofyear", "weekofyear"]

# Used by ColumnTransformer AFTER DateFeatureAdder runs
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_DATE_FEATURES

TARGET = "cnt"
DATE_COL = "dteday"

def all_features():
    # Used BEFORE the pipeline runs - must include dteday, must exclude derived cols
    return [DATE_COL] + CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
