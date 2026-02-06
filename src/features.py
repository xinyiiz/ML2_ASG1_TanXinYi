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

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_DATE_FEATURES

TARGET = "cnt"

def all_features():
    # These must exist in df BEFORE selecting X
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES
