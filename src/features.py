# =========================
# File: src/features.py
# =========================

CATEGORICAL_FEATURES = ["season", "mnth", "weekday", "weathersit"]

# Raw numeric columns that already exist in the CSV
BASE_NUMERIC_FEATURES = [
    "holiday",
    "workingday",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]

# Derived numeric columns created from dteday inside DateFeatureAdder
DERIVED_DATE_FEATURES = ["dayofweek", "dayofyear", "weekofyear"]

# What the preprocessor should select AFTER DateFeatureAdder runs
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_DATE_FEATURES

TARGET = "cnt"
DATE_COL = "dteday"

def all_features():
    """
    Features to select from the RAW dataframe BEFORE the pipeline runs.
    Must include dteday so DateFeatureAdder can create derived date features.
    Must NOT include derived date features (they don't exist yet).
    """
    return [DATE_COL] + CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
