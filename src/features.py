# src/features.py

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

# What the ColumnTransformer should see AFTER DateFeatureAdder runs
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + DERIVED_DATE_FEATURES

TARGET = "cnt"
DATE_COL = "dteday"

def all_features():
    """
    Use this for selecting columns from the RAW dataframe before the pipeline.
    Must include dteday so DateFeatureAdder can create derived date features.
    Must NOT include derived date features (they don't exist yet).
    """
    return [DATE_COL] + CATEGORICAL_FEATURES + BASE_NUMERIC_FEATURES
