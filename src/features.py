CATEGORICAL_FEATURES = ["season", "mnth", "weekday", "weathersit"]
NUMERIC_FEATURES = ["holiday", "workingday", "temp", "atemp", "hum", "windspeed", "dayofweek"]

TARGET = "cnt"

def all_features():
    return CATEGORICAL_FEATURES + NUMERIC_FEATURES
