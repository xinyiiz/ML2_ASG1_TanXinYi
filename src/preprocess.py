# src/preprocess.py
import pandas as pd

DATE_COL = "dteday"

def add_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has 'dayofweek' no matter what the input looks like.

    Priority:
    1) If dayofweek already exists, keep it
    2) Else if dteday exists, parse + derive dayofweek
    3) Else if weekday exists, copy it into dayofweek
    """
    df = df.copy()

    # 1) already present
    if "dayofweek" in df.columns:
        return df

    # 2) derive from date if possible
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
        df["dayofweek"] = df[DATE_COL].dt.dayofweek
        df = df.drop(columns=[DATE_COL])
        return df

    # 3) fallback - many datasets have weekday already
    if "weekday" in df.columns:
        df["dayofweek"] = df["weekday"]
        return df

    raise ValueError("Cannot create 'dayofweek' - missing both 'dteday' and 'weekday'.")
