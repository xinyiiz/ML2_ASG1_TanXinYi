# =========================
# File: src/preprocess.py
# =========================
import pandas as pd

DATE_COL = "dteday"

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures df has dayofweek, dayofyear, weekofyear.
    Does NOT drop dteday (safe for other code paths).
    """
    df = df.copy()

    # If we have dteday, derive from it
    if DATE_COL in df.columns:
        d = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["dayofweek"] = d.dt.dayofweek
        df["dayofyear"] = d.dt.dayofyear
        df["weekofyear"] = d.dt.isocalendar().week.astype("int64")
        return df

    # Fallbacks if date is missing
    if "dayofweek" not in df.columns:
        if "weekday" in df.columns:
            df["dayofweek"] = df["weekday"]
        else:
            df["dayofweek"] = 0

    if "dayofyear" not in df.columns:
        df["dayofyear"] = 0

    if "weekofyear" not in df.columns:
        df["weekofyear"] = 0

    return df
