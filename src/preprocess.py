
import pandas as pd

DATE_COL = "dteday"

def add_dayofweek(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matches your notebook (Cell 12 / Cell 37):
    - parse dteday (dd/mm/YYYY)
    - create dayofweek (0=Mon ... 6=Sun)
    - drop dteday
    """
    df = df.copy()

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
        df["dayofweek"] = df[DATE_COL].dt.dayofweek
        df = df.drop(columns=[DATE_COL])

    return df
