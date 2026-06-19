"""
Input/output helpers for loading raw data files.
"""

import pandas as pd

def _load_renewables_ninja(path: str, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")

    time_col = "local_time" if "local_time" in df.columns else "time"
    if time_col not in df.columns:
        raise ValueError(f"Missing time column in {path}")

    source_cols = ("electricity", value_col, value_col.replace("_per_kw", "_kw"))
    data_col = next((col for col in source_cols if col in df.columns), None)
    if data_col is None:
        raise ValueError(f"Missing generation column in {path}")

    df = df[[time_col, data_col]].copy()
    df = df.rename(columns={time_col: "timestamp", data_col: value_col})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    return df


def load_wind_raw(path: str) -> pd.DataFrame:
    """
    Load raw wind generation data from Renewables.ninja.

    Returns a DataFrame with 'timestamp' and 'wind_per_kw' columns.
    """
    return _load_renewables_ninja(path, "wind_per_kw")


def load_pv_raw(path: str) -> pd.DataFrame:
    """
    Load raw PV generation data from Renewables.ninja.

    Returns a DataFrame with 'timestamp' and 'pv_per_kw' columns.
    """
    return _load_renewables_ninja(path, "pv_per_kw")


def load_prices_raw(path: str) -> pd.DataFrame:
    """
    Load raw day-ahead prices from SMARD export.

    Returns a DataFrame with 'timestamp' and 'price_eur_per_mwh' columns.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    if df.shape[1] == 1:
        df = pd.read_csv(path)

    time_col = next((col for col in df.columns if col.lower().startswith("start date")), None)
    price_col = next((col for col in df.columns if "Germany/Luxembourg" in col), None)
    if time_col is None or price_col is None:
        raise ValueError("Price file is missing required columns")

    df = df[[time_col, price_col]].copy()
    df = df.rename(columns={time_col: "timestamp", price_col: "price_eur_per_mwh"})
    try:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%b %d, %Y %I:%M %p"
        )
    except ValueError:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price_eur_per_mwh"] = (
        df["price_eur_per_mwh"]
        .astype(str)
        .str.replace(",", ".")
    )
    df["price_eur_per_mwh"] = pd.to_numeric(df["price_eur_per_mwh"], errors="coerce")
    return df
