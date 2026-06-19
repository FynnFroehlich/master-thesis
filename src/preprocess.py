"""
Data preprocessing functions to clean and align time series data.
"""

import pandas as pd
from typing import Optional

from config import Config, DATA_PROCESSED_DIR
from data_io import load_wind_raw, load_pv_raw, load_prices_raw


def _prepare_series(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column for {value_col}")

    df = df[["timestamp", value_col]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df["hour_of_year"] = range(1, len(df) + 1)
    return df


def build_clean_dataset(
    wind_path: str,
    pv_path: str,
    price_path: str,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """
    Build a clean, aligned dataset from raw wind, PV, and price data.

    The inputs can come from different years; alignment is done by hour-of-year,
    using the price timestamps as the reference timeline. Leap days are removed
    so all series normalize to 8760 hours.
    """
    if config is None:
        config = Config()

    wind_raw = load_wind_raw(wind_path)
    pv_raw = load_pv_raw(pv_path)
    price_raw = load_prices_raw(price_path)

    target_hours = 8760

    wind_clean = _prepare_series(wind_raw, "wind_per_kw")
    pv_clean = _prepare_series(pv_raw, "pv_per_kw")
    price_clean = _prepare_series(price_raw, "price_eur_per_mwh")

    def drop_leap_day(series: pd.DataFrame) -> pd.DataFrame:
        series = series.loc[
            ~((series["timestamp"].dt.month == 2) & (series["timestamp"].dt.day == 29))
        ].reset_index(drop=True)
        series["hour_of_year"] = range(1, len(series) + 1)
        return series

    wind_clean = drop_leap_day(wind_clean)
    pv_clean = drop_leap_day(pv_clean)
    price_clean = drop_leap_day(price_clean)

    n_hours = min(target_hours, len(wind_clean), len(pv_clean), len(price_clean))
    if n_hours == 0:
        raise ValueError("No overlapping hours between wind, PV, and price data")

    wind_clean = wind_clean.head(n_hours)
    pv_clean = pv_clean.head(n_hours)
    price_clean = price_clean.head(n_hours)

    df = pd.DataFrame({"hour_of_year": range(1, n_hours + 1)})
    df = df.merge(pv_clean[["hour_of_year", "pv_per_kw"]], on="hour_of_year", how="left")
    df = df.merge(wind_clean[["hour_of_year", "wind_per_kw"]], on="hour_of_year", how="left")
    df = df.merge(price_clean[["hour_of_year", "price_eur_per_mwh"]], on="hour_of_year", how="left")
    df = df.merge(price_clean[["hour_of_year", "timestamp"]], on="hour_of_year", how="left")

    df = df[["timestamp", "price_eur_per_mwh", "pv_per_kw", "wind_per_kw"]]

    missing = df[["price_eur_per_mwh", "pv_per_kw", "wind_per_kw"]].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values after alignment: {missing.to_dict()}")

    return df


def save_clean_dataset(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """
    Save cleaned dataset to CSV or parquet based on file extension.
    """
    if path is None:
        path = DATA_PROCESSED_DIR / "hourly_data.csv"

    if str(path).lower().endswith(".parquet"):
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved cleaned dataset to {path}")
