"""
Core simulation functions for renewable energy generation and export.
"""

import pandas as pd
import numpy as np
from typing import Dict

from config import POI_CAPACITY_MW


def simulate_hybrid_year(
    df: pd.DataFrame,
    pv_capacity_mw: float,
    wind_capacity_mw: float,
    poi_capacity_mw: float = POI_CAPACITY_MW,
) -> Dict[str, float]:
    """
    Simulate one year for a co-located PV and wind plant behind a shared POI.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    pv_capacity_mw : float
        Installed PV capacity in MW.
    wind_capacity_mw : float
        Installed wind capacity in MW.
    poi_capacity_mw : float, default 10.0
        Export limit at the point of interconnection in MW.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'annual_export_mwh_total', 'annual_export_mwh_pv', 'annual_export_mwh_wind'
        - 'annual_curtailment_mwh_total', 'annual_curtailment_mwh_pv', 'annual_curtailment_mwh_wind'
        - 'annual_revenue_eur_total', 'annual_revenue_eur_pv', 'annual_revenue_eur_wind'
    """
    # Validate columns
    for col in ["price_eur_per_mwh", "pv_per_kw", "wind_per_kw"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    # Convert capacities to kW
    pv_kw = pv_capacity_mw * 1000.0
    wind_kw = wind_capacity_mw * 1000.0

    # Compute potential generation (vectorized)
    pv_gen_kw = df["pv_per_kw"].values * pv_kw
    wind_gen_kw = df["wind_per_kw"].values * wind_kw
    pv_gen_mw = pv_gen_kw / 1000.0
    wind_gen_mw = wind_gen_kw / 1000.0
    total_gen_mw = pv_gen_mw + wind_gen_mw
    prices = df["price_eur_per_mwh"].values

    # Initialize output arrays
    n_hours = len(df)
    export_pv_mw = np.zeros(n_hours)
    export_wind_mw = np.zeros(n_hours)
    curtail_pv_mw = np.zeros(n_hours)
    curtail_wind_mw = np.zeros(n_hours)

    # Case 1: prices < 0 -> economic curtailment (no export)
    neg_price_mask = prices < 0
    curtail_pv_mw[neg_price_mask] = pv_gen_mw[neg_price_mask]
    curtail_wind_mw[neg_price_mask] = wind_gen_mw[neg_price_mask]

    # Case 2: prices >= 0 and total_gen <= POI -> no curtailment
    pos_price_mask = ~neg_price_mask
    no_curtail_mask = pos_price_mask & (total_gen_mw <= poi_capacity_mw)
    export_pv_mw[no_curtail_mask] = pv_gen_mw[no_curtail_mask]
    export_wind_mw[no_curtail_mask] = wind_gen_mw[no_curtail_mask]

    # Case 3: prices >= 0 and total_gen > POI -> proportional allocation
    curtail_mask = pos_price_mask & (total_gen_mw > poi_capacity_mw)
    
    # Handle proportional allocation (avoid division by zero)
    total_gen_safe = np.where(total_gen_mw > 0, total_gen_mw, 1.0)
    ratio_pv = pv_gen_mw / total_gen_safe
    ratio_wind = wind_gen_mw / total_gen_safe
    
    export_pv_mw[curtail_mask] = ratio_pv[curtail_mask] * poi_capacity_mw
    export_wind_mw[curtail_mask] = ratio_wind[curtail_mask] * poi_capacity_mw
    curtail_pv_mw[curtail_mask] = pv_gen_mw[curtail_mask] - export_pv_mw[curtail_mask]
    curtail_wind_mw[curtail_mask] = wind_gen_mw[curtail_mask] - export_wind_mw[curtail_mask]

    # Compute revenues (hourly data: MW = MWh for 1 hour)
    revenue_pv = export_pv_mw * prices
    revenue_wind = export_wind_mw * prices

    # Aggregate annual totals
    return {
        "annual_export_mwh_total": float(np.sum(export_pv_mw + export_wind_mw)),
        "annual_export_mwh_pv": float(np.sum(export_pv_mw)),
        "annual_export_mwh_wind": float(np.sum(export_wind_mw)),
        "annual_curtailment_mwh_total": float(np.sum(curtail_pv_mw + curtail_wind_mw)),
        "annual_curtailment_mwh_pv": float(np.sum(curtail_pv_mw)),
        "annual_curtailment_mwh_wind": float(np.sum(curtail_wind_mw)),
        "annual_revenue_eur_total": float(np.sum(revenue_pv + revenue_wind)),
        "annual_revenue_eur_pv": float(np.sum(revenue_pv)),
        "annual_revenue_eur_wind": float(np.sum(revenue_wind)),
    }


def simulate_single_tech_year(
    df: pd.DataFrame,
    tech: str,
    capacity_mw: float,
    poi_capacity_mw: float = POI_CAPACITY_MW,
) -> Dict[str, float]:
    """
    Simulate one year for a stand-alone PV or wind plant.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hourly columns: 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    tech : str
        Technology type: 'pv' or 'wind'.
    capacity_mw : float
        Installed generator capacity in MW.
    poi_capacity_mw : float, optional
        Export limit at the point of interconnection in MW (default: 10.0).
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'annual_export_mwh': Total energy exported in the year.
        - 'annual_curtailment_mwh': Total energy curtailed in the year.
        - 'annual_revenue_eur': Total revenue from energy sales.
        
    Raises
    ------
    ValueError
        If tech is not 'pv' or 'wind', or if required columns are missing.
    """
    if tech not in ("pv", "wind"):
        raise ValueError(f"tech must be 'pv' or 'wind', got '{tech}'")
    
    col_name = "pv_per_kw" if tech == "pv" else "wind_per_kw"
    if col_name not in df.columns:
        raise ValueError(f"DataFrame must contain '{col_name}' column")
    if "price_eur_per_mwh" not in df.columns:
        raise ValueError("DataFrame must contain 'price_eur_per_mwh' column")
    
    # Convert installed capacity to kW
    capacity_kw = capacity_mw * 1000.0
    
    # Compute potential generation in kW, then convert to MW
    gen_kw = df[col_name] * capacity_kw
    gen_mw = gen_kw / 1000.0
    
    # Get prices
    price = df["price_eur_per_mwh"].values
    
    # Apply price rule and POI constraint
    # If price < 0: export = 0, curtailment = gen_mw (economic curtailment)
    # Else: export = min(gen_mw, poi_capacity_mw), curtailment = max(0, gen_mw - poi_capacity_mw)
    negative_price_mask = price < 0
    
    export_mw = np.where(
        negative_price_mask,
        0.0,
        np.minimum(gen_mw.values, poi_capacity_mw)
    )
    
    curtailment_mw = np.where(
        negative_price_mask,
        gen_mw.values,
        np.maximum(0.0, gen_mw.values - poi_capacity_mw)
    )
    
    # Revenue per hour (assuming hourly data, 1 hour = 1 MWh per MW)
    # Revenue = export_mw * price (since hourly, export_mw equals export_mwh for that hour)
    hourly_revenue = export_mw * price
    
    # Aggregate annual totals
    annual_export_mwh = float(np.sum(export_mw))
    annual_curtailment_mwh = float(np.sum(curtailment_mw))
    annual_revenue_eur = float(np.sum(hourly_revenue))
    
    return {
        "annual_export_mwh": annual_export_mwh,
        "annual_curtailment_mwh": annual_curtailment_mwh,
        "annual_revenue_eur": annual_revenue_eur,
    }


def simulate_export_wind(
    df: pd.DataFrame,
    wind_capacity_mw: float,
    poi_capacity_mw: float = POI_CAPACITY_MW
) -> pd.DataFrame:
    """
    Simulate wind-only export with POI capacity constraint.
    
    Args:
        df: Cleaned DataFrame with 'wind_per_kw' column
        wind_capacity_mw: Installed wind capacity in MW
        poi_capacity_mw: Export limit at POI (MW)
        
    Returns:
        DataFrame with columns:
        - timestamp: Datetime index
        - export_mw: Exported power (capped at POI capacity)
        - curtailment_mw: Curtailed generation (generation - export)
        - state_of_charge_mwh: Always 0 for wind-only (no storage)
    """
    # Validate column
    if 'wind_per_kw' not in df.columns:
        raise ValueError("DataFrame must contain 'wind_per_kw' column")
        
    # Calculate actual generation
    wind_kw = wind_capacity_mw * 1000.0
    generation = (df['wind_per_kw'] * wind_kw) / 1000.0
    
    # Apply POI capacity constraint
    export = np.minimum(generation, poi_capacity_mw)
    curtailment = generation - export
    
    result = pd.DataFrame({
        'timestamp': df['timestamp'] if 'timestamp' in df.columns else df.index,
        'export_mw': export,
        'curtailment_mw': curtailment,
        'state_of_charge_mwh': 0.0
    })
    
    return result


def simulate_export_pv(
    df: pd.DataFrame,
    pv_capacity_mw: float,
    poi_capacity_mw: float = POI_CAPACITY_MW
) -> pd.DataFrame:
    """
    Simulate PV-only export with POI capacity constraint.
    
    Args:
        df: Cleaned DataFrame with 'pv_per_kw' column
        pv_capacity_mw: Installed PV capacity in MW
        poi_capacity_mw: Export limit at POI (MW)
        
    Returns:
        DataFrame with columns:
        - timestamp: Datetime index
        - export_mw: Exported power (capped at POI capacity)
        - curtailment_mw: Curtailed generation (generation - export)
        - state_of_charge_mwh: Always 0 for PV-only (no storage)
    """
    # Validate column
    if 'pv_per_kw' not in df.columns:
        raise ValueError("DataFrame must contain 'pv_per_kw' column")
        
    # Calculate actual generation
    pv_kw = pv_capacity_mw * 1000.0
    generation = (df['pv_per_kw'] * pv_kw) / 1000.0
    
    # Apply POI capacity constraint
    export = np.minimum(generation, poi_capacity_mw)
    curtailment = generation - export
    
    result = pd.DataFrame({
        'timestamp': df['timestamp'] if 'timestamp' in df.columns else df.index,
        'export_mw': export,
        'curtailment_mw': curtailment,
        'state_of_charge_mwh': 0.0
    })
    
    return result


def simulate_export_hybrid(
    df: pd.DataFrame,
    wind_capacity_mw: float,
    pv_capacity_mw: float,
    poi_capacity_mw: float = POI_CAPACITY_MW
) -> pd.DataFrame:
    """
    Simulate co-located wind and PV export sharing a single POI.
    
    Args:
        df: Cleaned DataFrame with 'wind_per_kw' and 'pv_per_kw' columns
        wind_capacity_mw: Installed wind capacity in MW
        pv_capacity_mw: Installed PV capacity in MW
        poi_capacity_mw: Export limit at POI (MW)
        
    Returns:
        DataFrame with columns:
        - timestamp: Datetime index
        - export_mw: Total exported power (wind + PV, capped at POI)
        - curtailment_mw: Total curtailed generation
        - state_of_charge_mwh: Always 0 (no storage)
    """
    # Validate columns
    for col in ['wind_per_kw', 'pv_per_kw']:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column")

    # Calculate generation
    wind_gen = (df['wind_per_kw'] * wind_capacity_mw * 1000.0) / 1000.0
    pv_gen = (df['pv_per_kw'] * pv_capacity_mw * 1000.0) / 1000.0
    total_generation = wind_gen + pv_gen
    
    # Apply POI capacity constraint
    export = np.minimum(total_generation, poi_capacity_mw)
    curtailment = total_generation - export
    
    result = pd.DataFrame({
        'timestamp': df['timestamp'] if 'timestamp' in df.columns else df.index,
        'export_mw': export,
        'curtailment_mw': curtailment,
        'state_of_charge_mwh': 0.0
    })
    
    return result

