"""
PV + Wind + BESS hybrid analysis module.

This module provides functions to evaluate NPV for co-located PV, Wind, and
Battery Energy Storage Systems (BESS) behind a shared POI.

Parameterization:
- α_tot: Total overplanting factor = (P_pv + P_wind) / P_POI
- s_wind: Wind share of installed capacity = P_wind / (P_pv + P_wind)
- γ: Battery power ratio = P_bess / P_POI

The battery can charge from both PV and Wind generation (green from any renewable).
Battery duration is fixed (default 2 hours).
"""

from dataclasses import dataclass
from typing import List, Sequence, Optional, Dict

import numpy as np
import pandas as pd

from config import (
    POI_CAPACITY_MW,
    CAPEX_PV_EUR_PER_KW,
    CAPEX_WIND_EUR_PER_KW,
    CAPEX_BESS_EUR_PER_KWH,
    OPEX_PV_EUR_PER_KW_PER_YEAR,
    OPEX_WIND_EUR_PER_KW_PER_YEAR,
    OPEX_BESS_EUR_PER_KW_PER_YEAR,
    WACC_PV_REAL,
    WACC_WIND_REAL,
    WACC_BESS_REAL,
    DEGRADATION_PV_PER_YEAR,
    LIFETIME_PV_YEARS,
    LIFETIME_WIND_YEARS,
    LIFETIME_BESS_YEARS,
    PROJECT_LIFETIME_YEARS,
    BESS_REPLACEMENT_SHARE,
    BESS_ROUNDTRIP_EFF,
    INNOVATION_TENDER_MIN_BESS_POWER_RATIO,
    INNOVATION_TENDER_MIN_BESS_DURATION_H,
    INNOVATION_TENDER_PREMIUM_YEARS,
)
from economics import irr, npv


@dataclass
class HybridBessResult:
    """Result container for a single PV+Wind+BESS configuration evaluation."""

    # Configuration parameters
    alpha_tot: float              # Total overplanting factor
    s_wind: float                 # Wind share of installed capacity
    gamma: float                  # Battery power ratio (P_bess / P_POI)

    # Capacities
    pv_capacity_mw: float         # Installed PV capacity (MW)
    wind_capacity_mw: float       # Installed wind capacity (MW)
    bess_power_mw: float          # Battery power rating (MW)
    bess_energy_mwh: float        # Battery energy capacity (MWh)

    # Financial parameters
    wacc: float                   # CAPEX-weighted average WACC
    npv_eur: float                # Net Present Value (EUR)
    irr: float                    # Internal Rate of Return

    # CAPEX breakdown
    capex_pv_eur: float
    capex_wind_eur: float
    capex_bess_eur: float
    capex_total_eur: float

    # OPEX breakdown
    annual_opex_pv_eur: float
    annual_opex_wind_eur: float
    annual_opex_bess_eur: float
    annual_opex_total_eur: float

    # Revenue (Year 1)
    annual_revenue_eur: float

    # Energy metrics (Year 1)
    annual_export_mwh_total: float
    annual_export_mwh_pv: float
    annual_export_mwh_wind: float
    annual_export_mwh_bess: float
    annual_curtailment_mwh_total: float
    annual_curtailment_mwh_pv: float
    annual_curtailment_mwh_wind: float

    # Battery metrics
    annual_charge_mwh: float
    annual_discharge_mwh: float
    battery_cycles: float

    # System-oriented metrics
    poi_utilisation: float        # Annual export / (POI * 8760)
    curtailment_rate: float       # Curtailment / Total generation
    annual_revenue_eur_merchant: float = float("nan")  # yr1 merchant revenue for premium->merchant transition; NaN for non-innovation


def build_cash_flows_hybrid_bess(
    annual_revenue_pv_year1: float,
    annual_revenue_wind_year1: float,
    annual_opex_pv: float,
    annual_opex_wind: float,
    annual_opex_bess: float,
    lifetime_pv_years: int,
    lifetime_wind_years: int,
    lifetime_bess_years: int,
    project_lifetime_years: int,
    degradation_pv: float,
    bess_replacement_share: float,
    capex_bess: float,
    annual_revenue_pv_year1_merchant: Optional[float] = None,
    annual_revenue_wind_year1_merchant: Optional[float] = None,
    premium_years: Optional[int] = None,
) -> np.ndarray:
    """
    Build yearly net cash flows for a PV + Wind + BESS hybrid project.

    - PV revenue degrades over time, wind revenue stays constant (no degradation).
    - Each technology contributes revenue and OPEX only during its respective lifetime.
    - BESS is replaced at year 15 with a fraction of initial CAPEX.

    Parameters
    ----------
    annual_revenue_pv_year1 : float
        PV revenue in year 1 (before degradation).
    annual_revenue_wind_year1 : float
        Wind revenue in year 1 (constant, no degradation).
    annual_opex_pv : float
        Annual PV OPEX.
    annual_opex_wind : float
        Annual Wind OPEX.
    annual_opex_bess : float
        Annual BESS OPEX.
    lifetime_pv_years : int
        PV project lifetime (years).
    lifetime_wind_years : int
        Wind project lifetime (years).
    lifetime_bess_years : int
        BESS lifetime before replacement (years).
    project_lifetime_years : int
        Total project evaluation period (years).
    degradation_pv : float
        Annual PV degradation rate.
    bess_replacement_share : float
        BESS replacement cost as fraction of initial CAPEX.
    capex_bess : float
        Initial BESS CAPEX (EUR).
    annual_revenue_pv_year1_merchant : float, optional
        Merchant PV revenue in year 1 (used after premium support ends).
    annual_revenue_wind_year1_merchant : float, optional
        Merchant wind revenue in year 1 (used after premium support ends).
    premium_years : int, optional
        Number of years the premium applies before switching to merchant revenue.

    Returns
    -------
    np.ndarray
        Array of net cash flows for years 1 to project_lifetime_years.
    """
    cash_flows = np.zeros(project_lifetime_years)

    use_merchant_revenue = (
        annual_revenue_pv_year1_merchant is not None
        and annual_revenue_wind_year1_merchant is not None
        and premium_years is not None
        and premium_years > 0
    )

    for y in range(1, project_lifetime_years + 1):
        if use_merchant_revenue and y > premium_years:
            revenue_pv_base = annual_revenue_pv_year1_merchant
            revenue_wind_base = annual_revenue_wind_year1_merchant
        else:
            revenue_pv_base = annual_revenue_pv_year1
            revenue_wind_base = annual_revenue_wind_year1

        # PV revenue degrades, only within PV lifetime
        revenue_pv = (
            revenue_pv_base * (1 - degradation_pv) ** (y - 1)
            if y <= lifetime_pv_years else 0.0
        )
        # Wind revenue stays constant, only within wind lifetime
        revenue_wind = (
            revenue_wind_base if y <= lifetime_wind_years else 0.0
        )
        revenue_y = revenue_pv + revenue_wind

        # OPEX only applies during respective lifetimes
        opex_pv = annual_opex_pv if y <= lifetime_pv_years else 0.0
        opex_wind = annual_opex_wind if y <= lifetime_wind_years else 0.0
        opex_bess = annual_opex_bess if y <= project_lifetime_years else 0.0

        cash_flows[y - 1] = revenue_y - (opex_pv + opex_wind + opex_bess)

        # BESS replacement at year 15
        if y == lifetime_bess_years:
            cash_flows[y - 1] -= bess_replacement_share * capex_bess

    return cash_flows


def evaluate_hybrid_bess_single(
    df: pd.DataFrame,
    alpha_tot: float,
    s_wind: float,
    gamma: float,
    bess_duration_h: float = 2.0,
    poi_capacity_mw: float = POI_CAPACITY_MW,
    use_lp: bool = True,
    lp_time_limit: int = 60,
    verbose: bool = False,
) -> HybridBessResult:
    """
    Evaluate a single PV+Wind+BESS configuration.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    alpha_tot : float
        Total overplanting factor = (P_pv + P_wind) / P_POI.
    s_wind : float
        Wind share of installed capacity = P_wind / (P_pv + P_wind).
    gamma : float
        Battery power ratio = P_bess / P_POI.
    bess_duration_h : float, default 2.0
        Battery duration at rated power (hours).
    poi_capacity_mw : float, default 10.0
        POI capacity (MW).
    use_lp : bool, default True
        If True, use LP optimization for dispatch. Otherwise, use rule-based.
    lp_time_limit : int, default 60
        Time limit for LP solver (seconds).
    verbose : bool, default False
        If True, print solver progress.

    Returns
    -------
    HybridBessResult
        Result container with all metrics.
    """
    # 1. Calculate capacities
    total_capacity_mw = alpha_tot * poi_capacity_mw
    wind_capacity_mw = total_capacity_mw * s_wind
    pv_capacity_mw = total_capacity_mw * (1.0 - s_wind)
    bess_power_mw = gamma * poi_capacity_mw
    bess_energy_mwh = bess_power_mw * bess_duration_h

    # 2. Calculate CAPEX
    capex_pv = pv_capacity_mw * 1000.0 * CAPEX_PV_EUR_PER_KW
    capex_wind = wind_capacity_mw * 1000.0 * CAPEX_WIND_EUR_PER_KW
    capex_bess = bess_energy_mwh * 1000.0 * CAPEX_BESS_EUR_PER_KWH
    capex_total = capex_pv + capex_wind + capex_bess

    # 3. Calculate annual OPEX
    annual_opex_pv = pv_capacity_mw * 1000.0 * OPEX_PV_EUR_PER_KW_PER_YEAR
    annual_opex_wind = wind_capacity_mw * 1000.0 * OPEX_WIND_EUR_PER_KW_PER_YEAR
    annual_opex_bess = bess_power_mw * 1000.0 * OPEX_BESS_EUR_PER_KW_PER_YEAR
    annual_opex_total = annual_opex_pv + annual_opex_wind + annual_opex_bess

    # 4. CAPEX-weighted WACC
    if capex_total > 0:
        w_pv = capex_pv / capex_total
        w_wind = capex_wind / capex_total
        w_bess = capex_bess / capex_total
        wacc = w_pv * WACC_PV_REAL + w_wind * WACC_WIND_REAL + w_bess * WACC_BESS_REAL
    else:
        wacc = WACC_PV_REAL  # Fallback

    # 5. Prepare data for simulation
    pv_gen_mw = df['pv_per_kw'].values * pv_capacity_mw
    wind_gen_mw = df['wind_per_kw'].values * wind_capacity_mw

    sim_df = pd.DataFrame({
        'pv_generation_mw': pv_gen_mw,
        'wind_generation_mw': wind_gen_mw,
        'day_ahead_price_eur_mwh': df['price_eur_per_mwh'].values,
    })

    # 6. Run simulation
    if use_lp and bess_power_mw > 0:
        from simulation_lp import simulate_pv_wind_bess_lp

        params = {
            'bess_cap_mwh': bess_energy_mwh,
            'bess_pwr_mw': bess_power_mw,
            'poi_mw': poi_capacity_mw,
            'eta_ch': np.sqrt(BESS_ROUNDTRIP_EFF),
            'eta_dis': np.sqrt(BESS_ROUNDTRIP_EFF),
        }

        result = simulate_pv_wind_bess_lp(
            sim_df, params,
            time_limit_seconds=lp_time_limit,
            verbose=verbose,
        )

        annual_revenue_eur = float(result['revenue_eur'].sum())
        annual_revenue_pv_direct = float(result['revenue_pv_eur'].sum())
        annual_revenue_wind_direct = float(result['revenue_wind_eur'].sum())
        annual_revenue_bess = float(result['revenue_bess_eur'].sum())
        
        annual_charge_pv_mwh = float(result['charge_pv_mw'].sum())
        annual_charge_wind_mwh = float(result['charge_wind_mw'].sum())

        annual_export_mwh_total = float(result['export_mw'].sum())
        annual_export_mwh_pv = float(result['export_pv_mw'].sum())
        annual_export_mwh_wind = float(result['export_wind_mw'].sum())
        annual_export_mwh_bess = float(result['export_bess_mw'].sum())
        annual_curtailment_mwh_total = float(result['curtailment_mw'].sum())
        annual_curtailment_mwh_pv = float(result['curtailment_pv_mw'].sum())
        annual_curtailment_mwh_wind = float(result['curtailment_wind_mw'].sum())
        annual_charge_mwh = float(result['charge_mw'].sum())
        annual_discharge_mwh = float(result['discharge_mw'].sum())

    elif bess_power_mw > 0:
        # Rule-based dispatch (fallback) - not implemented yet
        raise NotImplementedError("Rule-based dispatch for PV+Wind+BESS not implemented")

    else:
        # No battery case: use hybrid simulation without battery
        from simulation import simulate_hybrid_year

        sim_result = simulate_hybrid_year(
            df=df,
            pv_capacity_mw=pv_capacity_mw,
            wind_capacity_mw=wind_capacity_mw,
            poi_capacity_mw=poi_capacity_mw,
        )

        annual_revenue_eur = sim_result['annual_revenue_eur_total']
        annual_export_mwh_total = sim_result['annual_export_mwh_total']
        annual_export_mwh_pv = sim_result['annual_export_mwh_pv']
        annual_export_mwh_wind = sim_result['annual_export_mwh_wind']
        annual_export_mwh_bess = 0.0
        annual_curtailment_mwh_total = sim_result['annual_curtailment_mwh_total']
        annual_curtailment_mwh_pv = sim_result['annual_curtailment_mwh_pv']
        annual_curtailment_mwh_wind = sim_result['annual_curtailment_mwh_wind']
        annual_charge_mwh = 0.0
        annual_discharge_mwh = 0.0

    # 7. Battery metrics
    battery_cycles = annual_discharge_mwh / bess_energy_mwh if bess_energy_mwh > 0 else 0.0

    # 8. System-oriented metrics
    poi_utilisation = annual_export_mwh_total / (poi_capacity_mw * 8760.0)
    total_generation_mwh = annual_export_mwh_total + annual_curtailment_mwh_total
    curtailment_rate = (
        annual_curtailment_mwh_total / total_generation_mwh
        if total_generation_mwh > 0 else 0.0
    )

    # 9. Build cash flows and compute NPV
    # Split revenue into PV and Wind portions for proper degradation handling
    # When s_wind = 1.0, all revenue is from wind (no degradation)
    # When s_wind = 0.0, all revenue is from PV (with degradation)
    
    # Use granular revenue data from LP if available (more accurate)
    if 'annual_revenue_pv_direct' in locals():
        # Allocate BESS revenue based on charging source
        total_charge_mwh = annual_charge_pv_mwh + annual_charge_wind_mwh
        if total_charge_mwh > 0:
            bess_rev_pv_share = annual_charge_pv_mwh / total_charge_mwh
            annual_revenue_pv = annual_revenue_pv_direct + annual_revenue_bess * bess_rev_pv_share
            annual_revenue_wind = annual_revenue_wind_direct + annual_revenue_bess * (1.0 - bess_rev_pv_share)
        else:
            # If no charging, use direct revenue
            annual_revenue_pv = annual_revenue_pv_direct
            annual_revenue_wind = annual_revenue_wind_direct + annual_revenue_bess
            
    # Fallback for non-LP or no-battery cases: split revenue by export shares
    elif annual_export_mwh_total > 0:
        # Use actual export shares for revenue split
        pv_revenue_share = annual_export_mwh_pv / annual_export_mwh_total
        wind_revenue_share = annual_export_mwh_wind / annual_export_mwh_total
        # BESS export is arbitrage, attribute to source that charged it (simplified: split proportionally)
        annual_revenue_pv = annual_revenue_eur * pv_revenue_share
        annual_revenue_wind = annual_revenue_eur * wind_revenue_share
    elif pv_capacity_mw > 0 and wind_capacity_mw == 0:
        # Pure PV
        annual_revenue_pv = annual_revenue_eur
        annual_revenue_wind = 0.0
    elif wind_capacity_mw > 0 and pv_capacity_mw == 0:
        # Pure Wind (s_wind = 1.0)
        annual_revenue_pv = 0.0
        annual_revenue_wind = annual_revenue_eur
    else:
        # Fallback: split by capacity if no exports (shouldn't happen)
        total_capacity = pv_capacity_mw + wind_capacity_mw
        if total_capacity > 0:
            pv_revenue_share = pv_capacity_mw / total_capacity
            wind_revenue_share = wind_capacity_mw / total_capacity
            annual_revenue_pv = annual_revenue_eur * pv_revenue_share
            annual_revenue_wind = annual_revenue_eur * wind_revenue_share
        else:
            annual_revenue_pv = 0.0
            annual_revenue_wind = 0.0
    
    cash_flows = build_cash_flows_hybrid_bess(
        annual_revenue_pv_year1=annual_revenue_pv,
        annual_revenue_wind_year1=annual_revenue_wind,
        annual_opex_pv=annual_opex_pv,
        annual_opex_wind=annual_opex_wind,
        annual_opex_bess=annual_opex_bess,
        lifetime_pv_years=LIFETIME_PV_YEARS,
        lifetime_wind_years=LIFETIME_WIND_YEARS,
        lifetime_bess_years=LIFETIME_BESS_YEARS,
        project_lifetime_years=PROJECT_LIFETIME_YEARS,
        degradation_pv=DEGRADATION_PV_PER_YEAR,
        bess_replacement_share=BESS_REPLACEMENT_SHARE,
        capex_bess=capex_bess,
    )

    full_cash_flows = np.insert(cash_flows, 0, -capex_total)
    npv_total = npv(full_cash_flows, wacc)
    try:
        irr_val = irr(full_cash_flows)
    except ValueError:
        irr_val = np.nan

    return HybridBessResult(
        alpha_tot=alpha_tot,
        s_wind=s_wind,
        gamma=gamma,
        pv_capacity_mw=pv_capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        bess_power_mw=bess_power_mw,
        bess_energy_mwh=bess_energy_mwh,
        wacc=wacc,
        npv_eur=npv_total,
        irr=irr_val,
        capex_pv_eur=capex_pv,
        capex_wind_eur=capex_wind,
        capex_bess_eur=capex_bess,
        capex_total_eur=capex_total,
        annual_opex_pv_eur=annual_opex_pv,
        annual_opex_wind_eur=annual_opex_wind,
        annual_opex_bess_eur=annual_opex_bess,
        annual_opex_total_eur=annual_opex_total,
        annual_revenue_eur=annual_revenue_eur,
        annual_export_mwh_total=annual_export_mwh_total,
        annual_export_mwh_pv=annual_export_mwh_pv,
        annual_export_mwh_wind=annual_export_mwh_wind,
        annual_export_mwh_bess=annual_export_mwh_bess,
        annual_curtailment_mwh_total=annual_curtailment_mwh_total,
        annual_curtailment_mwh_pv=annual_curtailment_mwh_pv,
        annual_curtailment_mwh_wind=annual_curtailment_mwh_wind,
        annual_charge_mwh=annual_charge_mwh,
        annual_discharge_mwh=annual_discharge_mwh,
        battery_cycles=battery_cycles,
        poi_utilisation=poi_utilisation,
        curtailment_rate=curtailment_rate,
    )


def evaluate_hybrid_bess_grid(
    df: pd.DataFrame,
    alpha_tot_values: Sequence[float],
    s_wind_values: Sequence[float],
    gamma_values: Sequence[float],
    bess_duration_h: float = 2.0,
    poi_capacity_mw: float = POI_CAPACITY_MW,
    use_lp: bool = True,
    lp_time_limit: int = 60,
    verbose: bool = False,
) -> List[HybridBessResult]:
    """
    Evaluate NPV for a grid of PV+Wind+BESS configurations.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    alpha_tot_values : Sequence[float]
        List of total overplanting factors to evaluate.
    s_wind_values : Sequence[float]
        List of wind share values to evaluate.
    gamma_values : Sequence[float]
        List of battery power ratios to evaluate.
    bess_duration_h : float, default 2.0
        Battery duration at rated power (hours).
    poi_capacity_mw : float, default 10.0
        POI capacity (MW).
    use_lp : bool, default True
        If True, use LP optimization for dispatch.
    lp_time_limit : int, default 60
        Time limit for LP solver per configuration (seconds).
    verbose : bool, default False
        If True, print progress.

    Returns
    -------
    List[HybridBessResult]
        List of results for each (alpha_tot, s_wind, gamma) combination.
    """
    results: List[HybridBessResult] = []
    total_configs = len(alpha_tot_values) * len(s_wind_values) * len(gamma_values)
    count = 0

    for alpha_tot in alpha_tot_values:
        for s_wind in s_wind_values:
            for gamma in gamma_values:
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  Evaluating config {count}/{total_configs}...")

                result = evaluate_hybrid_bess_single(
                    df=df,
                    alpha_tot=alpha_tot,
                    s_wind=s_wind,
                    gamma=gamma,
                    bess_duration_h=bess_duration_h,
                    poi_capacity_mw=poi_capacity_mw,
                    use_lp=use_lp,
                    lp_time_limit=lp_time_limit,
                    verbose=False,
                )
                results.append(result)

    return results


def find_best_by_npv(results: Sequence[HybridBessResult]) -> HybridBessResult:
    """Return the configuration with the highest NPV."""
    return max(results, key=lambda r: r.npv_eur)


def find_best_by_poi_utilisation(results: Sequence[HybridBessResult]) -> HybridBessResult:
    """Return the configuration with the highest POI utilisation."""
    return max(results, key=lambda r: r.poi_utilisation)


def results_to_dataframe(results: Sequence[HybridBessResult]) -> pd.DataFrame:
    """Convert a list of HybridBessResult objects to a DataFrame."""
    return pd.DataFrame([r.__dict__ for r in results])


def pivot_heatmap(
    df_results: pd.DataFrame,
    value_column: str,
    row_column: str = 'alpha_tot',
    col_column: str = 's_wind',
    filter_column: Optional[str] = None,
    filter_value: Optional[float] = None,
) -> pd.DataFrame:
    """
    Pivot the results DataFrame to create a heatmap-ready matrix.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame from results_to_dataframe.
    value_column : str
        Column to use as values (e.g., 'npv_eur', 'poi_utilisation').
    row_column : str, default 'alpha_tot'
        Column to use as row index.
    col_column : str, default 's_wind'
        Column to use as column index.
    filter_column : str, optional
        Column to filter by (e.g., 'gamma').
    filter_value : float, optional
        Value to filter for.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame ready for heatmap plotting.
    """
    df = df_results.copy()

    if filter_column is not None and filter_value is not None:
        df = df[df[filter_column] == filter_value]

    return df.pivot(index=row_column, columns=col_column, values=value_column)


def evaluate_hybrid_bess_single_innovationsausschreibung(
    df: pd.DataFrame,
    alpha_tot: float,
    s_wind: float,
    gamma: float,
    bess_duration_h: float = 2.0,
    poi_capacity_mw: float = POI_CAPACITY_MW,
    lp_time_limit: int = 60,
    verbose: bool = False,
) -> HybridBessResult:
    """
    Evaluate a single PV+Wind+BESS configuration using Innovationsausschreibung 2024.

    This function uses the Floating Market Premium (FMP) mechanism from the EEG
    Innovation Tender 2024, which adds a market premium to the revenue for the
    first INNOVATION_TENDER_PREMIUM_YEARS years before switching to merchant revenues.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    alpha_tot : float
        Total overplanting factor = (P_pv + P_wind) / P_POI.
    s_wind : float
        Wind share of installed capacity = P_wind / (P_pv + P_wind).
    gamma : float
        Battery power ratio = P_bess / P_POI.
    bess_duration_h : float, default 2.0
        Battery duration at rated power (hours).
    poi_capacity_mw : float, default 10.0
        POI capacity (MW).
    lp_time_limit : int, default 60
        Time limit for LP solver (seconds).
    verbose : bool, default False
        If True, print solver progress.

    Returns
    -------
    HybridBessResult
        Result container with all metrics.

    Notes
    -----
    BESS sizing constraints for Innovation Tender eligibility:
    - P_BESS >= 0.25 × P_PV (or P_Wind for wind-only cases)
    - E_BESS >= 2.0 × P_BESS
    Configurations that don't meet these constraints will raise a ValueError.
    """
    # 1. Calculate capacities
    total_capacity_mw = alpha_tot * poi_capacity_mw
    wind_capacity_mw = total_capacity_mw * s_wind
    pv_capacity_mw = total_capacity_mw * (1.0 - s_wind)
    bess_power_mw = gamma * poi_capacity_mw
    bess_energy_mwh = bess_power_mw * bess_duration_h

    # Check Innovation Tender eligibility constraints
    # Innovation Tender requires BESS, so gamma=0 configurations are not eligible
    if bess_power_mw == 0:
        raise ValueError(
            "Innovation Tender requires BESS (gamma > 0). "
            "Configurations without battery are not eligible."
        )
    
    base_capacity_mw = pv_capacity_mw if pv_capacity_mw > 0 else wind_capacity_mw
    capacity_label = "PV" if pv_capacity_mw > 0 else "Wind"
    min_bess_power = INNOVATION_TENDER_MIN_BESS_POWER_RATIO * base_capacity_mw
    min_bess_energy = INNOVATION_TENDER_MIN_BESS_DURATION_H * bess_power_mw
    
    if bess_power_mw < min_bess_power:
        raise ValueError(
            f"BESS power {bess_power_mw} MW must be >= {min_bess_power} MW "
            f"(0.25 × {capacity_label} capacity {base_capacity_mw} MW) for Innovation Tender eligibility"
        )
    
    if bess_energy_mwh < min_bess_energy:
        raise ValueError(
            f"BESS energy {bess_energy_mwh} MWh must be >= {min_bess_energy} MWh "
            f"(2.0 × BESS power {bess_power_mw} MW) for Innovation Tender eligibility"
        )

    # 2. Calculate CAPEX
    capex_pv = pv_capacity_mw * 1000.0 * CAPEX_PV_EUR_PER_KW
    capex_wind = wind_capacity_mw * 1000.0 * CAPEX_WIND_EUR_PER_KW
    capex_bess = bess_energy_mwh * 1000.0 * CAPEX_BESS_EUR_PER_KWH
    capex_total = capex_pv + capex_wind + capex_bess

    # 3. Calculate annual OPEX
    annual_opex_pv = pv_capacity_mw * 1000.0 * OPEX_PV_EUR_PER_KW_PER_YEAR
    annual_opex_wind = wind_capacity_mw * 1000.0 * OPEX_WIND_EUR_PER_KW_PER_YEAR
    annual_opex_bess = bess_power_mw * 1000.0 * OPEX_BESS_EUR_PER_KW_PER_YEAR
    annual_opex_total = annual_opex_pv + annual_opex_wind + annual_opex_bess

    # 4. CAPEX-weighted WACC
    if capex_total > 0:
        w_pv = capex_pv / capex_total
        w_wind = capex_wind / capex_total
        w_bess = capex_bess / capex_total
        wacc = w_pv * WACC_PV_REAL + w_wind * WACC_WIND_REAL + w_bess * WACC_BESS_REAL
    else:
        wacc = WACC_PV_REAL  # Fallback

    # 5. Prepare data for simulation
    pv_gen_mw = df['pv_per_kw'].values * pv_capacity_mw
    wind_gen_mw = df['wind_per_kw'].values * wind_capacity_mw

    sim_df = pd.DataFrame({
        'pv_generation_mw': pv_gen_mw,
        'wind_generation_mw': wind_gen_mw,
        'day_ahead_price_eur_mwh': df['price_eur_per_mwh'].values,
    })
    
    # Add timestamp if available
    if 'timestamp' in df.columns:
        sim_df['timestamp'] = df['timestamp'].values
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        sim_df['timestamp'] = df.index

    # 6. Run simulation with Innovationsausschreibung
    # (bess_power_mw > 0 is guaranteed by eligibility check above)
    from simulation_lp import (
        simulate_pv_wind_bess_lp,
        simulate_pv_wind_bess_lp_innovationsausschreibung,
    )

    params = {
        'bess_cap_mwh': bess_energy_mwh,
        'bess_pwr_mw': bess_power_mw,
        'poi_mw': poi_capacity_mw,
        'eta_ch': np.sqrt(BESS_ROUNDTRIP_EFF),
        'eta_dis': np.sqrt(BESS_ROUNDTRIP_EFF),
    }

    result = simulate_pv_wind_bess_lp_innovationsausschreibung(
        sim_df, params,
        pv_capacity_mw=pv_capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        time_limit_seconds=lp_time_limit,
        verbose=verbose,
    )

    annual_revenue_eur = float(result['revenue_eur'].sum())
    annual_revenue_pv_direct = float(result['revenue_pv_eur'].sum())
    annual_revenue_wind_direct = float(result['revenue_wind_eur'].sum())
    annual_revenue_bess = float(result['revenue_bess_eur'].sum())
    
    annual_charge_pv_mwh = float(result['charge_pv_mw'].sum())
    annual_charge_wind_mwh = float(result['charge_wind_mw'].sum())

    annual_export_mwh_total = float(result['export_mw'].sum())
    annual_export_mwh_pv = float(result['export_pv_mw'].sum())
    annual_export_mwh_wind = float(result['export_wind_mw'].sum())
    annual_export_mwh_bess = float(result['export_bess_mw'].sum())
    annual_curtailment_mwh_total = float(result['curtailment_mw'].sum())
    annual_curtailment_mwh_pv = float(result['curtailment_pv_mw'].sum())
    annual_curtailment_mwh_wind = float(result['curtailment_wind_mw'].sum())
    annual_charge_mwh = float(result['charge_mw'].sum())
    annual_discharge_mwh = float(result['discharge_mw'].sum())

    annual_revenue_pv_merchant: Optional[float] = None
    annual_revenue_wind_merchant: Optional[float] = None
    try:
        merchant_result = simulate_pv_wind_bess_lp(
            sim_df, params,
            time_limit_seconds=lp_time_limit,
            verbose=verbose,
        )
        annual_revenue_pv_direct_merchant = float(merchant_result['revenue_pv_eur'].sum())
        annual_revenue_wind_direct_merchant = float(merchant_result['revenue_wind_eur'].sum())
        annual_revenue_bess_merchant = float(merchant_result['revenue_bess_eur'].sum())

        annual_charge_pv_mwh_merchant = float(merchant_result['charge_pv_mw'].sum())
        annual_charge_wind_mwh_merchant = float(merchant_result['charge_wind_mw'].sum())

        total_charge_mwh_merchant = annual_charge_pv_mwh_merchant + annual_charge_wind_mwh_merchant
        if total_charge_mwh_merchant > 0:
            bess_rev_pv_share_merchant = annual_charge_pv_mwh_merchant / total_charge_mwh_merchant
            annual_revenue_pv_merchant = (
                annual_revenue_pv_direct_merchant
                + annual_revenue_bess_merchant * bess_rev_pv_share_merchant
            )
            annual_revenue_wind_merchant = (
                annual_revenue_wind_direct_merchant
                + annual_revenue_bess_merchant * (1.0 - bess_rev_pv_share_merchant)
            )
        else:
            annual_revenue_pv_merchant = annual_revenue_pv_direct_merchant
            annual_revenue_wind_merchant = annual_revenue_wind_direct_merchant + annual_revenue_bess_merchant
    except Exception:
        annual_revenue_pv_direct_merchant = float(
            (result['export_pv_mw'] * result['price_eur_mwh']).sum()
        )
        annual_revenue_wind_direct_merchant = float(
            (result['export_wind_mw'] * result['price_eur_mwh']).sum()
        )
        annual_revenue_bess_merchant = float(
            (result['export_bess_mw'] * result['price_eur_mwh']).sum()
        )

        total_charge_mwh = annual_charge_pv_mwh + annual_charge_wind_mwh
        if total_charge_mwh > 0:
            bess_rev_pv_share = annual_charge_pv_mwh / total_charge_mwh
            annual_revenue_pv_merchant = (
                annual_revenue_pv_direct_merchant
                + annual_revenue_bess_merchant * bess_rev_pv_share
            )
            annual_revenue_wind_merchant = (
                annual_revenue_wind_direct_merchant
                + annual_revenue_bess_merchant * (1.0 - bess_rev_pv_share)
            )
        else:
            annual_revenue_pv_merchant = annual_revenue_pv_direct_merchant
            annual_revenue_wind_merchant = annual_revenue_wind_direct_merchant + annual_revenue_bess_merchant

    # 7. Battery metrics
    battery_cycles = annual_discharge_mwh / bess_energy_mwh if bess_energy_mwh > 0 else 0.0

    # 8. System-oriented metrics
    poi_utilisation = annual_export_mwh_total / (poi_capacity_mw * 8760.0)
    total_generation_mwh = annual_export_mwh_total + annual_curtailment_mwh_total
    curtailment_rate = (
        annual_curtailment_mwh_total / total_generation_mwh
        if total_generation_mwh > 0 else 0.0
    )

    # 9. Build cash flows and compute NPV
    # Split revenue into PV and Wind portions for proper degradation handling
    
    # Use granular revenue data from LP (more accurate)
    # Allocate BESS revenue based on charging source
    total_charge_mwh = annual_charge_pv_mwh + annual_charge_wind_mwh
    if total_charge_mwh > 0:
        bess_rev_pv_share = annual_charge_pv_mwh / total_charge_mwh
        annual_revenue_pv = annual_revenue_pv_direct + annual_revenue_bess * bess_rev_pv_share
        annual_revenue_wind = annual_revenue_wind_direct + annual_revenue_bess * (1.0 - bess_rev_pv_share)
    else:
        # If no charging, use direct revenue
        annual_revenue_pv = annual_revenue_pv_direct
        annual_revenue_wind = annual_revenue_wind_direct + annual_revenue_bess
    
    cash_flows = build_cash_flows_hybrid_bess(
        annual_revenue_pv_year1=annual_revenue_pv,
        annual_revenue_wind_year1=annual_revenue_wind,
        annual_opex_pv=annual_opex_pv,
        annual_opex_wind=annual_opex_wind,
        annual_opex_bess=annual_opex_bess,
        lifetime_pv_years=LIFETIME_PV_YEARS,
        lifetime_wind_years=LIFETIME_WIND_YEARS,
        lifetime_bess_years=LIFETIME_BESS_YEARS,
        project_lifetime_years=PROJECT_LIFETIME_YEARS,
        degradation_pv=DEGRADATION_PV_PER_YEAR,
        bess_replacement_share=BESS_REPLACEMENT_SHARE,
        capex_bess=capex_bess,
        annual_revenue_pv_year1_merchant=annual_revenue_pv_merchant,
        annual_revenue_wind_year1_merchant=annual_revenue_wind_merchant,
        premium_years=INNOVATION_TENDER_PREMIUM_YEARS,
    )

    full_cash_flows = np.insert(cash_flows, 0, -capex_total)
    npv_total = npv(full_cash_flows, wacc)
    try:
        irr_val = irr(full_cash_flows)
    except ValueError:
        irr_val = np.nan

    return HybridBessResult(
        alpha_tot=alpha_tot,
        s_wind=s_wind,
        gamma=gamma,
        pv_capacity_mw=pv_capacity_mw,
        wind_capacity_mw=wind_capacity_mw,
        bess_power_mw=bess_power_mw,
        bess_energy_mwh=bess_energy_mwh,
        wacc=wacc,
        npv_eur=npv_total,
        irr=irr_val,
        capex_pv_eur=capex_pv,
        capex_wind_eur=capex_wind,
        capex_bess_eur=capex_bess,
        capex_total_eur=capex_total,
        annual_opex_pv_eur=annual_opex_pv,
        annual_opex_wind_eur=annual_opex_wind,
        annual_opex_bess_eur=annual_opex_bess,
        annual_opex_total_eur=annual_opex_total,
        annual_revenue_eur=annual_revenue_eur,
        annual_export_mwh_total=annual_export_mwh_total,
        annual_export_mwh_pv=annual_export_mwh_pv,
        annual_export_mwh_wind=annual_export_mwh_wind,
        annual_export_mwh_bess=annual_export_mwh_bess,
        annual_curtailment_mwh_total=annual_curtailment_mwh_total,
        annual_curtailment_mwh_pv=annual_curtailment_mwh_pv,
        annual_curtailment_mwh_wind=annual_curtailment_mwh_wind,
        annual_charge_mwh=annual_charge_mwh,
        annual_discharge_mwh=annual_discharge_mwh,
        battery_cycles=battery_cycles,
        poi_utilisation=poi_utilisation,
        curtailment_rate=curtailment_rate,
        annual_revenue_eur_merchant=(annual_revenue_pv_merchant or 0.0) + (annual_revenue_wind_merchant or 0.0),
    )


def evaluate_hybrid_bess_grid_innovationsausschreibung(
    df: pd.DataFrame,
    alpha_tot_values: Sequence[float],
    s_wind_values: Sequence[float],
    gamma_values: Sequence[float],
    bess_duration_h: float = 2.0,
    poi_capacity_mw: float = POI_CAPACITY_MW,
    lp_time_limit: int = 60,
    verbose: bool = False,
) -> List[HybridBessResult]:
    """
    Evaluate NPV for a grid of PV+Wind+BESS configurations using Innovationsausschreibung 2024.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    alpha_tot_values : Sequence[float]
        List of total overplanting factors to evaluate.
    s_wind_values : Sequence[float]
        List of wind share values to evaluate.
    gamma_values : Sequence[float]
        List of battery power ratios to evaluate.
    bess_duration_h : float, default 2.0
        Battery duration at rated power (hours).
    poi_capacity_mw : float, default 10.0
        POI capacity (MW).
    lp_time_limit : int, default 60
        Time limit for LP solver per configuration (seconds).
    verbose : bool, default False
        If True, print progress.

    Returns
    -------
    List[HybridBessResult]
        List of results for each (alpha_tot, s_wind, gamma) combination.
        Configurations that don't meet Innovation Tender eligibility constraints
        are skipped (not included in results).
    """
    results: List[HybridBessResult] = []
    total_configs = len(alpha_tot_values) * len(s_wind_values) * len(gamma_values)
    count = 0
    skipped = 0

    for alpha_tot in alpha_tot_values:
        for s_wind in s_wind_values:
            for gamma in gamma_values:
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  Evaluating config {count}/{total_configs} (skipped {skipped})...")

                try:
                    result = evaluate_hybrid_bess_single_innovationsausschreibung(
                        df=df,
                        alpha_tot=alpha_tot,
                        s_wind=s_wind,
                        gamma=gamma,
                        bess_duration_h=bess_duration_h,
                        poi_capacity_mw=poi_capacity_mw,
                        lp_time_limit=lp_time_limit,
                        verbose=False,
                    )
                    results.append(result)
                except ValueError as e:
                    # Skip configurations that don't meet Innovation Tender eligibility
                    skipped += 1
                    if verbose and skipped <= 5:
                        print(f"    Skipping config (α_tot={alpha_tot:.2f}, s_wind={s_wind:.2f}, γ={gamma:.2f}): {e}")

    return results
