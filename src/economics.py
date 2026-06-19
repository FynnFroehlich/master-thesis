"""
Economic evaluation functions for renewable energy projects.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from config import Config


def npv(cash_flows: np.ndarray, discount_rate: float) -> float:
    """
    Compute Net Present Value (NPV) for yearly cash flows.
    
    Assumes cash_flows[0] corresponds to year 0, cash_flows[1] to year 1, etc.
    Cash flows are discounted as: CF_t / (1 + r)^t.
    
    Parameters
    ----------
    cash_flows : np.ndarray
        Array of yearly cash flows (year 0 to N).
    discount_rate : float
        Annual discount rate (e.g., 0.035 for 3.5%).
        
    Returns
    -------
    float
        Net present value of the cash flows.
    """
    if len(cash_flows) == 0:
        return 0.0
    years = np.arange(len(cash_flows))
    return np.sum(cash_flows / (1 + discount_rate) ** years)


def irr(cash_flows: np.ndarray, guess: float = 0.1, max_rate: float = 1.0) -> float:
    """
    Compute Internal Rate of Return (IRR) using a year-0 cash flow convention.
    
    Parameters
    ----------
    cash_flows : np.ndarray
        Array of cash flows (year 0 to N).
    guess : float, optional
        Initial guess for the IRR (used to seed the bracket).
    max_rate : float, optional
        Maximum rate to search when bracketing a root.
        
    Returns
    -------
    float
        Internal rate of return as a decimal.
        
    Raises
    ------
    ValueError
        If a valid IRR cannot be bracketed.
    """
    cash_flows = np.asarray(cash_flows, dtype=float)
    if cash_flows.size == 0:
        raise ValueError("cash_flows must not be empty")

    years = np.arange(cash_flows.size)

    def npv_at(rate: float) -> float:
        return np.sum(cash_flows / (1 + rate) ** years)

    if npv_at(0.0) == 0.0:
        return 0.0

    low = -0.9
    high = max(guess, 0.1)
    npv_low = npv_at(low)
    npv_high = npv_at(high)

    while npv_low * npv_high > 0 and high < max_rate:
        high = min(max_rate, high * 2 + 0.1)
        npv_high = npv_at(high)

    if npv_low * npv_high > 0:
        raise ValueError("IRR not bracketed for the given cash flows")

    for _ in range(100):
        mid = 0.5 * (low + high)
        npv_mid = npv_at(mid)
        if abs(npv_mid) < 1e-8:
            return mid
        if npv_low * npv_mid <= 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid

    return 0.5 * (low + high)





def combined_wacc_from_capex(
    capex_pv: float,
    capex_bess: float,
    wacc_pv: float,
    wacc_bess: float,
) -> float:
    """
    Compute a project WACC as CAPEX-weighted average of PV and BESS WACCs.
    
    Parameters
    ----------
    capex_pv : float
        PV capital expenditure (EUR).
    capex_bess : float
        BESS capital expenditure (EUR).
    wacc_pv : float
        PV weighted average cost of capital (e.g., 0.035 for 3.5%).
    wacc_bess : float
        BESS weighted average cost of capital (e.g., 0.025 for 2.5%).
        
    Returns
    -------
    float
        Combined project WACC.
        
    Raises
    ------
    ValueError
        If total CAPEX is not positive.
    """
    total = capex_pv + capex_bess
    if total <= 0:
        raise ValueError("Total CAPEX must be positive")
    w_pv = capex_pv / total
    w_bess = capex_bess / total
    return w_pv * wacc_pv + w_bess * wacc_bess


def build_cash_flows_pv_bess(
    annual_revenue_year1: float,
    annual_opex_pv: float,
    annual_opex_bess: float,
    project_lifetime_years: int,
    degradation_pv: float,
    annual_revenue_year1_merchant: Optional[float] = None,
    premium_years: Optional[int] = None,
) -> np.ndarray:
    """
    Build yearly net operating cash flows for a greenfield PV+BESS plant.
    
    PV-driven revenue degrades with 'degradation_pv' per year.
    Battery revenue follows PV degradation (no separate degradation).
    Fixed OPEX (PV and BESS) is constant in real terms.
    
    Initial CAPEX and battery replacement CAPEX are handled outside this function.
    
    Parameters
    ----------
    annual_revenue_year1 : float
        Total annual revenue in year 1 (PV + BESS exports).
    annual_opex_pv : float
        Annual PV OPEX (constant in real terms).
    annual_opex_bess : float
        Annual BESS OPEX (constant in real terms).
    project_lifetime_years : int
        Project evaluation period in years.
    degradation_pv : float
        Annual PV degradation rate (e.g., 0.0025 for 0.25%/year).
    annual_revenue_year1_merchant : float, optional
        Merchant revenue in year 1 (used after premium support ends).
    premium_years : int, optional
        Number of years the premium applies before switching to merchant revenue.
        
    Returns
    -------
    np.ndarray
        Array of net operating cash flows for years 1 to project_lifetime_years.
        Length = project_lifetime_years.
    """
    cash_flows = np.zeros(project_lifetime_years)
    total_opex = annual_opex_pv + annual_opex_bess
    use_merchant_revenue = (
        annual_revenue_year1_merchant is not None
        and premium_years is not None
        and premium_years > 0
    )
    
    for y in range(1, project_lifetime_years + 1):
        revenue_base = annual_revenue_year1
        if use_merchant_revenue and y > premium_years:
            revenue_base = annual_revenue_year1_merchant
        rev_y = revenue_base * (1.0 - degradation_pv) ** (y - 1)
        cash_flows[y - 1] = rev_y - total_opex
    
    return cash_flows


def build_cash_flows_single_tech(
    annual_revenue_year1: float,
    annual_opex: float,
    lifetime_years: int,
    wacc: float,
    degradation_rate: float = 0.0,
) -> np.ndarray:
    """
    Build a vector of yearly net cash flows for a single technology plant.
    
    Parameters
    ----------
    annual_revenue_year1 : float
        Revenue in year 1 (before degradation).
    annual_opex : float
        Fixed OPEX per year (constant over lifetime).
    lifetime_years : int
        Project lifetime in years.
    wacc : float
        Real discount rate (used only for reference, not applied here).
    degradation_rate : float, optional
        Fractional annual degradation applied to revenue only (default 0.0).
        
    Returns
    -------
    np.ndarray
        Array of net cash flows for years 1 to lifetime_years.
        cash_flow[y-1] = revenue_y - annual_opex
        where revenue_y = annual_revenue_year1 * (1 - degradation_rate)^(y-1)
    """
    cash_flows = np.zeros(lifetime_years)
    for y in range(1, lifetime_years + 1):
        revenue_y = annual_revenue_year1 * (1 - degradation_rate) ** (y - 1)
        cash_flows[y - 1] = revenue_y - annual_opex
    return cash_flows


def build_cash_flows_hybrid_pv_wind(
    annual_revenue_pv_year1: float,
    annual_revenue_wind_year1: float,
    annual_opex_pv: float,
    annual_opex_wind: float,
    lifetime_pv_years: int,
    lifetime_wind_years: int,
    wacc: float,
    pv_degradation_rate: float = 0.0025,
) -> np.ndarray:
    """
    Build yearly net cash flows for a hybrid PV + wind plant.

    PV revenue degrades over time, wind revenue is kept constant.
    Each technology contributes revenue and OPEX only during its respective lifetime.

    Parameters
    ----------
    annual_revenue_pv_year1 : float
        PV revenue in year 1 (before degradation).
    annual_revenue_wind_year1 : float
        Wind revenue in year 1 (constant over wind lifetime).
    annual_opex_pv : float
        Annual PV OPEX (constant over PV lifetime).
    annual_opex_wind : float
        Annual wind OPEX (constant over wind lifetime).
    lifetime_pv_years : int
        PV project lifetime in years.
    lifetime_wind_years : int
        Wind project lifetime in years.
    wacc : float
        Real discount rate (for reference, not applied here).
    pv_degradation_rate : float, optional
        Fractional annual degradation applied to PV revenue (default 0.0025).

    Returns
    -------
    np.ndarray
        Array of net cash flows for years 1 to max(lifetime_pv_years, lifetime_wind_years).
    """
    n_years = max(lifetime_pv_years, lifetime_wind_years)
    cash_flows = np.zeros(n_years)
    
    for y in range(1, n_years + 1):
        # PV revenue degrades, only within PV lifetime
        rev_pv = (annual_revenue_pv_year1 * (1 - pv_degradation_rate) ** (y - 1)
                  if y <= lifetime_pv_years else 0.0)
        # Wind revenue stays constant, only within wind lifetime
        rev_wind = annual_revenue_wind_year1 if y <= lifetime_wind_years else 0.0
        # OPEX only applies during respective lifetimes
        opex_pv = annual_opex_pv if y <= lifetime_pv_years else 0.0
        opex_wind = annual_opex_wind if y <= lifetime_wind_years else 0.0
        cash_flows[y - 1] = rev_pv + rev_wind - (opex_pv + opex_wind)
    
    return cash_flows


def compute_annual_revenue(sim_df: pd.DataFrame) -> float:
    """
    Compute annual revenue from simulation results.
    
    Assumes sim_df has been merged with price data and contains:
    - 'export_mw': Exported power in each hour
    - 'price_eur_per_mwh': Price in EUR/MWh for each hour
    
    Args:
        sim_df: DataFrame with export and price columns
        
    Returns:
        Annual revenue in EUR
    """
    if 'price_eur_per_mwh' not in sim_df.columns:
        raise ValueError("sim_df must contain 'price_eur_per_mwh' column")
    
    if 'export_mw' not in sim_df.columns:
        raise ValueError("sim_df must contain 'export_mw' column")
    
    # Calculate hourly revenue: export (MWh) * price (EUR/MWh)
    hourly_revenue = sim_df['export_mw'] * sim_df['price_eur_per_mwh']
    annual_revenue = hourly_revenue.sum()
    
    return annual_revenue


def evaluate_project(
    sim_df: pd.DataFrame,
    capex_eur: float,
    opex_eur_per_year: float,
    lifetime_years: int,
    discount_rate: float,
    degradation_rate: float = 0.0,
    config: Optional[Config] = None
) -> Dict[str, float]:
    """
    Evaluate project economics and return key financial indicators.
    
    Args:
        sim_df: DataFrame with 'export_mw' and 'price_eur_per_mwh' columns
        capex_eur: Total capital expenditure in EUR
        opex_eur_per_year: Annual operational expenditure in EUR
        lifetime_years: Project lifetime in years
        discount_rate: Annual discount rate (e.g., 0.05 for 5%)
        degradation_rate: Annual generation degradation rate (e.g., 0.0025 for 0.25%/yr).
            Applied to LCOE calculation. Default 0.0 (no degradation, e.g., for wind).
        config: Optional Config object (currently unused, for future use)
        
    Returns:
        Dictionary with keys:
        - 'annual_revenue_eur': Annual revenue from energy sales
        - 'annual_cash_flow_eur': Annual revenue minus OPEX
        - 'npv_eur': Net Present Value
        - 'irr': Internal Rate of Return (as decimal)
        - 'simple_payback_years': Simple payback period
        - 'lcoe_eur_per_mwh': Levelized Cost of Energy (accounts for degradation)
    """
    # Compute annual revenue
    annual_revenue = compute_annual_revenue(sim_df)
    
    # Annual cash flow (revenue - OPEX)
    annual_cash_flow = annual_revenue - opex_eur_per_year
    
    # Build cash flow array: initial investment (negative) + annual cash flows
    cash_flows = np.zeros(lifetime_years + 1)
    cash_flows[0] = -capex_eur  # Initial investment
    cash_flows[1:] = annual_cash_flow  # Annual cash flows
    
    # Compute NPV and IRR
    npv_val = npv(cash_flows, discount_rate)
    try:
        irr_val = irr(cash_flows)
    except ValueError:
        irr_val = np.nan
    
    # Simple payback period
    cumulative_cash_flow = 0.0
    payback_years = None
    for year in range(1, lifetime_years + 1):
        cumulative_cash_flow += annual_cash_flow
        if cumulative_cash_flow >= capex_eur:
            payback_years = year
            break
    
    if payback_years is None:
        payback_years = np.inf
    
    # Levelized Cost of Energy (LCOE)
    # LCOE = (CAPEX + sum(OPEX/(1+r)^t)) / sum(Generation_t/(1+r)^t)
    # where Generation_t = Generation_year1 * (1 - degradation_rate)^(t-1)
    total_discounted_cost = capex_eur
    for year in range(1, lifetime_years + 1):
        total_discounted_cost += opex_eur_per_year / (1 + discount_rate) ** year
    
    # Total discounted generation (with degradation applied)
    annual_generation_year1_mwh = sim_df['export_mw'].sum()
    total_discounted_generation = 0.0
    for year in range(1, lifetime_years + 1):
        generation_year_t = annual_generation_year1_mwh * (1 - degradation_rate) ** (year - 1)
        total_discounted_generation += generation_year_t / (1 + discount_rate) ** year
    
    if total_discounted_generation > 0:
        lcoe = total_discounted_cost / total_discounted_generation
    else:
        lcoe = np.nan
    
    return {
        'annual_revenue_eur': annual_revenue,
        'annual_cash_flow_eur': annual_cash_flow,
        'npv_eur': npv_val,
        'irr': irr_val,
        'simple_payback_years': payback_years,
        'lcoe_eur_per_mwh': lcoe
    }


@dataclass
class PvBessNpvResult:
    """
    Result container for PV+BESS NPV analysis.
    
    Contains detailed breakdown of costs, revenues, and financial metrics.
    """
    # CAPEX breakdown
    capex_pv_eur: float
    capex_bess_eur: float
    capex_total_eur: float
    
    # Annual values (Year 1)
    annual_revenue_year1_eur: float
    annual_opex_pv_eur: float
    annual_opex_bess_eur: float
    annual_opex_total_eur: float
    annual_net_cashflow_year1_eur: float
    
    # Battery replacement
    bess_replacement_year: int
    bess_replacement_cost_eur: float
    
    # Financial metrics
    npv_eur: float
    irr: float
    simple_payback_years: float
    
    # Cash flows
    cash_flows: np.ndarray
    discounted_cash_flows: np.ndarray


def compute_pv_bess_npv(
    pv_capacity_mw: float,
    batt_power_mw: float,
    batt_duration_h: float,
    annual_revenue_year1: float,
    capex_pv_eur_per_kw: float = 700.0,
    opex_pv_eur_per_kw_per_year: float = 13.3,
    capex_bess_eur_per_kwh: float = 400.0,
    opex_bess_eur_per_kw_per_year: float = 5.3,
    lifetime_pv_years: int = 30,
    lifetime_bess_years: int = 15,
    wacc_pv: float = 0.035,
    wacc_bess: float = 0.025,
    degradation_pv_per_year: float = 0.0025,
    bess_replacement_cost_fraction: float = 0.30,
) -> PvBessNpvResult:
    """
    Compute NPV for a PV + BESS hybrid project.
    
    Handles different lifetimes for PV and BESS, with battery replacement
    at the end of BESS lifetime (year 15). Uses a blended WACC based on
    capital proportions.
    
    Parameters
    ----------
    pv_capacity_mw : float
        Installed PV capacity in MW.
    batt_power_mw : float
        Battery power rating in MW.
    batt_duration_h : float
        Battery duration in hours (energy = power * duration).
    annual_revenue_year1 : float
        Total annual revenue in year 1 (PV + BESS exports).
    capex_pv_eur_per_kw : float, default 700.0
        PV CAPEX in €/kW.
    opex_pv_eur_per_kw_per_year : float, default 13.3
        PV annual OPEX in €/kW/year.
    capex_bess_eur_per_kwh : float, default 400.0
        BESS CAPEX in €/kWh.
    opex_bess_eur_per_kw_per_year : float, default 5.3
        BESS annual OPEX in €/kW/year.
    lifetime_pv_years : int, default 30
        PV project lifetime in years.
    lifetime_bess_years : int, default 15
        BESS lifetime in years (replacement occurs at this point).
    wacc_pv : float, default 0.035
        PV real WACC (3.5%).
    wacc_bess : float, default 0.025
        BESS real WACC (2.5%).
    degradation_pv_per_year : float, default 0.0025
        Annual PV degradation rate (0.25% per year).
    bess_replacement_cost_fraction : float, default 0.30
        Battery replacement cost as fraction of initial BESS CAPEX.
        
    Returns
    -------
    PvBessNpvResult
        Dataclass with detailed financial metrics and cash flows.
        
    Notes
    -----
    - Project lifetime is equal to PV lifetime (30 years).
    - Battery is replaced at year 15 with 30% of initial BESS CAPEX.
    - Revenue degrades annually due to PV degradation.
    - BESS OPEX continues throughout project lifetime (battery is replaced).
    - A blended WACC is used based on initial capital proportions.
    """
    # Convert capacities to kW/kWh
    pv_capacity_kw = pv_capacity_mw * 1000.0
    batt_power_kw = batt_power_mw * 1000.0
    batt_energy_kwh = batt_power_mw * batt_duration_h * 1000.0
    
    # Calculate CAPEX
    capex_pv = pv_capacity_kw * capex_pv_eur_per_kw
    capex_bess = batt_energy_kwh * capex_bess_eur_per_kwh
    capex_total = capex_pv + capex_bess
    
    # Calculate annual OPEX
    opex_pv_annual = pv_capacity_kw * opex_pv_eur_per_kw_per_year
    # Updated to use power-based OPEX to match hybrid_bess_analysis.py
    opex_bess_annual = batt_power_kw * opex_bess_eur_per_kw_per_year
    opex_total_annual = opex_pv_annual + opex_bess_annual
    
    # Battery replacement cost
    bess_replacement_cost = capex_bess * bess_replacement_cost_fraction
    bess_replacement_year = lifetime_bess_years
    
    # Calculate blended WACC based on initial capital proportions
    weight_pv = capex_pv / capex_total if capex_total > 0 else 0.5
    weight_bess = capex_bess / capex_total if capex_total > 0 else 0.5
    wacc_blended = weight_pv * wacc_pv + weight_bess * wacc_bess
    
    # Project lifetime is PV lifetime
    n_years = lifetime_pv_years
    
    # Build cash flows
    # Year 0: Initial CAPEX (negative)
    # Years 1-N: Revenue - OPEX (with PV degradation)
    # Year 15: Battery replacement cost
    cash_flows = np.zeros(n_years + 1)
    cash_flows[0] = -capex_total
    
    for y in range(1, n_years + 1):
        # Revenue degrades due to PV degradation
        # Note: We apply degradation to total revenue as it's dominated by PV
        revenue_y = annual_revenue_year1 * (1 - degradation_pv_per_year) ** (y - 1)
        
        # OPEX (both PV and BESS OPEX continue)
        opex_y = opex_total_annual
        
        # Net cash flow
        cash_flows[y] = revenue_y - opex_y
        
        # Battery replacement at year 15
        if y == bess_replacement_year:
            cash_flows[y] -= bess_replacement_cost
    
    # Compute discounted cash flows
    years = np.arange(n_years + 1)
    discounted_cash_flows = cash_flows / (1 + wacc_blended) ** years

    # NPV and IRR
    npv_val = npv(cash_flows, wacc_blended)
    try:
        irr_val = irr(cash_flows, guess=0.05)
    except ValueError:
        irr_val = np.nan
    
    # Simple payback (undiscounted)
    cumulative = 0.0
    payback_years = np.inf
    for y in range(1, n_years + 1):
        cumulative += cash_flows[y]
        if cumulative >= capex_total:
            # Interpolate for fractional year
            excess = cumulative - capex_total
            payback_years = y - (excess / cash_flows[y]) if cash_flows[y] != 0 else y
            break
    
    return PvBessNpvResult(
        capex_pv_eur=capex_pv,
        capex_bess_eur=capex_bess,
        capex_total_eur=capex_total,
        annual_revenue_year1_eur=annual_revenue_year1,
        annual_opex_pv_eur=opex_pv_annual,
        annual_opex_bess_eur=opex_bess_annual,
        annual_opex_total_eur=opex_total_annual,
        annual_net_cashflow_year1_eur=annual_revenue_year1 - opex_total_annual,
        bess_replacement_year=bess_replacement_year,
        bess_replacement_cost_eur=bess_replacement_cost,
        npv_eur=npv_val,
        irr=irr_val,
        simple_payback_years=payback_years,
        cash_flows=cash_flows,
        discounted_cash_flows=discounted_cash_flows,
    )


def compute_pv_bess_npv_grid(
    df: pd.DataFrame,
    pv_capacities_mw: list,
    batt_powers_mw: list,
    batt_durations_h: list,
    simulate_func,
    **npv_kwargs,
) -> pd.DataFrame:
    """
    Compute NPV for a grid of PV+BESS configurations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hourly data for simulation.
    pv_capacities_mw : list
        List of PV capacities to test (MW).
    batt_powers_mw : list
        List of battery powers to test (MW).
    batt_durations_h : list
        List of battery durations to test (hours).
    simulate_func : callable
        Simulation function that returns annual PV+BESS metrics.
    **npv_kwargs
        Additional arguments passed to compute_pv_bess_npv.
        
    Returns
    -------
    pd.DataFrame
        Results grid with columns for configuration and financial metrics.
    """
    results = []
    
    for pv_cap in pv_capacities_mw:
        for batt_pwr in batt_powers_mw:
            for batt_dur in batt_durations_h:
                # Run simulation
                sim_result = simulate_func(
                    df=df,
                    pv_capacity_mw=pv_cap,
                    batt_power_mw=batt_pwr,
                    batt_duration_h=batt_dur,
                )
                
                # Compute NPV
                npv_result = compute_pv_bess_npv(
                    pv_capacity_mw=pv_cap,
                    batt_power_mw=batt_pwr,
                    batt_duration_h=batt_dur,
                    annual_revenue_year1=sim_result.annual_revenue_eur,
                    opex_bess_eur_per_kw_per_year=npv_kwargs.get('opex_bess_eur_per_kw_per_year', 5.3),
                    **{k: v for k, v in npv_kwargs.items() if k != 'opex_bess_eur_per_kw_per_year'}
                )
                
                results.append({
                    'pv_capacity_mw': pv_cap,
                    'batt_power_mw': batt_pwr,
                    'batt_duration_h': batt_dur,
                    'batt_energy_mwh': batt_pwr * batt_dur,
                    'annual_revenue_eur': sim_result.annual_revenue_eur,
                    'annual_export_mwh': sim_result.annual_export_mwh_total,
                    'annual_curtailment_mwh': sim_result.annual_curtailment_mwh,
                    'capex_total_eur': npv_result.capex_total_eur,
                    'capex_pv_eur': npv_result.capex_pv_eur,
                    'capex_bess_eur': npv_result.capex_bess_eur,
                    'opex_total_eur': npv_result.annual_opex_total_eur,
                    'npv_eur': npv_result.npv_eur,
                    'irr': npv_result.irr,
                    'simple_payback_years': npv_result.simple_payback_years,
                })
    
    return pd.DataFrame(results)
