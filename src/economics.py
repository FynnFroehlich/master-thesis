"""
Economic evaluation functions for renewable energy projects.
"""

import numpy as np
from typing import Optional


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
