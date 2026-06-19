"""
Oversizing analysis for stand-alone PV and wind plants at a fixed POI.

This module provides functions to evaluate the optimal oversizing factor
for each technology based on NPV analysis.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from config import (
    POI_CAPACITY_MW,
    CAPEX_PV_EUR_PER_KW,
    OPEX_PV_EUR_PER_KW_PER_YEAR,
    LIFETIME_PV_YEARS,
    WACC_PV_REAL,
    DEGRADATION_PV_PER_YEAR,
    CAPEX_WIND_EUR_PER_KW,
    OPEX_WIND_EUR_PER_KW_PER_YEAR,
    LIFETIME_WIND_YEARS,
    WACC_WIND_REAL,
    DEGRADATION_WIND_PER_YEAR,
)
from economics import irr, npv, build_cash_flows_single_tech
from simulation import simulate_single_tech_year


@dataclass
class OversizingResult:
    """Result container for a single oversizing evaluation."""
    
    tech: str                    # 'pv' or 'wind'
    alpha: float                 # Oversizing factor (installed / POI)
    capacity_mw: float           # Installed capacity in MW
    npv_eur: float               # Net Present Value in EUR
    irr: float                   # Internal Rate of Return
    annual_revenue_eur: float    # Annual revenue (year 1) in EUR
    annual_export_mwh: float     # Annual exported energy in MWh
    annual_curtailment_mwh: float  # Annual curtailed energy in MWh
    # System-oriented metrics
    export_per_mw_poi_mwh: float   # Annual export per MW of POI capacity
    poi_utilisation_factor: float  # POI utilisation (0-1), export / max possible export
    # Financial parameters (added for consistency with other modules)
    capex_total_eur: float        # Total CAPEX in EUR
    annual_opex_total_eur: float  # Total annual OPEX in EUR
    wacc: float                   # Discount rate used for NPV
    lifetime_years: int           # Project lifetime in years


def get_tech_parameters(tech: str) -> dict:
    """
    Get techno-economic parameters for a given technology.
    
    Parameters
    ----------
    tech : str
        Technology type: 'pv' or 'wind'.
        
    Returns
    -------
    dict
        Dictionary with keys: capex_eur_per_kw, opex_eur_per_kw_per_year,
        lifetime_years, wacc, degradation_rate.
    """
    if tech == "pv":
        return {
            "capex_eur_per_kw": CAPEX_PV_EUR_PER_KW,
            "opex_eur_per_kw_per_year": OPEX_PV_EUR_PER_KW_PER_YEAR,
            "lifetime_years": LIFETIME_PV_YEARS,
            "wacc": WACC_PV_REAL,
            "degradation_rate": DEGRADATION_PV_PER_YEAR,
        }
    elif tech == "wind":
        return {
            "capex_eur_per_kw": CAPEX_WIND_EUR_PER_KW,
            "opex_eur_per_kw_per_year": OPEX_WIND_EUR_PER_KW_PER_YEAR,
            "lifetime_years": LIFETIME_WIND_YEARS,
            "wacc": WACC_WIND_REAL,
            "degradation_rate": DEGRADATION_WIND_PER_YEAR,
        }
    else:
        raise ValueError(f"Unknown technology: {tech}")


def evaluate_oversizing_grid(
    df: pd.DataFrame,
    tech: str,
    alphas: List[float],
    poi_capacity_mw: float = POI_CAPACITY_MW,
) -> List[OversizingResult]:
    """
    Evaluate NPV and related metrics over a grid of oversizing factors.
    
    For each oversizing factor alpha, computes the installed capacity,
    simulates one year of operation, and calculates the project NPV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hourly columns: 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    tech : str
        Technology type: 'pv' or 'wind'.
    alphas : List[float]
        List of oversizing factors to evaluate.
        alpha = P_installed_mw / POI_CAPACITY_MW.
    poi_capacity_mw : float, optional
        POI capacity in MW (default: 10.0).
        
    Returns
    -------
    List[OversizingResult]
        List of results for each oversizing factor.
    """
    params = get_tech_parameters(tech)
    results = []
    
    for alpha in alphas:
        # Installed capacity
        capacity_mw = alpha * poi_capacity_mw
        capacity_kw = capacity_mw * 1000.0
        
        # Simulate one year
        sim_result = simulate_single_tech_year(
            df=df,
            tech=tech,
            capacity_mw=capacity_mw,
            poi_capacity_mw=poi_capacity_mw,
        )
        
        annual_revenue_eur = sim_result["annual_revenue_eur"]
        annual_export_mwh = sim_result["annual_export_mwh"]
        annual_curtailment_mwh = sim_result["annual_curtailment_mwh"]
        
        # Fixed OPEX per year
        annual_opex = capacity_kw * params["opex_eur_per_kw_per_year"]
        
        # Build yearly cash flows (excluding CAPEX)
        cash_flows = build_cash_flows_single_tech(
            annual_revenue_year1=annual_revenue_eur,
            annual_opex=annual_opex,
            lifetime_years=params["lifetime_years"],
            wacc=params["wacc"],
            degradation_rate=params["degradation_rate"],
        )
        
        # Initial CAPEX
        initial_capex = capacity_kw * params["capex_eur_per_kw"]
        
        # Compute NPV: use Year-0 convention
        full_cash_flows = np.insert(cash_flows, 0, -initial_capex)
        npv_total = npv(full_cash_flows, params["wacc"])

        try:
            irr_val = irr(full_cash_flows)
        except ValueError:
            irr_val = np.nan
        
        # System-oriented metrics
        export_per_mw_poi = annual_export_mwh / poi_capacity_mw
        poi_utilisation = annual_export_mwh / (poi_capacity_mw * 8760.0)
        
        results.append(
            OversizingResult(
                tech=tech,
                alpha=alpha,
                capacity_mw=capacity_mw,
                npv_eur=npv_total,
                irr=irr_val,
                annual_revenue_eur=annual_revenue_eur,
                annual_export_mwh=annual_export_mwh,
                annual_curtailment_mwh=annual_curtailment_mwh,
                export_per_mw_poi_mwh=export_per_mw_poi,
                poi_utilisation_factor=poi_utilisation,
                capex_total_eur=initial_capex,
                annual_opex_total_eur=annual_opex,
                wacc=params["wacc"],
                lifetime_years=params["lifetime_years"],
            )
        )
    
    return results


def find_optimal_alpha(results: List[OversizingResult]) -> OversizingResult:
    """
    Find the entry with the highest NPV.
    
    Parameters
    ----------
    results : List[OversizingResult]
        List of oversizing results for a single technology.
        
    Returns
    -------
    OversizingResult
        The result with the maximum NPV.
        
    Raises
    ------
    ValueError
        If the results list is empty.
    """
    if not results:
        raise ValueError("Results list cannot be empty")
    return max(results, key=lambda r: r.npv_eur)


def results_to_dataframe(results: List[OversizingResult]) -> pd.DataFrame:
    """
    Convert a list of OversizingResult objects to a DataFrame.
    
    Parameters
    ----------
    results : List[OversizingResult]
        List of oversizing results.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching OversizingResult fields.
    """
    return pd.DataFrame([
        {
            "tech": r.tech,
            "alpha": r.alpha,
            "capacity_mw": r.capacity_mw,
            "npv_eur": r.npv_eur,
            "irr": r.irr,
            "annual_revenue_eur": r.annual_revenue_eur,
            "annual_export_mwh": r.annual_export_mwh,
            "annual_curtailment_mwh": r.annual_curtailment_mwh,
            "export_per_mw_poi_mwh": r.export_per_mw_poi_mwh,
            "poi_utilisation_factor": r.poi_utilisation_factor,
            "capex_total_eur": r.capex_total_eur,
            "annual_opex_total_eur": r.annual_opex_total_eur,
            "wacc": r.wacc,
            "lifetime_years": r.lifetime_years,
        }
        for r in results
    ])

