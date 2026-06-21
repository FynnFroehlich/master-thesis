"""
Greenfield PV + BESS analysis module.

This module provides tools for analyzing greenfield PV + battery energy storage
system (BESS) projects behind a fixed point of interconnection (POI).

The battery is a "green battery" that can only charge from on-site PV generation.
"""

from dataclasses import dataclass
from typing import List, Sequence, Optional

import numpy as np
import pandas as pd

from config import (
    POI_CAPACITY_MW,
    CAPEX_PV_EUR_PER_KW,
    CAPEX_BESS_EUR_PER_KWH,
    OPEX_PV_EUR_PER_KW_PER_YEAR,
    OPEX_BESS_EUR_PER_KW_PER_YEAR,
    LIFETIME_PV_YEARS,
    LIFETIME_BESS_YEARS,
    WACC_PV_REAL,
    WACC_BESS_REAL,
    DEGRADATION_PV_PER_YEAR,
    PROJECT_LIFETIME_YEARS,
    BESS_REPLACEMENT_SHARE,
    BESS_ROUNDTRIP_EFF,
)
from simulation_lp import simulate_pv_bess_lp, simulate_pv_bess_lp_innovationsausschreibung
from economics import (
    build_cash_flows_pv_bess,
    combined_wacc_from_capex,
    npv,
    irr,
)
from config import (
    INNOVATION_TENDER_MIN_BESS_POWER_RATIO,
    INNOVATION_TENDER_MIN_BESS_DURATION_H,
    INNOVATION_TENDER_PREMIUM_YEARS,
)


@dataclass
class PvBessResult:
    """
    Result container for a single PV+BESS greenfield configuration.
    
    Attributes
    ----------
    alpha_pv : float
        PV overplanting factor (P_pv / P_POI).
    gamma : float
        Battery power ratio (P_batt / P_POI).
    pv_capacity_mw : float
        Installed PV capacity (MW).
    batt_power_mw : float
        Battery power rating (MW).
    batt_energy_mwh : float
        Battery energy capacity (MWh).
    wacc : float
        Combined project WACC (CAPEX-weighted).
    npv_eur : float
        Net Present Value (EUR).
    annual_revenue_eur : float
        Year 1 annual revenue (EUR).
    annual_export_mwh_total : float
        Total annual energy exported (MWh).
    annual_export_mwh_pv : float
        Annual PV direct export (MWh).
    annual_export_mwh_bess : float
        Annual BESS export (MWh).
    annual_curtailment_mwh : float
        Annual curtailed PV energy (MWh).
    poi_utilisation : float
        POI utilization factor (0-1).
    storage_throughput_mwh : float
        Annual battery discharge throughput (MWh).
    e_per_kwp_storage_kwh : float
        Battery energy per kW_p PV (kWh/kW_p).
    capex_pv_eur : float
        PV CAPEX (EUR).
    capex_bess_eur : float
        BESS CAPEX (EUR).
    capex_total_eur : float
        Total CAPEX (EUR).
    annual_opex_pv_eur : float
        Annual PV OPEX (EUR).
    annual_opex_bess_eur : float
        Annual BESS OPEX (EUR).
    irr : float
        Internal Rate of Return (if computable).
    """
    alpha_pv: float
    gamma: float
    pv_capacity_mw: float
    batt_power_mw: float
    batt_energy_mwh: float
    wacc: float
    npv_eur: float
    annual_revenue_eur: float
    annual_export_mwh_total: float
    annual_export_mwh_pv: float
    annual_export_mwh_bess: float
    annual_curtailment_mwh: float
    poi_utilisation: float
    storage_throughput_mwh: float
    e_per_kwp_storage_kwh: float
    capex_pv_eur: float = 0.0
    capex_bess_eur: float = 0.0
    capex_total_eur: float = 0.0
    annual_opex_pv_eur: float = 0.0
    annual_opex_bess_eur: float = 0.0
    annual_opex_total_eur: float = 0.0
    irr: float = 0.0
    annual_revenue_eur_merchant: float = float("nan")  # yr1 merchant revenue for premium->merchant transition; NaN for non-innovation


def evaluate_pv_bess_grid(
    df: pd.DataFrame,
    alpha_pv_values: Sequence[float],
    gamma_values: Sequence[float],
    batt_duration_h: float = 2.0,
    poi_capacity_mw: float = POI_CAPACITY_MW,
    capex_pv_eur_per_kw: float = CAPEX_PV_EUR_PER_KW,
    capex_bess_eur_per_kwh: float = CAPEX_BESS_EUR_PER_KWH,
    opex_pv_eur_per_kw_per_year: float = OPEX_PV_EUR_PER_KW_PER_YEAR,
    opex_bess_eur_per_kw_per_year: float = OPEX_BESS_EUR_PER_KW_PER_YEAR,
    wacc_pv: float = WACC_PV_REAL,
    wacc_bess: float = WACC_BESS_REAL,
    degradation_pv: float = DEGRADATION_PV_PER_YEAR,
    project_lifetime_years: int = PROJECT_LIFETIME_YEARS,
    bess_replacement_share: float = BESS_REPLACEMENT_SHARE,
    roundtrip_efficiency: float = BESS_ROUNDTRIP_EFF,
    solver: Optional[str] = None,
    time_limit_seconds: int = 300,
) -> List[PvBessResult]:
    """
    Greenfield PV+BESS design sweep at a fixed POI.
    
    Evaluates a grid of PV and battery configurations defined by:
        - alpha_pv = P_pv / P_POI (PV overplanting factor)
        - gamma = P_batt / P_POI (battery power ratio)
    
    Battery duration is fixed (default 2 hours).
    The battery is a green battery: it can only charge from on-site PV.
    Dispatch uses LP optimization (perfect foresight) rather than heuristics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns 'timestamp', 'price_eur_per_mwh', 'pv_per_kw'.
    alpha_pv_values : Sequence[float]
        PV overplanting factors to evaluate (e.g., [1.0, 1.5, 2.0]).
    gamma_values : Sequence[float]
        Battery power ratios to evaluate (e.g., [0.0, 0.2, 0.5, 1.0]).
    batt_duration_h : float, default 2.0
        Battery duration at rated power (hours).
    poi_capacity_mw : float, default 10.0
        Point of interconnection capacity (MW).
    capex_pv_eur_per_kw : float, default 700.0
        PV CAPEX (EUR/kW).
    capex_bess_eur_per_kwh : float, default 400.0
        BESS CAPEX (EUR/kWh).
    opex_pv_eur_per_kw_per_year : float, default 13.3
        PV annual OPEX (EUR/kW/year).
    opex_bess_eur_per_kw_per_year : float, default 5.3
        BESS annual OPEX (EUR/kW power/year).
    wacc_pv : float, default 0.035
        PV real WACC.
    wacc_bess : float, default 0.025
        BESS real WACC.
    degradation_pv : float, default 0.0025
        Annual PV degradation rate.
    project_lifetime_years : int, default 30
        Project evaluation period (years).
    bess_replacement_share : float, default 0.30
        Battery replacement cost as share of initial BESS CAPEX.
    roundtrip_efficiency : float, default 0.90
        Battery round-trip efficiency.
    solver : str, optional
        PuLP solver name (e.g., 'CBC', 'GUROBI', 'CPLEX').
    time_limit_seconds : int, default 300
        Maximum solver time per configuration (seconds).
        
    Returns
    -------
    List[PvBessResult]
        List of results for each (alpha_pv, gamma) combination.
    """
    if 'pv_per_kw' not in df.columns:
        raise ValueError("DataFrame must contain 'pv_per_kw' column")
    if 'price_eur_per_mwh' not in df.columns:
        raise ValueError("DataFrame must contain 'price_eur_per_mwh' column")

    pv_per_kw = df['pv_per_kw'].values
    prices = df['price_eur_per_mwh'].values

    results: List[PvBessResult] = []
    
    for alpha_pv in alpha_pv_values:
        for gamma in gamma_values:
            # 1. Capacities
            pv_capacity_mw = alpha_pv * poi_capacity_mw
            batt_power_mw = gamma * poi_capacity_mw
            
            if pv_capacity_mw <= 0:
                continue
            
            batt_energy_mwh = batt_power_mw * batt_duration_h
            
            # 2. CAPEX and OPEX
            capex_pv = pv_capacity_mw * 1000.0 * capex_pv_eur_per_kw
            capex_bess = batt_energy_mwh * 1000.0 * capex_bess_eur_per_kwh
            
            annual_opex_pv = pv_capacity_mw * 1000.0 * opex_pv_eur_per_kw_per_year
            annual_opex_bess = batt_power_mw * 1000.0 * opex_bess_eur_per_kw_per_year
            
            # 3. Scenario WACC
            if capex_bess > 0:
                wacc = combined_wacc_from_capex(
                    capex_pv=capex_pv,
                    capex_bess=capex_bess,
                    wacc_pv=wacc_pv,
                    wacc_bess=wacc_bess,
                )
            else:
                wacc = wacc_pv  # PV-only case
            
            # 4. Dispatch simulation (LP optimization)
            if batt_power_mw > 0:
                pv_generation_mw = pv_per_kw * (pv_capacity_mw * 1000.0) / 1000.0
                df_lp_input = pd.DataFrame(
                    {
                        'pv_generation_mw': pv_generation_mw,
                        'day_ahead_price_eur_mwh': prices,
                    },
                    index=df.index,
                )

                eta_ch = np.sqrt(roundtrip_efficiency)
                eta_dis = np.sqrt(roundtrip_efficiency)

                params = {
                    'bess_cap_mwh': batt_energy_mwh,
                    'bess_pwr_mw': batt_power_mw,
                    'poi_mw': poi_capacity_mw,
                    'eta_ch': eta_ch,
                    'eta_dis': eta_dis,
                }

                try:
                    lp_result = simulate_pv_bess_lp(
                        df=df_lp_input,
                        params=params,
                        solver=solver,
                        time_limit_seconds=time_limit_seconds,
                        verbose=False,
                    )
                except Exception as e:
                    print(
                        f"Warning: LP optimization failed for alpha_pv={alpha_pv:.2f}, "
                        f"gamma={gamma:.2f}: {e}"
                    )
                    continue

                annual_revenue_eur = lp_result['revenue_eur'].sum()
                annual_export_mwh_total = lp_result['export_mw'].sum()
                annual_export_mwh_pv = (
                    (lp_result['export_mw'] - lp_result['discharge_mw'])
                    .clip(lower=0)
                    .sum()
                )
                annual_export_mwh_bess = lp_result['discharge_mw'].sum()
                annual_curtailment_mwh = lp_result['curtailment_mw'].sum()
                storage_throughput_mwh = lp_result['discharge_mw'].sum()
            else:
                # PV-only case: use single tech simulation
                from simulation import simulate_single_tech_year
                sim_result = simulate_single_tech_year(
                    df=df,
                    tech='pv',
                    capacity_mw=pv_capacity_mw,
                    poi_capacity_mw=poi_capacity_mw,
                )
                annual_revenue_eur = sim_result['annual_revenue_eur']
                annual_export_mwh_total = sim_result['annual_export_mwh']
                annual_export_mwh_pv = sim_result['annual_export_mwh']
                annual_export_mwh_bess = 0.0
                annual_curtailment_mwh = sim_result['annual_curtailment_mwh']
                storage_throughput_mwh = 0.0
            
            # 5. Annual metrics
            poi_utilisation = annual_export_mwh_total / (poi_capacity_mw * 8760.0)
            
            # 6. Yearly cash flows
            cash_flows = build_cash_flows_pv_bess(
                annual_revenue_year1=annual_revenue_eur,
                annual_opex_pv=annual_opex_pv,
                annual_opex_bess=annual_opex_bess,
                project_lifetime_years=project_lifetime_years,
                degradation_pv=degradation_pv,
            )
            
            # 7. Battery replacement at year 15
            replacement_capex = bess_replacement_share * capex_bess
            if project_lifetime_years >= 15 and batt_power_mw > 0:
                cash_flows[15 - 1] -= replacement_capex
            
            # 8. NPV (Year-0 convention)
            capex_total = capex_pv + capex_bess
            full_cash_flows = np.insert(cash_flows, 0, -capex_total)
            npv_total = npv(full_cash_flows, wacc)
            
            # 9. Storage ratio (kWh per kW_p PV)
            e_per_kwp_storage_kwh = (
                batt_energy_mwh * 1000.0 / (pv_capacity_mw * 1000.0)
                if pv_capacity_mw > 0
                else 0.0
            )
            
            # 10. IRR calculation (Year-0 convention)
            try:
                irr_val = irr(full_cash_flows)
            except ValueError:
                irr_val = np.nan
            
            # Create result
            result = PvBessResult(
                alpha_pv=alpha_pv,
                gamma=gamma,
                pv_capacity_mw=pv_capacity_mw,
                batt_power_mw=batt_power_mw,
                batt_energy_mwh=batt_energy_mwh,
                wacc=wacc,
                npv_eur=npv_total,
                annual_revenue_eur=annual_revenue_eur,
                annual_export_mwh_total=annual_export_mwh_total,
                annual_export_mwh_pv=annual_export_mwh_pv,
                annual_export_mwh_bess=annual_export_mwh_bess,
                annual_curtailment_mwh=annual_curtailment_mwh,
                poi_utilisation=poi_utilisation,
                storage_throughput_mwh=storage_throughput_mwh,
                e_per_kwp_storage_kwh=e_per_kwp_storage_kwh,
                capex_pv_eur=capex_pv,
                capex_bess_eur=capex_bess,
                capex_total_eur=capex_total,
                annual_opex_pv_eur=annual_opex_pv,
                annual_opex_bess_eur=annual_opex_bess,
                annual_opex_total_eur=annual_opex_pv + annual_opex_bess,
                irr=irr_val,
            )
            results.append(result)
    
    return results


def evaluate_pv_bess_grid_innovationsausschreibung(
    df: pd.DataFrame,
    alpha_pv_values: Sequence[float],
    gamma_values: Sequence[float],
    batt_duration_h: float = 2.0,
    poi_capacity_mw: float = POI_CAPACITY_MW,
    capex_pv_eur_per_kw: float = CAPEX_PV_EUR_PER_KW,
    capex_bess_eur_per_kwh: float = CAPEX_BESS_EUR_PER_KWH,
    opex_pv_eur_per_kw_per_year: float = OPEX_PV_EUR_PER_KW_PER_YEAR,
    opex_bess_eur_per_kw_per_year: float = OPEX_BESS_EUR_PER_KW_PER_YEAR,
    wacc_pv: float = WACC_PV_REAL,
    wacc_bess: float = WACC_BESS_REAL,
    degradation_pv: float = DEGRADATION_PV_PER_YEAR,
    project_lifetime_years: int = PROJECT_LIFETIME_YEARS,
    bess_replacement_share: float = BESS_REPLACEMENT_SHARE,
    roundtrip_efficiency: float = BESS_ROUNDTRIP_EFF,
    solver: Optional[str] = None,
    time_limit_seconds: int = 300,
) -> List[PvBessResult]:
    """
    Greenfield PV+BESS design sweep using LP optimization with Innovationsausschreibung 2024.
    
    This function evaluates a grid of PV and battery configurations using the LP optimization
    with Floating Market Premium (FMP) from the Innovation Tender 2024. The simulation uses
    optimal dispatch rather than rule-based heuristics, and applies the premium only for the
    first INNOVATION_TENDER_PREMIUM_YEARS years before switching to merchant revenues.
    
    Evaluates a grid of PV and battery configurations defined by:
        - alpha_pv = P_pv / P_POI (PV overplanting factor)
        - gamma = P_batt / P_POI (battery power ratio)
    
    Battery duration is fixed (default 2 hours, minimum 2 hours for eligibility).
    The battery is a green battery: it can only charge from on-site PV.
    
    BESS sizing constraints for Innovation Tender eligibility:
        - P_BESS >= 0.25 × P_PV
        - E_BESS >= 2 × P_BESS (batt_duration_h >= 2.0)
    
    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns 'timestamp', 'price_eur_per_mwh', 'pv_per_kw'.
        Must have 'timestamp' column for monthly market premium calculation.
    alpha_pv_values : Sequence[float]
        PV overplanting factors to evaluate (e.g., [1.0, 1.5, 2.0]).
    gamma_values : Sequence[float]
        Battery power ratios to evaluate (e.g., [0.0, 0.2, 0.5, 1.0]).
        Note: gamma must be >= 0.25 * alpha_pv for Innovation Tender eligibility.
    batt_duration_h : float, default 2.0
        Battery duration at rated power (hours). Must be >= 2.0 for eligibility.
    poi_capacity_mw : float, default 10.0
        Point of interconnection capacity (MW).
    capex_pv_eur_per_kw : float, default 700.0
        PV CAPEX (EUR/kW).
    capex_bess_eur_per_kwh : float, default 400.0
        BESS CAPEX (EUR/kWh).
    opex_pv_eur_per_kw_per_year : float, default 13.3
        PV annual OPEX (EUR/kW/year).
    opex_bess_eur_per_kw_per_year : float, default 5.3
        BESS annual OPEX (EUR/kW power/year).
    wacc_pv : float, default 0.035
        PV real WACC.
    wacc_bess : float, default 0.025
        BESS real WACC.
    degradation_pv : float, default 0.0025
        Annual PV degradation rate.
    project_lifetime_years : int, default 30
        Project evaluation period (years).
    bess_replacement_share : float, default 0.30
        Battery replacement cost as share of initial BESS CAPEX.
    roundtrip_efficiency : float, default 0.90
        Battery round-trip efficiency.
    solver : str, optional
        PuLP solver name (e.g., 'CBC', 'GUROBI', 'CPLEX').
    time_limit_seconds : int, default 300
        Maximum solver time per configuration (seconds).
        
    Returns
    -------
    List[PvBessResult]
        List of results for each (alpha_pv, gamma) combination.
        Configurations that don't meet BESS sizing requirements are skipped.
    """
    # Ensure batt_duration_h meets minimum requirement
    if batt_duration_h < INNOVATION_TENDER_MIN_BESS_DURATION_H:
        raise ValueError(
            f"Battery duration {batt_duration_h} hours must be >= "
            f"{INNOVATION_TENDER_MIN_BESS_DURATION_H} hours for Innovation Tender eligibility"
        )
    
    # Prepare data for LP simulation
    # Convert pv_per_kw to pv_generation_mw
    if 'pv_per_kw' in df.columns:
        df_lp = df.copy()
        # We'll calculate pv_generation_mw in the loop based on pv_capacity_mw
    else:
        raise ValueError("DataFrame must contain 'pv_per_kw' column")
    
    # Ensure timestamp column exists
    if 'timestamp' not in df_lp.columns:
        if df_lp.index.name == 'timestamp' or isinstance(df_lp.index, pd.DatetimeIndex):
            df_lp = df_lp.reset_index()
        else:
            raise ValueError("DataFrame must have 'timestamp' column for monthly market premium calculation")
    
    results: List[PvBessResult] = []
    
    for alpha_pv in alpha_pv_values:
        for gamma in gamma_values:
            # 1. Capacities
            pv_capacity_mw = alpha_pv * poi_capacity_mw
            batt_power_mw = gamma * poi_capacity_mw
            
            if pv_capacity_mw <= 0:
                continue
            
            batt_energy_mwh = batt_power_mw * batt_duration_h
            
            # Check BESS sizing constraints for Innovation Tender eligibility
            min_bess_power = INNOVATION_TENDER_MIN_BESS_POWER_RATIO * pv_capacity_mw
            if batt_power_mw > 0 and batt_power_mw < min_bess_power:
                # Skip configurations that don't meet minimum BESS power requirement
                continue
            
            if batt_energy_mwh < INNOVATION_TENDER_MIN_BESS_DURATION_H * batt_power_mw:
                # Skip configurations that don't meet minimum BESS energy requirement
                continue
            
            # 2. CAPEX and OPEX
            capex_pv = pv_capacity_mw * 1000.0 * capex_pv_eur_per_kw
            capex_bess = batt_energy_mwh * 1000.0 * capex_bess_eur_per_kwh
            
            annual_opex_pv = pv_capacity_mw * 1000.0 * opex_pv_eur_per_kw_per_year
            annual_opex_bess = batt_power_mw * 1000.0 * opex_bess_eur_per_kw_per_year
            
            # 3. Scenario WACC
            if capex_bess > 0:
                wacc = combined_wacc_from_capex(
                    capex_pv=capex_pv,
                    capex_bess=capex_bess,
                    wacc_pv=wacc_pv,
                    wacc_bess=wacc_bess,
                )
            else:
                wacc = wacc_pv  # PV-only case
            
            # 4. Dispatch simulation using LP with Innovationsausschreibung
            annual_revenue_eur_merchant: Optional[float] = None
            if batt_power_mw > 0:
                # Calculate PV generation from pv_per_kw
                pv_generation_mw = df_lp['pv_per_kw'].values * (pv_capacity_mw * 1000.0) / 1000.0
                
                # Prepare data for LP simulation
                df_lp_input = pd.DataFrame({
                    'timestamp': df_lp['timestamp'],
                    'pv_generation_mw': pv_generation_mw,
                    'day_ahead_price_eur_mwh': df_lp['price_eur_per_mwh'].values,
                })
                
                # Calculate charge/discharge efficiencies from roundtrip efficiency
                eta_ch = np.sqrt(roundtrip_efficiency)
                eta_dis = np.sqrt(roundtrip_efficiency)
                
                # Run LP optimization
                params = {
                    'bess_cap_mwh': batt_energy_mwh,
                    'bess_pwr_mw': batt_power_mw,
                    'poi_mw': poi_capacity_mw,
                    'eta_ch': eta_ch,
                    'eta_dis': eta_dis,
                }
                
                try:
                    lp_result = simulate_pv_bess_lp_innovationsausschreibung(
                        df=df_lp_input,
                        params=params,
                        pv_capacity_mw=pv_capacity_mw,
                        solver=solver,
                        time_limit_seconds=time_limit_seconds,
                        verbose=False,
                    )
                    
                    # Extract annual metrics from LP results
                    annual_revenue_eur = lp_result['revenue_eur'].sum()
                    annual_export_mwh_total = lp_result['export_mw'].sum()
                    
                    # Calculate PV and BESS exports
                    # From LP: export[t] = pv[t] - charge[t] - curtailment[t] + discharge[t]
                    # So: export[t] - discharge[t] = pv[t] - charge[t] - curtailment[t]
                    # This represents the net PV contribution to export
                    # BESS export is simply the discharge
                    annual_export_mwh_pv = (lp_result['export_mw'] - lp_result['discharge_mw']).clip(lower=0).sum()
                    annual_export_mwh_bess = lp_result['discharge_mw'].sum()
                    
                    annual_curtailment_mwh = lp_result['curtailment_mw'].sum()
                    storage_throughput_mwh = lp_result['discharge_mw'].sum()

                    try:
                        merchant_result = simulate_pv_bess_lp(
                            df=df_lp_input,
                            params=params,
                            solver=solver,
                            time_limit_seconds=time_limit_seconds,
                            verbose=False,
                        )
                        annual_revenue_eur_merchant = float(merchant_result['revenue_eur'].sum())
                    except Exception:
                        annual_revenue_eur_merchant = float(
                            (lp_result['export_mw'] * lp_result['price_eur_mwh']).sum()
                        )
                    
                except Exception as e:
                    # If LP fails (e.g., infeasible), skip this configuration
                    print(f"Warning: LP optimization failed for alpha_pv={alpha_pv:.2f}, gamma={gamma:.2f}: {e}")
                    continue
            else:
                # PV-only case: use single tech simulation
                from simulation import simulate_single_tech_year
                sim_result = simulate_single_tech_year(
                    df=df,
                    tech='pv',
                    capacity_mw=pv_capacity_mw,
                    poi_capacity_mw=poi_capacity_mw,
                )
                annual_revenue_eur = sim_result['annual_revenue_eur']
                annual_export_mwh_total = sim_result['annual_export_mwh']
                annual_export_mwh_pv = sim_result['annual_export_mwh']
                annual_export_mwh_bess = 0.0
                annual_curtailment_mwh = sim_result['annual_curtailment_mwh']
                storage_throughput_mwh = 0.0
            
            # 5. Annual metrics
            poi_utilisation = annual_export_mwh_total / (poi_capacity_mw * 8760.0)
            
            # 6. Yearly cash flows
            premium_years = INNOVATION_TENDER_PREMIUM_YEARS if batt_power_mw > 0 else None
            merchant_revenue_year1 = annual_revenue_eur_merchant if batt_power_mw > 0 else None
            cash_flows = build_cash_flows_pv_bess(
                annual_revenue_year1=annual_revenue_eur,
                annual_opex_pv=annual_opex_pv,
                annual_opex_bess=annual_opex_bess,
                project_lifetime_years=project_lifetime_years,
                degradation_pv=degradation_pv,
                annual_revenue_year1_merchant=merchant_revenue_year1,
                premium_years=premium_years,
            )
            
            # 7. Battery replacement at year 15
            replacement_capex = bess_replacement_share * capex_bess
            if project_lifetime_years >= 15 and batt_power_mw > 0:
                cash_flows[15 - 1] -= replacement_capex
            
            # 8. NPV (Year-0 convention)
            capex_total = capex_pv + capex_bess
            full_cash_flows = np.insert(cash_flows, 0, -capex_total)
            npv_total = npv(full_cash_flows, wacc)
            
            # 9. Storage ratio (kWh per kW_p PV)
            e_per_kwp_storage_kwh = (
                batt_energy_mwh * 1000.0 / (pv_capacity_mw * 1000.0)
                if pv_capacity_mw > 0
                else 0.0
            )
            
            # 10. IRR calculation (Year-0 convention)
            try:
                irr_val = irr(full_cash_flows)
            except ValueError:
                irr_val = np.nan
            
            # Create result
            result = PvBessResult(
                alpha_pv=alpha_pv,
                gamma=gamma,
                pv_capacity_mw=pv_capacity_mw,
                batt_power_mw=batt_power_mw,
                batt_energy_mwh=batt_energy_mwh,
                wacc=wacc,
                npv_eur=npv_total,
                annual_revenue_eur=annual_revenue_eur,
                annual_export_mwh_total=annual_export_mwh_total,
                annual_export_mwh_pv=annual_export_mwh_pv,
                annual_export_mwh_bess=annual_export_mwh_bess,
                annual_curtailment_mwh=annual_curtailment_mwh,
                poi_utilisation=poi_utilisation,
                storage_throughput_mwh=storage_throughput_mwh,
                e_per_kwp_storage_kwh=e_per_kwp_storage_kwh,
                capex_pv_eur=capex_pv,
                capex_bess_eur=capex_bess,
                capex_total_eur=capex_total,
                annual_opex_pv_eur=annual_opex_pv,
                annual_opex_bess_eur=annual_opex_bess,
                annual_opex_total_eur=annual_opex_pv + annual_opex_bess,
                irr=irr_val,
                annual_revenue_eur_merchant=annual_revenue_eur_merchant,
            )
            results.append(result)
    
    return results


def find_best_pv_bess_by_npv(results: Sequence[PvBessResult]) -> PvBessResult:
    """
    Return the PV+BESS design with the highest NPV.
    
    Parameters
    ----------
    results : Sequence[PvBessResult]
        List of PV+BESS evaluation results.
        
    Returns
    -------
    PvBessResult
        Configuration with maximum NPV.
    """
    return max(results, key=lambda r: r.npv_eur)


def find_best_pv_bess_by_poi_utilisation(results: Sequence[PvBessResult]) -> PvBessResult:
    """
    Return the PV+BESS design with the highest POI utilisation.
    
    Parameters
    ----------
    results : Sequence[PvBessResult]
        List of PV+BESS evaluation results.
        
    Returns
    -------
    PvBessResult
        Configuration with maximum POI utilisation.
    """
    return max(results, key=lambda r: r.poi_utilisation)


def results_to_dataframe(results: Sequence[PvBessResult]) -> pd.DataFrame:
    """
    Convert a list of PvBessResult to a pandas DataFrame.
    
    Parameters
    ----------
    results : Sequence[PvBessResult]
        List of PV+BESS evaluation results.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with all result fields as columns.
    """
    return pd.DataFrame([r.__dict__ for r in results])


# =============================================================================
# Example Usage
# =============================================================================
# Example grid for greenfield PV+BESS with 2h battery:
#
# alpha_pv_values = [1.0 + 0.1 * i for i in range(0, 21)]   # 1.0 to 3.0
# gamma_values = [0.0 + 0.1 * i for i in range(0, 11)]      # 0.0 to 1.0
#
# results = evaluate_pv_bess_grid(df, alpha_pv_values, gamma_values, batt_duration_h=2.0)
# best = find_best_pv_bess_by_npv(results)
#
# df_res = results_to_dataframe(results)
# # Then plot NPV as heatmap over (alpha_pv, gamma)
