"""
Linear Programming (LP) optimization for PV + BESS dispatch.

This module provides an optimal battery dispatch using the PuLP library.

The LP formulation maximizes revenue while respecting battery and POI constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import pulp
from config import (
    INNOVATION_TENDER_STRIKE_PRICE_EUR_PER_MWH,
    INNOVATION_TENDER_MIN_BESS_POWER_RATIO,
    INNOVATION_TENDER_MIN_BESS_DURATION_H,
    BESS_MAX_ANNUAL_DISCHARGE_CYCLES,
)


def _get_lp_solver(solver: Optional[str], time_limit_seconds: int, verbose: bool):
    """
    Get the appropriate PuLP solver instance based on solver name.
    
    Parameters
    ----------
    solver : str, optional
        Solver name ('CBC', 'GUROBI', 'CPLEX', or None for default CBC).
    time_limit_seconds : int
        Maximum solver time in seconds.
    verbose : bool
        If True, print solver progress.
        
    Returns
    -------
    pulp.LpSolver
        Configured solver instance.
    """
    if solver is None or solver.upper() == 'CBC':
        return pulp.PULP_CBC_CMD(timeLimit=time_limit_seconds, msg=verbose)
    elif solver.upper() == 'GUROBI':
        return pulp.GUROBI(timeLimit=time_limit_seconds, msg=verbose)
    elif solver.upper() == 'CPLEX':
        return pulp.CPLEX(timeLimit=time_limit_seconds, msg=verbose)
    else:
        return pulp.getSolver(solver, timeLimit=time_limit_seconds, msg=verbose)


def simulate_pv_bess_lp(
    df: pd.DataFrame,
    params: Dict[str, float],
    solver: Optional[str] = None,
    time_limit_seconds: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Optimize PV+BESS dispatch using Linear Programming.

    Finds the optimal battery charge/discharge schedule to maximize revenue
    subject to battery and POI constraints. This is the theoretically optimal
    dispatch assuming perfect price foresight.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns:
        - 'pv_generation_mw': PV generation in MW (8760 rows for a year)
        - 'day_ahead_price_eur_mwh': Day-ahead price in EUR/MWh
    params : dict
        Dictionary containing:
        - 'bess_cap_mwh': Battery energy capacity (MWh)
        - 'bess_pwr_mw': Battery power rating for charge/discharge (MW)
        - 'poi_mw': Point of interconnection limit (MW)
        - 'eta_ch': Charging efficiency (0-1)
        - 'eta_dis': Discharging efficiency (0-1)
    solver : str, optional
        PuLP solver name (e.g., 'PULP_CBC_CMD', 'GUROBI', 'CPLEX').
        If None, uses the default CBC solver.
    time_limit_seconds : int, default 300
        Maximum solver time in seconds.
    verbose : bool, default False
        If True, print solver progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly dispatch results:
        - 'charge_mw': Battery charging power (MW)
        - 'discharge_mw': Battery discharging power (MW)
        - 'soc_mwh': State of charge at end of hour (MWh)
        - 'export_mw': Power exported to grid (MW)
        - 'pv_generation_mw': Original PV generation (MW)
        - 'price_eur_mwh': Price (EUR/MWh)
        - 'revenue_eur': Hourly revenue (EUR)
        - 'curtailment_mw': Curtailed PV (MW)

    Notes
    -----
    The LP formulation:

    Decision variables:
        - charge[t] >= 0: Charging power at hour t (MW)
        - discharge[t] >= 0: Discharging power at hour t (MW)
        - soc[t] >= 0: State of charge at end of hour t (MWh)
        - export[t] >= 0: Power exported to grid at hour t (MW)

    Objective:
        Maximize: sum_t(export[t] * price[t])

    Constraints:
        1. Energy balance: export[t] = pv[t] - charge[t] - curtailment[t] + discharge[t]
        2. POI limit: export[t] <= poi_mw
        3. SOC dynamics: soc[t] = soc[t-1] + charge[t]*eta_ch - discharge[t]/eta_dis
        4. SOC bounds: 0 <= soc[t] <= bess_cap_mwh
        5. Charge power limit: 0 <= charge[t] <= bess_pwr_mw
        6. Discharge power limit: 0 <= discharge[t] <= bess_pwr_mw
        7. Non-negative export: export[t] >= 0 (no grid charging)
        8. Charge from PV only: charge[t] <= pv[t] (green battery constraint)
        9. Curtailment upper bound: curtailment[t] <= pv[t] - charge[t]
           (ensures battery discharge must go to export and face market price / POI limit)
        10. Annual discharge throughput cap: sum_t discharge[t] <= max_annual_discharge_mwh

    The battery is assumed to start and end with SOC = 0 (cyclic constraint).
    """
    # Extract parameters
    bess_cap_mwh = params['bess_cap_mwh']
    bess_pwr_mw = params['bess_pwr_mw']
    poi_mw = params['poi_mw']
    eta_ch = params['eta_ch']
    eta_dis = params['eta_dis']
    # Extract time series data
    pv = df['pv_generation_mw'].values
    price = df['day_ahead_price_eur_mwh'].values
    n_hours = len(df)

    # Create the LP problem
    prob = pulp.LpProblem("PV_BESS_Dispatch", pulp.LpMaximize)

    # Decision variables
    # Charging power (MW) for each hour
    charge = pulp.LpVariable.dicts(
        "charge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    # Discharging power (MW) for each hour
    discharge = pulp.LpVariable.dicts(
        "discharge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    # State of charge at end of each hour (MWh)
    soc = pulp.LpVariable.dicts(
        "soc", range(n_hours), lowBound=0, upBound=bess_cap_mwh
    )
    # Export power (MW) for each hour
    export = pulp.LpVariable.dicts(
        "export", range(n_hours), lowBound=0, upBound=poi_mw
    )
    # Curtailment power (MW) for each hour - PV that cannot be exported or stored
    curtailment = pulp.LpVariable.dicts(
        "curtailment", range(n_hours), lowBound=0
    )

    # Objective: Maximize revenue
    # Revenue = sum(export[t] * price[t]) for all t
    prob += pulp.lpSum(
        export[t] * price[t] for t in range(n_hours)
    ), "Total_Net_Revenue"

    # Annual discharge throughput cap (scaled to modeled horizon)
    hours_per_year = 8760.0
    annual_discharge_cap_mwh = (
        BESS_MAX_ANNUAL_DISCHARGE_CYCLES * bess_cap_mwh * (n_hours / hours_per_year)
    )
    prob += (
        pulp.lpSum(discharge[t] for t in range(n_hours)) <= annual_discharge_cap_mwh,
        "Annual_Discharge_Throughput_Cap",
    )

    # Constraints
    for t in range(n_hours):
        # 1. Energy balance: PV = Charge + Export + Curtailment - Discharge
        #    Rearranged: Export = PV - Charge - Curtailment + Discharge
        prob += (
            export[t] == pv[t] - charge[t] - curtailment[t] + discharge[t],
            f"Energy_Balance_{t}"
        )

        # 2. POI limit is already enforced by the upper bound on export

        # 3. SOC dynamics
        if t == 0:
            # Initial SOC is 0 (start empty)
            prob += (
                soc[t] == charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )
        else:
            prob += (
                soc[t] == soc[t - 1] + charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )

        # 4. SOC bounds are already enforced by variable bounds

        # 5. & 6. Charge/discharge power limits are already enforced by variable bounds

        # 7. Non-negative export is already enforced by variable lower bound

        # 8. Green battery: can only charge from PV (charge <= pv)
        prob += (
            charge[t] <= pv[t],
            f"Charge_From_PV_Only_{t}"
        )

        # 9. Curtailment upper bound: can only curtail PV, not battery discharge
        # This ensures battery discharge must go to export and face market price / POI limit
        prob += (
            curtailment[t] <= pv[t] - charge[t],
            f"Curtailment_Upper_Bound_{t}"
        )

    # Cyclic constraint (end SOC = start SOC = 0)
    prob += soc[n_hours - 1] == 0, "Cyclic_SOC"

    # Select solver
    solver_instance = _get_lp_solver(solver, time_limit_seconds, verbose)

    # Solve the problem
    prob.solve(solver_instance)

    # Check solution status
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        raise RuntimeError(f"LP solver did not find optimal solution. Status: {status}")

    # Extract results
    charge_vals = np.array([pulp.value(charge[t]) for t in range(n_hours)])
    discharge_vals = np.array([pulp.value(discharge[t]) for t in range(n_hours)])
    soc_vals = np.array([pulp.value(soc[t]) for t in range(n_hours)])
    export_vals = np.array([pulp.value(export[t]) for t in range(n_hours)])
    curtailment_vals = np.array([pulp.value(curtailment[t]) for t in range(n_hours)])

    # Handle potential None values from solver (replace with 0)
    charge_vals = np.nan_to_num(charge_vals, nan=0.0)
    discharge_vals = np.nan_to_num(discharge_vals, nan=0.0)
    soc_vals = np.nan_to_num(soc_vals, nan=0.0)
    export_vals = np.nan_to_num(export_vals, nan=0.0)
    curtailment_vals = np.nan_to_num(curtailment_vals, nan=0.0)

    # Compute derived quantities
    revenue_vals = export_vals * price

    # Build result DataFrame
    result_df = pd.DataFrame({
        'charge_mw': charge_vals,
        'discharge_mw': discharge_vals,
        'soc_mwh': soc_vals,
        'export_mw': export_vals,
        'pv_generation_mw': pv,
        'price_eur_mwh': price,
        'revenue_eur': revenue_vals,
        'curtailment_mw': curtailment_vals,
    })

    # Preserve the original index if present
    if df.index is not None:
        result_df.index = df.index

    return result_df



def simulate_pv_bess_lp_innovationsausschreibung(
    df: pd.DataFrame,
    params: Dict[str, float],
    pv_capacity_mw: float,
    solver: Optional[str] = None,
    time_limit_seconds: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Optimize PV+BESS dispatch using Linear Programming with Innovationsausschreibung 2024.

    Implements the Floating Market Premium (FMP) mechanism from the EEG Innovation Tender 2024:
    - Market Premium = max(0, strike_price - monthly_pv_market_value)
    - Premium goes to 0 in hours where the (hourly) Day-Ahead price
      – interpreted here as the arithmetic mean of the quarter-hourly prices – is negative
    - Objective: maximize sum(P_export * (Price_DA + MP))

    BESS sizing constraints for eligibility:
    - P_BESS >= 0.25 × P_PV
    - E_BESS >= 2 × P_BESS

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns:
        - 'pv_generation_mw': PV generation in MW (8760 rows for a year)
        - 'day_ahead_price_eur_mwh': Day-ahead price in EUR/MWh
        - 'timestamp': Datetime index (for monthly grouping)
    params : dict
        Dictionary containing:
        - 'bess_cap_mwh': Battery energy capacity (MWh)
        - 'bess_pwr_mw': Battery power rating for charge/discharge (MW)
        - 'poi_mw': Point of interconnection limit (MW)
        - 'eta_ch': Charging efficiency (0-1)
        - 'eta_dis': Discharging efficiency (0-1)
    pv_capacity_mw : float
        Installed PV capacity (MW). Required for BESS sizing constraints.
    solver : str, optional
        PuLP solver name (e.g., 'PULP_CBC_CMD', 'GUROBI', 'CPLEX').
        If None, uses the default CBC solver.
    time_limit_seconds : int, default 300
        Maximum solver time in seconds.
    verbose : bool, default False
        If True, print solver progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly dispatch results:
        - 'charge_mw': Battery charging power (MW)
        - 'discharge_mw': Battery discharging power (MW)
        - 'soc_mwh': State of charge at end of hour (MWh)
        - 'export_mw': Power exported to grid (MW)
        - 'pv_generation_mw': Original PV generation (MW)
        - 'price_eur_mwh': Price (EUR/MWh)
        - 'market_premium_eur_mwh': Market premium applied (EUR/MWh)
        - 'revenue_eur': Hourly revenue including market premium (EUR)
        - 'curtailment_mw': Curtailed PV (MW)

    Notes
    -----
    The LP formulation:

    Decision variables:
        - charge[t] >= 0: Charging power at hour t (MW)
        - discharge[t] >= 0: Discharging power at hour t (MW)
        - soc[t] >= 0: State of charge at end of hour t (MWh)
        - export[t] >= 0: Power exported to grid at hour t (MW)

    Objective:
        Maximize: sum_t(export[t] * (price[t] + mp[t]))

    Constraints:
        1. Energy balance: export[t] = pv[t] - charge[t] - curtailment[t] + discharge[t]
        2. POI limit: export[t] <= poi_mw
        3. SOC dynamics: soc[t] = soc[t-1] + charge[t]*eta_ch - discharge[t]/eta_dis
        4. SOC bounds: 0 <= soc[t] <= bess_cap_mwh
        5. Charge power limit: 0 <= charge[t] <= bess_pwr_mw
        6. Discharge power limit: 0 <= discharge[t] <= bess_pwr_mw
        7. Green battery: charge[t] <= pv[t]
        8. Curtailment upper bound: curtailment[t] <= pv[t] - charge[t]
        9. Cyclic SOC: soc[last] = 0
        10. Annual discharge throughput cap: sum_t discharge[t] <= max_annual_discharge_mwh

    BESS sizing requirements (validated before solving):
        - bess_pwr_mw >= 0.25 * pv_capacity_mw
        - bess_cap_mwh >= 2.0 * bess_pwr_mw
    """
    # Extract parameters
    bess_cap_mwh = params['bess_cap_mwh']
    bess_pwr_mw = params['bess_pwr_mw']
    poi_mw = params['poi_mw']
    eta_ch = params['eta_ch']
    eta_dis = params['eta_dis']
    # Validate BESS sizing constraints for Innovation Tender eligibility
    min_bess_power = INNOVATION_TENDER_MIN_BESS_POWER_RATIO * pv_capacity_mw
    min_bess_energy = INNOVATION_TENDER_MIN_BESS_DURATION_H * bess_pwr_mw
    
    if bess_pwr_mw < min_bess_power:
        raise ValueError(
            f"BESS power {bess_pwr_mw} MW must be >= {min_bess_power} MW "
            f"(0.25 × PV capacity {pv_capacity_mw} MW) for Innovation Tender eligibility"
        )
    
    if bess_cap_mwh < min_bess_energy:
        raise ValueError(
            f"BESS energy {bess_cap_mwh} MWh must be >= {min_bess_energy} MWh "
            f"(2.0 × BESS power {bess_pwr_mw} MW) for Innovation Tender eligibility"
        )

    # Extract time series data
    pv = df['pv_generation_mw'].values
    price = df['day_ahead_price_eur_mwh'].values
    n_hours = len(df)

    # Ensure we have a timestamp column for monthly grouping
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        # Create a default datetime index if not available
        timestamps = pd.date_range(start='2023-01-01', periods=n_hours, freq='H')

    # Calculate monthly market premium
    # Step 1: Calculate monthly PV market value (weighted average price of PV generation hours)
    # The market value is calculated based on the PV generation profile (normalized),
    # weighted by the actual prices. This gives the average price that PV receives.
    df_temp = pd.DataFrame({
        'timestamp': timestamps,
        'pv_generation_mw': pv,
        'price_eur_mwh': price,
    })
    df_temp['month'] = df_temp['timestamp'].dt.month
    
    # Calculate monthly PV market value (weighted by PV generation)
    # This represents the average price weighted by when PV actually generates
    monthly_pv_market_value = {}
    for month in range(1, 13):
        month_data = df_temp[df_temp['month'] == month]
        if len(month_data) > 0:
            # Weighted average price (weighted by PV generation)
            # This gives the market value of PV generation for this month
            total_pv = month_data['pv_generation_mw'].sum()
            if total_pv > 0:
                monthly_pv_market_value[month] = (
                    (month_data['pv_generation_mw'] * month_data['price_eur_mwh']).sum() 
                    / total_pv
                )
            else:
                # If no PV generation in this month, use simple average price
                monthly_pv_market_value[month] = month_data['price_eur_mwh'].mean()
        else:
            monthly_pv_market_value[month] = 0.0
    
    # Step 2: Calculate monthly market premium
    # MP_month = max(0, strike_price - monthly_pv_market_value)
    monthly_mp = {}
    for month in range(1, 13):
        mp = max(0.0, INNOVATION_TENDER_STRIKE_PRICE_EUR_PER_MWH - monthly_pv_market_value[month])
        monthly_mp[month] = mp
    
    # Step 3: Calculate hourly market premium
    # MP_t = 0 if hourly (mean) Day-Ahead price is negative, else MP_month
    hourly_mp = np.zeros(n_hours)
    for month in range(1, 13):
        month_mask = df_temp['month'] == month
        if month_mask.any():
            mp_month = monthly_mp[month]
            hourly_mp[month_mask] = np.where(price[month_mask] < 0, 0.0, mp_month)

    # Create the LP problem
    prob = pulp.LpProblem("PV_BESS_Dispatch_Innovationsausschreibung", pulp.LpMaximize)

    # Decision variables
    charge = pulp.LpVariable.dicts(
        "charge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    discharge = pulp.LpVariable.dicts(
        "discharge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    soc = pulp.LpVariable.dicts(
        "soc", range(n_hours), lowBound=0, upBound=bess_cap_mwh
    )
    export = pulp.LpVariable.dicts(
        "export", range(n_hours), lowBound=0, upBound=poi_mw
    )
    curtailment = pulp.LpVariable.dicts(
        "curtailment", range(n_hours), lowBound=0
    )

    # Objective: Maximize revenue including market premium
    # Revenue = sum(export[t] * (price[t] + mp[t]))
    prob += pulp.lpSum(
        export[t] * (price[t] + hourly_mp[t])
        for t in range(n_hours)
    ), "Total_Net_Revenue_With_MP"

    # Annual discharge throughput cap (scaled to modeled horizon)
    hours_per_year = 8760.0
    annual_discharge_cap_mwh = (
        BESS_MAX_ANNUAL_DISCHARGE_CYCLES * bess_cap_mwh * (n_hours / hours_per_year)
    )
    prob += (
        pulp.lpSum(discharge[t] for t in range(n_hours)) <= annual_discharge_cap_mwh,
        "Annual_Discharge_Throughput_Cap",
    )

    # Constraints
    for t in range(n_hours):
        # 1. Energy balance: PV = Charge + Export + Curtailment - Discharge
        prob += (
            export[t] == pv[t] - charge[t] - curtailment[t] + discharge[t],
            f"Energy_Balance_{t}"
        )

        # 2. SOC dynamics
        if t == 0:
            prob += (
                soc[t] == charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )
        else:
            prob += (
                soc[t] == soc[t - 1] + charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )

        # 3. Green battery: can only charge from PV
        prob += (
            charge[t] <= pv[t],
            f"Charge_From_PV_Only_{t}"
        )

        # 4. Curtailment upper bound
        prob += (
            curtailment[t] <= pv[t] - charge[t],
            f"Curtailment_Upper_Bound_{t}"
        )

    # Cyclic SOC constraint (end SOC = start SOC = 0)
    prob += soc[n_hours - 1] == 0, "Cyclic_SOC"

    # Select solver
    solver_instance = _get_lp_solver(solver, time_limit_seconds, verbose)

    # Solve the problem
    prob.solve(solver_instance)

    # Check solution status
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        raise RuntimeError(f"LP solver did not find optimal solution. Status: {status}")

    # Extract results
    charge_vals = np.array([pulp.value(charge[t]) for t in range(n_hours)])
    discharge_vals = np.array([pulp.value(discharge[t]) for t in range(n_hours)])
    soc_vals = np.array([pulp.value(soc[t]) for t in range(n_hours)])
    export_vals = np.array([pulp.value(export[t]) for t in range(n_hours)])
    curtailment_vals = np.array([pulp.value(curtailment[t]) for t in range(n_hours)])

    # Handle potential None values from solver
    charge_vals = np.nan_to_num(charge_vals, nan=0.0)
    discharge_vals = np.nan_to_num(discharge_vals, nan=0.0)
    soc_vals = np.nan_to_num(soc_vals, nan=0.0)
    export_vals = np.nan_to_num(export_vals, nan=0.0)
    curtailment_vals = np.nan_to_num(curtailment_vals, nan=0.0)

    # Compute revenue including market premium
    revenue_vals = export_vals * (price + hourly_mp)

    # Build result DataFrame
    result_df = pd.DataFrame({
        'charge_mw': charge_vals,
        'discharge_mw': discharge_vals,
        'soc_mwh': soc_vals,
        'export_mw': export_vals,
        'pv_generation_mw': pv,
        'price_eur_mwh': price,
        'market_premium_eur_mwh': hourly_mp,
        'revenue_eur': revenue_vals,
        'curtailment_mw': curtailment_vals,
    })

    # Preserve the original index if present
    if df.index is not None:
        result_df.index = df.index

    return result_df


def simulate_pv_wind_bess_lp_innovationsausschreibung(
    df: pd.DataFrame,
    params: Dict[str, float],
    pv_capacity_mw: float,
    wind_capacity_mw: float,
    solver: Optional[str] = None,
    time_limit_seconds: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Optimize PV+Wind+BESS dispatch using Linear Programming with Innovationsausschreibung 2024.

    Implements the Floating Market Premium (FMP) mechanism from the EEG Innovation Tender 2024
    for hybrid PV+Wind systems:
    - Market Premium = max(0, strike_price - monthly_hybrid_market_value)
    - Premium goes to 0 in hours where the (hourly) Day-Ahead price
      – interpreted here as the arithmetic mean of the quarter-hourly prices – is negative
    - Objective: maximize sum(P_export * (Price_DA + MP))

    BESS sizing constraints for eligibility:
    - P_BESS >= 0.25 × (P_PV + P_Wind)  [or just P_PV if only PV eligible]
    - E_BESS >= 2 × P_BESS

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns:
        - 'pv_generation_mw': PV generation in MW (8760 rows for a year)
        - 'wind_generation_mw': Wind generation in MW
        - 'day_ahead_price_eur_mwh': Day-ahead price in EUR/MWh
        - 'timestamp': Datetime index (for monthly grouping)
    params : dict
        Dictionary containing:
        - 'bess_cap_mwh': Battery energy capacity (MWh)
        - 'bess_pwr_mw': Battery power rating for charge/discharge (MW)
        - 'poi_mw': Point of interconnection limit (MW)
        - 'eta_ch': Charging efficiency (0-1)
        - 'eta_dis': Discharging efficiency (0-1)
    pv_capacity_mw : float
        Installed PV capacity (MW). Required for BESS sizing constraints.
    wind_capacity_mw : float
        Installed Wind capacity (MW). Required for BESS sizing constraints.
    solver : str, optional
        PuLP solver name (e.g., 'PULP_CBC_CMD', 'GUROBI', 'CPLEX').
        If None, uses the default CBC solver.
    time_limit_seconds : int, default 300
        Maximum solver time in seconds.
    verbose : bool, default False
        If True, print solver progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly dispatch results:
        - 'charge_mw': Battery charging power (MW)
        - 'discharge_mw': Battery discharging power (MW)
        - 'soc_mwh': State of charge at end of hour (MWh)
        - 'export_mw': Total power exported to grid (MW)
        - 'export_pv_mw': PV power exported to grid (MW)
        - 'export_wind_mw': Wind power exported to grid (MW)
        - 'export_bess_mw': Battery power exported to grid (MW)
        - 'pv_generation_mw': Original PV generation (MW)
        - 'wind_generation_mw': Original Wind generation (MW)
        - 'price_eur_mwh': Price (EUR/MWh)
        - 'market_premium_eur_mwh': Market premium applied (EUR/MWh)
        - 'revenue_eur': Hourly revenue including market premium (EUR)
        - 'curtailment_mw': Total curtailed power (MW)
        - 'curtailment_pv_mw': Curtailed PV (MW)
        - 'curtailment_wind_mw': Curtailed Wind (MW)

    Notes
    -----
    The LP formulation is similar to simulate_pv_wind_bess_lp but with market premium
    added to the objective function. The market premium is calculated based on the
    monthly market value of the hybrid generation (PV + Wind).
    """
    # Extract parameters
    bess_cap_mwh = params['bess_cap_mwh']
    bess_pwr_mw = params['bess_pwr_mw']
    poi_mw = params['poi_mw']
    eta_ch = params['eta_ch']
    eta_dis = params['eta_dis']
    # Validate BESS sizing constraints for Innovation Tender eligibility
    # Use PV capacity when present; fallback to wind capacity for wind-only cases.
    base_capacity_mw = pv_capacity_mw if pv_capacity_mw > 0 else wind_capacity_mw
    capacity_label = "PV" if pv_capacity_mw > 0 else "Wind"
    min_bess_power = INNOVATION_TENDER_MIN_BESS_POWER_RATIO * base_capacity_mw
    min_bess_energy = INNOVATION_TENDER_MIN_BESS_DURATION_H * bess_pwr_mw
    
    if bess_pwr_mw > 0 and bess_pwr_mw < min_bess_power:
        raise ValueError(
            f"BESS power {bess_pwr_mw} MW must be >= {min_bess_power} MW "
            f"(0.25 × {capacity_label} capacity {base_capacity_mw} MW) for Innovation Tender eligibility"
        )
    
    if bess_cap_mwh < min_bess_energy:
        raise ValueError(
            f"BESS energy {bess_cap_mwh} MWh must be >= {min_bess_energy} MWh "
            f"(2.0 × BESS power {bess_pwr_mw} MW) for Innovation Tender eligibility"
        )

    # Extract time series data
    pv = df['pv_generation_mw'].values
    wind = df['wind_generation_mw'].values
    total_gen = pv + wind
    price = df['day_ahead_price_eur_mwh'].values
    n_hours = len(df)

    # Ensure we have a timestamp column for monthly grouping
    if 'timestamp' in df.columns:
        timestamps = pd.to_datetime(df['timestamp'])
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        timestamps = df.index
    else:
        # Create a default datetime index if not available
        timestamps = pd.date_range(start='2023-01-01', periods=n_hours, freq='H')

    # Calculate monthly market premium
    # Step 1: Calculate monthly hybrid market value (weighted average price of generation hours)
    df_temp = pd.DataFrame({
        'timestamp': timestamps,
        'pv_generation_mw': pv,
        'wind_generation_mw': wind,
        'total_generation_mw': total_gen,
        'price_eur_mwh': price,
    })
    df_temp['month'] = df_temp['timestamp'].dt.month
    
    # Calculate monthly hybrid market value (weighted by total generation)
    monthly_hybrid_market_value = {}
    for month in range(1, 13):
        month_data = df_temp[df_temp['month'] == month]
        if len(month_data) > 0:
            # Weighted average price (weighted by total generation)
            total_gen_month = month_data['total_generation_mw'].sum()
            if total_gen_month > 0:
                monthly_hybrid_market_value[month] = (
                    (month_data['total_generation_mw'] * month_data['price_eur_mwh']).sum() 
                    / total_gen_month
                )
            else:
                # If no generation in this month, use simple average price
                monthly_hybrid_market_value[month] = month_data['price_eur_mwh'].mean()
        else:
            monthly_hybrid_market_value[month] = 0.0
    
    # Step 2: Calculate monthly market premium
    # MP_month = max(0, strike_price - monthly_hybrid_market_value)
    monthly_mp = {}
    for month in range(1, 13):
        mp = max(0.0, INNOVATION_TENDER_STRIKE_PRICE_EUR_PER_MWH - monthly_hybrid_market_value[month])
        monthly_mp[month] = mp
    
    # Step 3: Calculate hourly market premium
    # MP_t = 0 if hourly (mean) Day-Ahead price is negative, else MP_month
    hourly_mp = np.zeros(n_hours)
    for month in range(1, 13):
        month_mask = df_temp['month'] == month
        if month_mask.any():
            mp_month = monthly_mp[month]
            hourly_mp[month_mask] = np.where(price[month_mask] < 0, 0.0, mp_month)

    # Create the LP problem
    prob = pulp.LpProblem("PV_Wind_BESS_Dispatch_Innovationsausschreibung", pulp.LpMaximize)

    # Decision variables
    charge = pulp.LpVariable.dicts(
        "charge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    discharge = pulp.LpVariable.dicts(
        "discharge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    soc = pulp.LpVariable.dicts(
        "soc", range(n_hours), lowBound=0, upBound=bess_cap_mwh
    )
    export = pulp.LpVariable.dicts(
        "export", range(n_hours), lowBound=0, upBound=poi_mw
    )
    curtailment = pulp.LpVariable.dicts(
        "curtailment", range(n_hours), lowBound=0
    )

    # Objective: Maximize revenue including market premium
    # Revenue = sum(export[t] * (price[t] + mp[t]))
    prob += pulp.lpSum(
        export[t] * (price[t] + hourly_mp[t])
        for t in range(n_hours)
    ), "Total_Net_Revenue_With_MP"

    # Annual discharge throughput cap (scaled to modeled horizon)
    hours_per_year = 8760.0
    annual_discharge_cap_mwh = (
        BESS_MAX_ANNUAL_DISCHARGE_CYCLES * bess_cap_mwh * (n_hours / hours_per_year)
    )
    prob += (
        pulp.lpSum(discharge[t] for t in range(n_hours)) <= annual_discharge_cap_mwh,
        "Annual_Discharge_Throughput_Cap",
    )

    # Constraints
    for t in range(n_hours):
        # 1. Energy balance: PV + Wind = Charge + Export + Curtailment - Discharge
        prob += (
            export[t] == total_gen[t] - charge[t] - curtailment[t] + discharge[t],
            f"Energy_Balance_{t}"
        )

        # 2. SOC dynamics
        if t == 0:
            prob += (
                soc[t] == charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )
        else:
            prob += (
                soc[t] == soc[t - 1] + charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )

        # 3. Green battery: can only charge from renewables (PV + Wind)
        prob += (
            charge[t] <= total_gen[t],
            f"Charge_From_Renewables_Only_{t}"
        )

        # 4. Curtailment upper bound
        prob += (
            curtailment[t] <= total_gen[t] - charge[t],
            f"Curtailment_Upper_Bound_{t}"
        )

    # Cyclic SOC constraint (end SOC = start SOC = 0)
    prob += soc[n_hours - 1] == 0, "Cyclic_SOC"

    # Select solver
    solver_instance = _get_lp_solver(solver, time_limit_seconds, verbose)

    # Solve the problem
    prob.solve(solver_instance)

    # Check solution status
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        raise RuntimeError(f"LP solver did not find optimal solution. Status: {status}")

    # Extract results
    charge_vals = np.array([pulp.value(charge[t]) for t in range(n_hours)])
    discharge_vals = np.array([pulp.value(discharge[t]) for t in range(n_hours)])
    soc_vals = np.array([pulp.value(soc[t]) for t in range(n_hours)])
    export_vals = np.array([pulp.value(export[t]) for t in range(n_hours)])
    curtailment_vals = np.array([pulp.value(curtailment[t]) for t in range(n_hours)])

    # Handle potential None values from solver
    charge_vals = np.nan_to_num(charge_vals, nan=0.0)
    discharge_vals = np.nan_to_num(discharge_vals, nan=0.0)
    soc_vals = np.nan_to_num(soc_vals, nan=0.0)
    export_vals = np.nan_to_num(export_vals, nan=0.0)
    curtailment_vals = np.nan_to_num(curtailment_vals, nan=0.0)

    # Allocate curtailment proportionally between PV and Wind
    total_gen_safe = np.where(total_gen > 0, total_gen, 1.0)
    curtailment_pv_vals = curtailment_vals * (pv / total_gen_safe)
    curtailment_wind_vals = curtailment_vals * (wind / total_gen_safe)

    # Allocate charging proportionally between PV and Wind
    # charge_pv[t] = charge[t] * (pv[t] / (pv[t] + wind[t]))
    charge_pv_vals = charge_vals * (pv / total_gen_safe)
    charge_wind_vals = charge_vals * (wind / total_gen_safe)

    # Allocate export proportionally between PV, Wind, and Battery
    # Export from battery = discharge
    # Remaining export = PV + Wind contribution (proportional to generation after curtailment and charging)
    pv_after_curtail_and_charge = pv - curtailment_pv_vals - charge_pv_vals
    wind_after_curtail_and_charge = wind - curtailment_wind_vals - charge_wind_vals
    renewables_after_curtail_and_charge = pv_after_curtail_and_charge + wind_after_curtail_and_charge
    renewables_after_curtail_and_charge_safe = np.where(
        renewables_after_curtail_and_charge > 0, renewables_after_curtail_and_charge, 1.0
    )

    # Export = renewables_export + battery_export
    # renewables_export = export - discharge (what's exported from PV+Wind)
    renewables_export = np.maximum(export_vals - discharge_vals, 0.0)
    export_pv_vals = renewables_export * (pv_after_curtail_and_charge / renewables_after_curtail_and_charge_safe)
    export_wind_vals = renewables_export * (wind_after_curtail_and_charge / renewables_after_curtail_and_charge_safe)
    export_bess_vals = discharge_vals

    # Compute revenue including market premium
    # Revenue breakdown by technology
    effective_price = price + hourly_mp
    revenue_pv_vals = export_pv_vals * effective_price
    revenue_wind_vals = export_wind_vals * effective_price
    revenue_bess_vals = export_bess_vals * effective_price
    
    # Total revenue check
    revenue_vals = revenue_pv_vals + revenue_wind_vals + revenue_bess_vals

    # Build result DataFrame
    result_df = pd.DataFrame({
        'charge_mw': charge_vals,
        'charge_pv_mw': charge_pv_vals,
        'charge_wind_mw': charge_wind_vals,
        'discharge_mw': discharge_vals,
        'soc_mwh': soc_vals,
        'export_mw': export_vals,
        'export_pv_mw': export_pv_vals,
        'export_wind_mw': export_wind_vals,
        'export_bess_mw': export_bess_vals,
        'pv_generation_mw': pv,
        'wind_generation_mw': wind,
        'price_eur_mwh': price,
        'market_premium_eur_mwh': hourly_mp,
        'revenue_eur': revenue_vals,
        'revenue_pv_eur': revenue_pv_vals,
        'revenue_wind_eur': revenue_wind_vals,
        'revenue_bess_eur': revenue_bess_vals,
        'curtailment_mw': curtailment_vals,
        'curtailment_pv_mw': curtailment_pv_vals,
        'curtailment_wind_mw': curtailment_wind_vals,
    })

    # Preserve the original index if present
    if df.index is not None:
        result_df.index = df.index

    return result_df


def simulate_pv_wind_bess_lp(
    df: pd.DataFrame,
    params: Dict[str, float],
    solver: Optional[str] = None,
    time_limit_seconds: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Optimize PV+Wind+BESS dispatch using Linear Programming.

    Finds the optimal battery charge/discharge schedule to maximize revenue
    for a co-located PV, Wind, and Battery system behind a shared POI.
    The battery can charge from both PV and Wind generation (green battery).

    Parameters
    ----------
    df : pd.DataFrame
        Hourly data with columns:
        - 'pv_generation_mw': PV generation in MW (8760 rows for a year)
        - 'wind_generation_mw': Wind generation in MW
        - 'day_ahead_price_eur_mwh': Day-ahead price in EUR/MWh
    params : dict
        Dictionary containing:
        - 'bess_cap_mwh': Battery energy capacity (MWh)
        - 'bess_pwr_mw': Battery power rating for charge/discharge (MW)
        - 'poi_mw': Point of interconnection limit (MW)
        - 'eta_ch': Charging efficiency (0-1)
        - 'eta_dis': Discharging efficiency (0-1)
    solver : str, optional
        PuLP solver name. If None, uses the default CBC solver.
    time_limit_seconds : int, default 300
        Maximum solver time in seconds.
    verbose : bool, default False
        If True, print solver progress.

    Returns
    -------
    pd.DataFrame
        DataFrame with hourly dispatch results:
        - 'charge_mw': Battery charging power (MW)
        - 'discharge_mw': Battery discharging power (MW)
        - 'soc_mwh': State of charge at end of hour (MWh)
        - 'export_mw': Total power exported to grid (MW)
        - 'export_pv_mw': PV power exported to grid (MW)
        - 'export_wind_mw': Wind power exported to grid (MW)
        - 'pv_generation_mw': Original PV generation (MW)
        - 'wind_generation_mw': Original Wind generation (MW)
        - 'price_eur_mwh': Price (EUR/MWh)
        - 'revenue_eur': Hourly revenue (EUR)
        - 'curtailment_mw': Total curtailed power (MW)
        - 'curtailment_pv_mw': Curtailed PV (MW)
        - 'curtailment_wind_mw': Curtailed Wind (MW)

    Notes
    -----
    The LP formulation:

    Decision variables:
        - charge[t] >= 0: Charging power at hour t (MW)
        - discharge[t] >= 0: Discharging power at hour t (MW)
        - soc[t] >= 0: State of charge at end of hour t (MWh)
        - export[t] >= 0: Power exported to grid at hour t (MW)
        - curtailment[t] >= 0: Power curtailed at hour t (MW)

    Objective:
        Maximize: sum_t(export[t] * price[t])

    Constraints:
        1. Energy balance: export[t] = pv[t] + wind[t] - charge[t] - curtailment[t] + discharge[t]
        2. POI limit: export[t] <= poi_mw
        3. SOC dynamics: soc[t] = soc[t-1] + charge[t]*eta_ch - discharge[t]/eta_dis
        4. SOC bounds: 0 <= soc[t] <= bess_cap_mwh
        5. Charge power limit: 0 <= charge[t] <= bess_pwr_mw
        6. Discharge power limit: 0 <= discharge[t] <= bess_pwr_mw
        7. Non-negative export: export[t] >= 0 (no grid charging)
        8. Charge from renewables only: charge[t] <= pv[t] + wind[t]
        9. Curtailment upper bound: curtailment[t] <= pv[t] + wind[t] - charge[t]
           (ensures battery discharge must go to export and face market price / POI limit)
        10. Annual discharge throughput cap: sum_t discharge[t] <= max_annual_discharge_mwh

    Curtailment is allocated proportionally between PV and Wind based on generation.
    """
    # Extract parameters
    bess_cap_mwh = params['bess_cap_mwh']
    bess_pwr_mw = params['bess_pwr_mw']
    poi_mw = params['poi_mw']
    eta_ch = params['eta_ch']
    eta_dis = params['eta_dis']
    # Extract time series data
    pv = df['pv_generation_mw'].values
    wind = df['wind_generation_mw'].values
    total_gen = pv + wind
    price = df['day_ahead_price_eur_mwh'].values
    n_hours = len(df)

    # Create the LP problem
    prob = pulp.LpProblem("PV_Wind_BESS_Dispatch", pulp.LpMaximize)

    # Decision variables
    charge = pulp.LpVariable.dicts(
        "charge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    discharge = pulp.LpVariable.dicts(
        "discharge", range(n_hours), lowBound=0, upBound=bess_pwr_mw
    )
    soc = pulp.LpVariable.dicts(
        "soc", range(n_hours), lowBound=0, upBound=bess_cap_mwh
    )
    export = pulp.LpVariable.dicts(
        "export", range(n_hours), lowBound=0, upBound=poi_mw
    )
    curtailment = pulp.LpVariable.dicts(
        "curtailment", range(n_hours), lowBound=0
    )

    # Objective: Maximize revenue
    prob += pulp.lpSum(
        export[t] * price[t] for t in range(n_hours)
    ), "Total_Net_Revenue"

    # Annual discharge throughput cap (scaled to modeled horizon)
    hours_per_year = 8760.0
    annual_discharge_cap_mwh = (
        BESS_MAX_ANNUAL_DISCHARGE_CYCLES * bess_cap_mwh * (n_hours / hours_per_year)
    )
    prob += (
        pulp.lpSum(discharge[t] for t in range(n_hours)) <= annual_discharge_cap_mwh,
        "Annual_Discharge_Throughput_Cap",
    )

    # Constraints
    for t in range(n_hours):
        # 1. Energy balance: PV + Wind = Charge + Export + Curtailment - Discharge
        prob += (
            export[t] == total_gen[t] - charge[t] - curtailment[t] + discharge[t],
            f"Energy_Balance_{t}"
        )

        # 3. SOC dynamics
        if t == 0:
            prob += (
                soc[t] == charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )
        else:
            prob += (
                soc[t] == soc[t - 1] + charge[t] * eta_ch - discharge[t] / eta_dis,
                f"SOC_Dynamics_{t}"
            )

        # 8. Green battery: can only charge from renewables (PV + Wind)
        prob += (
            charge[t] <= total_gen[t],
            f"Charge_From_Renewables_Only_{t}"
        )

        # 9. Curtailment upper bound: can only curtail renewables, not battery discharge
        # This ensures battery discharge must go to export and face market price / POI limit
        prob += (
            curtailment[t] <= total_gen[t] - charge[t],
            f"Curtailment_Upper_Bound_{t}"
        )

    # Cyclic SOC constraint (end SOC = start SOC = 0)
    prob += soc[n_hours - 1] == 0, "Cyclic_SOC"

    # Select solver
    solver_instance = _get_lp_solver(solver, time_limit_seconds, verbose)

    # Solve the problem
    prob.solve(solver_instance)

    # Check solution status
    status = pulp.LpStatus[prob.status]
    if status != 'Optimal':
        raise RuntimeError(f"LP solver did not find optimal solution. Status: {status}")

    # Extract results
    charge_vals = np.array([pulp.value(charge[t]) for t in range(n_hours)])
    discharge_vals = np.array([pulp.value(discharge[t]) for t in range(n_hours)])
    soc_vals = np.array([pulp.value(soc[t]) for t in range(n_hours)])
    export_vals = np.array([pulp.value(export[t]) for t in range(n_hours)])
    curtailment_vals = np.array([pulp.value(curtailment[t]) for t in range(n_hours)])

    # Handle potential None values from solver (replace with 0)
    charge_vals = np.nan_to_num(charge_vals, nan=0.0)
    discharge_vals = np.nan_to_num(discharge_vals, nan=0.0)
    soc_vals = np.nan_to_num(soc_vals, nan=0.0)
    export_vals = np.nan_to_num(export_vals, nan=0.0)
    curtailment_vals = np.nan_to_num(curtailment_vals, nan=0.0)

    # Allocate curtailment proportionally between PV and Wind
    # curtailment_pv[t] = curtailment[t] * (pv[t] / (pv[t] + wind[t]))
    total_gen_safe = np.where(total_gen > 0, total_gen, 1.0)
    curtailment_pv_vals = curtailment_vals * (pv / total_gen_safe)
    curtailment_wind_vals = curtailment_vals * (wind / total_gen_safe)

    # Allocate charging proportionally between PV and Wind
    # charge_pv[t] = charge[t] * (pv[t] / (pv[t] + wind[t]))
    charge_pv_vals = charge_vals * (pv / total_gen_safe)
    charge_wind_vals = charge_vals * (wind / total_gen_safe)

    # Allocate export proportionally between PV, Wind, and Battery
    # Export from battery = discharge
    # Remaining export = PV + Wind contribution (proportional to generation after curtailment and charging)
    pv_after_curtail_and_charge = pv - curtailment_pv_vals - charge_pv_vals
    wind_after_curtail_and_charge = wind - curtailment_wind_vals - charge_wind_vals
    renewables_after_curtail_and_charge = pv_after_curtail_and_charge + wind_after_curtail_and_charge
    renewables_after_curtail_and_charge_safe = np.where(
        renewables_after_curtail_and_charge > 0, renewables_after_curtail_and_charge, 1.0
    )

    # Export = renewables_export + battery_export
    # renewables_export = export - discharge (what's exported from PV+Wind)
    renewables_export = np.maximum(export_vals - discharge_vals, 0.0)
    export_pv_vals = renewables_export * (pv_after_curtail_and_charge / renewables_after_curtail_and_charge_safe)
    export_wind_vals = renewables_export * (wind_after_curtail_and_charge / renewables_after_curtail_and_charge_safe)
    export_bess_vals = discharge_vals

    # Compute revenue breakdown
    revenue_pv_vals = export_pv_vals * price
    revenue_wind_vals = export_wind_vals * price
    revenue_bess_vals = export_bess_vals * price

    # Total revenue check
    revenue_vals = revenue_pv_vals + revenue_wind_vals + revenue_bess_vals

    # Build result DataFrame
    result_df = pd.DataFrame({
        'charge_mw': charge_vals,
        'charge_pv_mw': charge_pv_vals,
        'charge_wind_mw': charge_wind_vals,
        'discharge_mw': discharge_vals,
        'soc_mwh': soc_vals,
        'export_mw': export_vals,
        'export_pv_mw': export_pv_vals,
        'export_wind_mw': export_wind_vals,
        'export_bess_mw': export_bess_vals,
        'pv_generation_mw': pv,
        'wind_generation_mw': wind,
        'price_eur_mwh': price,
        'revenue_eur': revenue_vals,
        'revenue_pv_eur': revenue_pv_vals,
        'revenue_wind_eur': revenue_wind_vals,
        'revenue_bess_eur': revenue_bess_vals,
        'curtailment_mw': curtailment_vals,
        'curtailment_pv_mw': curtailment_pv_vals,
        'curtailment_wind_mw': curtailment_wind_vals,
    })

    # Preserve the original index if present
    if df.index is not None:
        result_df.index = df.index

    return result_df


# =============================================================================
# Example Usage
# =============================================================================
if __name__ == "__main__":
    # Example usage with synthetic data
    import numpy as np

    # Create synthetic hourly data for one year
    np.random.seed(42)
    hours = 8760

    # Synthetic PV profile (bell curve during day, zero at night)
    hour_of_day = np.tile(np.arange(24), 365)
    pv_profile = np.maximum(0, np.sin((hour_of_day - 6) * np.pi / 12))
    pv_profile = pv_profile * (0.8 + 0.4 * np.random.rand(hours))  # Add noise

    # Synthetic price profile (higher during day, lower at night)
    base_price = 50 + 20 * np.sin((hour_of_day - 18) * np.pi / 12)
    price = base_price + 30 * np.random.randn(hours)

    df = pd.DataFrame({
        'pv_generation_mw': pv_profile * 15,  # 15 MW peak PV
        'day_ahead_price_eur_mwh': price,
    })

    params = {
        'bess_cap_mwh': 20.0,   # 20 MWh battery
        'bess_pwr_mw': 10.0,   # 10 MW power
        'poi_mw': 10.0,        # 10 MW POI limit
        'eta_ch': 0.95,        # 95% charge efficiency
        'eta_dis': 0.95,       # 95% discharge efficiency
    }

    print("Running LP optimization...")
    result = simulate_pv_bess_lp(df, params, verbose=True)

    print("\n=== LP Optimization Results ===")
    print(f"Total Revenue: €{result['revenue_eur'].sum():,.0f}")
    print(f"Total Export: {result['export_mw'].sum():,.0f} MWh")
    print(f"Total Discharge: {result['discharge_mw'].sum():,.0f} MWh")
    print(f"Total Curtailment: {result['curtailment_mw'].sum():,.0f} MWh")
    print(f"Battery Cycles: {result['discharge_mw'].sum() / params['bess_cap_mwh']:.1f}")
