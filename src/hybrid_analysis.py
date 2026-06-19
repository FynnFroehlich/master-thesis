"""
Hybrid sizing analysis for co-located PV and wind plants at a fixed POI.

This module provides functions to evaluate NPV over a grid of total overplanting
factors and wind shares, finding the optimal hybrid configuration.
"""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

from config import (
    POI_CAPACITY_MW,
    CAPEX_PV_EUR_PER_KW,
    CAPEX_WIND_EUR_PER_KW,
    OPEX_PV_EUR_PER_KW_PER_YEAR,
    OPEX_WIND_EUR_PER_KW_PER_YEAR,
    WACC_PV_REAL,
    WACC_WIND_REAL,
    DEGRADATION_PV_PER_YEAR,
    LIFETIME_PV_YEARS,
    LIFETIME_WIND_YEARS,
)
from economics import irr, npv, build_cash_flows_hybrid_pv_wind
from simulation import simulate_hybrid_year


@dataclass
class HybridResult:
    """Result container for a single hybrid configuration evaluation."""
    
    alpha_tot: float              # Total overplanting factor
    s_wind: float                 # Wind share of installed capacity
    pv_capacity_mw: float         # Installed PV capacity in MW
    wind_capacity_mw: float       # Installed wind capacity in MW
    wacc: float                   # CAPEX-weighted average WACC
    npv_eur: float                # Net Present Value in EUR
    irr: float                    # Internal Rate of Return
    annual_revenue_eur_total: float
    annual_revenue_eur_pv: float
    annual_revenue_eur_wind: float
    annual_export_mwh_total: float
    annual_export_mwh_pv: float
    annual_export_mwh_wind: float
    annual_curtailment_mwh_total: float
    annual_curtailment_mwh_pv: float
    annual_curtailment_mwh_wind: float
    # System-oriented metrics
    export_per_mw_poi_mwh: float   # Annual export per MW of POI capacity
    poi_utilisation_factor: float  # POI utilisation (0-1), export / max possible export
    # Financial parameters (added for consistency)
    capex_pv_eur: float           # PV CAPEX in EUR
    capex_wind_eur: float         # Wind CAPEX in EUR
    capex_total_eur: float        # Total CAPEX in EUR
    annual_opex_pv_eur: float     # Annual PV OPEX in EUR
    annual_opex_wind_eur: float   # Annual Wind OPEX in EUR
    annual_opex_total_eur: float  # Total annual OPEX in EUR


def evaluate_hybrid_grid(
    df: pd.DataFrame,
    alpha_tot_values: Sequence[float],
    s_wind_values: Sequence[float],
    poi_capacity_mw: float = POI_CAPACITY_MW,
    lifetime_pv_years: int = LIFETIME_PV_YEARS,
    lifetime_wind_years: int = LIFETIME_WIND_YEARS,
) -> List[HybridResult]:
    """
    Evaluate NPV for a grid of hybrid layouts defined by total overplanting and wind share.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'.
    alpha_tot_values : Sequence[float]
        List of total overplanting factors to evaluate.
        alpha_tot = (P_pv + P_wind) / P_POI
    s_wind_values : Sequence[float]
        List of wind share values to evaluate.
        s_wind = P_wind / (P_pv + P_wind)
    poi_capacity_mw : float, optional
        POI capacity in MW (default: 10.0).
    lifetime_pv_years : int, optional
        PV project lifetime in years (default: 30).
    lifetime_wind_years : int, optional
        Wind project lifetime in years (default: 25).

    Returns
    -------
    List[HybridResult]
        List of results for each (alpha_tot, s_wind) combination.
    """
    results = []

    for alpha_tot in alpha_tot_values:
        for s_wind in s_wind_values:
            # Calculate capacities
            total_capacity_mw = alpha_tot * poi_capacity_mw
            wind_capacity_mw = total_capacity_mw * s_wind
            pv_capacity_mw = total_capacity_mw * (1.0 - s_wind)

            # Calculate CAPEX
            capex_pv = pv_capacity_mw * 1000.0 * CAPEX_PV_EUR_PER_KW
            capex_wind = wind_capacity_mw * 1000.0 * CAPEX_WIND_EUR_PER_KW
            capex_total = capex_pv + capex_wind

            # Skip if no capacity (shouldn't happen but be safe)
            if capex_total <= 0:
                continue

            # CAPEX-weighted WACC
            w_pv = capex_pv / capex_total
            w_wind = capex_wind / capex_total
            wacc = w_pv * WACC_PV_REAL + w_wind * WACC_WIND_REAL

            # Annual OPEX
            annual_opex_pv = pv_capacity_mw * 1000.0 * OPEX_PV_EUR_PER_KW_PER_YEAR
            annual_opex_wind = wind_capacity_mw * 1000.0 * OPEX_WIND_EUR_PER_KW_PER_YEAR

            # Simulate one year
            sim_result = simulate_hybrid_year(
                df=df,
                pv_capacity_mw=pv_capacity_mw,
                wind_capacity_mw=wind_capacity_mw,
                poi_capacity_mw=poi_capacity_mw,
            )

            # Build cash flows
            cash_flows = build_cash_flows_hybrid_pv_wind(
                annual_revenue_pv_year1=sim_result["annual_revenue_eur_pv"],
                annual_revenue_wind_year1=sim_result["annual_revenue_eur_wind"],
                annual_opex_pv=annual_opex_pv,
                annual_opex_wind=annual_opex_wind,
                lifetime_pv_years=lifetime_pv_years,
                lifetime_wind_years=lifetime_wind_years,
                wacc=wacc,
                pv_degradation_rate=DEGRADATION_PV_PER_YEAR,
            )

            # Calculate NPV
            # Use Year-0 convention: prepend -capex to cash flows
            full_cash_flows = np.insert(cash_flows, 0, -capex_total)
            npv_total = npv(full_cash_flows, wacc)
            try:
                irr_val = irr(full_cash_flows)
            except ValueError:
                irr_val = np.nan

            # System-oriented metrics
            annual_export_total = sim_result["annual_export_mwh_total"]
            export_per_mw_poi = annual_export_total / poi_capacity_mw
            poi_utilisation = annual_export_total / (poi_capacity_mw * 8760.0)

            # Store result
            results.append(
                HybridResult(
                    alpha_tot=alpha_tot,
                    s_wind=s_wind,
                    pv_capacity_mw=pv_capacity_mw,
                    wind_capacity_mw=wind_capacity_mw,
                    wacc=wacc,
                    npv_eur=npv_total,
                    irr=irr_val,
                    annual_revenue_eur_total=sim_result["annual_revenue_eur_total"],
                    annual_revenue_eur_pv=sim_result["annual_revenue_eur_pv"],
                    annual_revenue_eur_wind=sim_result["annual_revenue_eur_wind"],
                    annual_export_mwh_total=annual_export_total,
                    annual_export_mwh_pv=sim_result["annual_export_mwh_pv"],
                    annual_export_mwh_wind=sim_result["annual_export_mwh_wind"],
                    annual_curtailment_mwh_total=sim_result["annual_curtailment_mwh_total"],
                    annual_curtailment_mwh_pv=sim_result["annual_curtailment_mwh_pv"],
                    annual_curtailment_mwh_wind=sim_result["annual_curtailment_mwh_wind"],
                    export_per_mw_poi_mwh=export_per_mw_poi,
                    poi_utilisation_factor=poi_utilisation,
                    capex_pv_eur=capex_pv,
                    capex_wind_eur=capex_wind,
                    capex_total_eur=capex_total,
                    annual_opex_pv_eur=annual_opex_pv,
                    annual_opex_wind_eur=annual_opex_wind,
                    annual_opex_total_eur=annual_opex_pv + annual_opex_wind,
                )
            )

    return results


def find_best_hybrid_by_npv(results: List[HybridResult]) -> HybridResult:
    """
    Return the hybrid configuration with the highest NPV.

    Parameters
    ----------
    results : List[HybridResult]
        List of hybrid results from evaluate_hybrid_grid.

    Returns
    -------
    HybridResult
        The result with the maximum NPV.

    Raises
    ------
    ValueError
        If the results list is empty.
    """
    if not results:
        raise ValueError("Results list cannot be empty")
    return max(results, key=lambda r: r.npv_eur)


def results_to_dataframe(results: List[HybridResult]) -> pd.DataFrame:
    """
    Convert a list of HybridResult objects to a DataFrame.

    Parameters
    ----------
    results : List[HybridResult]
        List of hybrid results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching HybridResult fields.
    """
    return pd.DataFrame([
        {
            "alpha_tot": r.alpha_tot,
            "s_wind": r.s_wind,
            "pv_capacity_mw": r.pv_capacity_mw,
            "wind_capacity_mw": r.wind_capacity_mw,
            "wacc": r.wacc,
            "npv_eur": r.npv_eur,
            "irr": r.irr,
            "annual_revenue_eur_total": r.annual_revenue_eur_total,
            "annual_revenue_eur_pv": r.annual_revenue_eur_pv,
            "annual_revenue_eur_wind": r.annual_revenue_eur_wind,
            "annual_export_mwh_total": r.annual_export_mwh_total,
            "annual_export_mwh_pv": r.annual_export_mwh_pv,
            "annual_export_mwh_wind": r.annual_export_mwh_wind,
            "annual_curtailment_mwh_total": r.annual_curtailment_mwh_total,
            "annual_curtailment_mwh_pv": r.annual_curtailment_mwh_pv,
            "annual_curtailment_mwh_wind": r.annual_curtailment_mwh_wind,
            "export_per_mw_poi_mwh": r.export_per_mw_poi_mwh,
            "poi_utilisation_factor": r.poi_utilisation_factor,
            "capex_pv_eur": r.capex_pv_eur,
            "capex_wind_eur": r.capex_wind_eur,
            "capex_total_eur": r.capex_total_eur,
            "annual_opex_pv_eur": r.annual_opex_pv_eur,
            "annual_opex_wind_eur": r.annual_opex_wind_eur,
            "annual_opex_total_eur": r.annual_opex_total_eur,
        }
        for r in results
    ])


def pivot_npv_heatmap(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the results DataFrame to create a heatmap-ready matrix.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame from results_to_dataframe.

    Returns
    -------
    pd.DataFrame
        Pivoted DataFrame with alpha_tot as index, s_wind as columns, NPV as values.
    """
    return df_results.pivot(index="alpha_tot", columns="s_wind", values="npv_eur")


# =============================================================================
# Example Usage
# =============================================================================
"""
Example usage of the hybrid analysis module:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hybrid_analysis import (
    evaluate_hybrid_grid,
    find_best_hybrid_by_npv,
    results_to_dataframe,
    pivot_npv_heatmap,
)

# Assume df is your cleaned DataFrame with columns:
#   'timestamp', 'price_eur_per_mwh', 'pv_per_kw', 'wind_per_kw'

# Define grids
alpha_tot_values = [round(x, 1) for x in np.arange(1.0, 3.1, 0.1)]  # 1.0 to 3.0
s_wind_values = [0.0, 0.25, 0.5, 0.75, 1.0]

# Evaluate all combinations
hybrid_results = evaluate_hybrid_grid(df, alpha_tot_values, s_wind_values)

# Find optimal configuration
best = find_best_hybrid_by_npv(hybrid_results)
print(f"Optimal hybrid: α_tot={best.alpha_tot:.1f}, s_wind={best.s_wind:.0%}")
print(f"  PV: {best.pv_capacity_mw:.1f} MW, Wind: {best.wind_capacity_mw:.1f} MW")
print(f"  NPV: €{best.npv_eur:,.0f}")

# Convert to DataFrame
df_hybrid = results_to_dataframe(hybrid_results)

# Create NPV heatmap
npv_matrix = pivot_npv_heatmap(df_hybrid)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(npv_matrix.values / 1e6, aspect='auto', origin='lower', cmap='RdYlGn')

# Set axis labels
ax.set_xticks(range(len(npv_matrix.columns)))
ax.set_xticklabels([f"{x:.0%}" for x in npv_matrix.columns])
ax.set_yticks(range(len(npv_matrix.index)))
ax.set_yticklabels([f"{x:.1f}" for x in npv_matrix.index])

ax.set_xlabel("Wind Share (s_wind)")
ax.set_ylabel("Total Overplanting Factor (α_tot)")
ax.set_title("NPV Heatmap: Hybrid PV+Wind Configurations")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("NPV (Million EUR)")

# Mark optimal point
best_row = list(npv_matrix.index).index(best.alpha_tot)
best_col = list(npv_matrix.columns).index(best.s_wind)
ax.scatter([best_col], [best_row], color='black', s=200, marker='*', edgecolors='white', linewidths=2)

plt.tight_layout()
plt.savefig("hybrid_npv_heatmap.png", dpi=150)
plt.show()
```
"""
