"""
Configuration constants and settings for the simulation model.
"""

from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# Point of Interconnection (POI)
# =============================================================================
POI_CAPACITY_MW = 10.0

# =============================================================================
# PV Techno-Economic Parameters (all values in real EUR)
# =============================================================================
CAPEX_PV_EUR_PER_KW = 700.0
OPEX_PV_EUR_PER_KW_PER_YEAR = 13.3
LIFETIME_PV_YEARS = 30
WACC_PV_REAL = 0.035
DEGRADATION_PV_PER_YEAR = 0.0025

# =============================================================================
# Wind Techno-Economic Parameters (all values in real EUR)
# =============================================================================
CAPEX_WIND_EUR_PER_KW = 1300.0
OPEX_WIND_EUR_PER_KW_PER_YEAR = 32.0
LIFETIME_WIND_YEARS = 25
WACC_WIND_REAL = 0.039
DEGRADATION_WIND_PER_YEAR = 0.0  # No degradation for wind

# =============================================================================
# Battery Energy Storage System (BESS) Parameters
# =============================================================================
BESS_ROUNDTRIP_EFF = 0.90  # Round-trip efficiency (90%)
CAPEX_BESS_EUR_PER_KWH = 400.0  # €/kWh capacity
OPEX_BESS_EUR_PER_KW_PER_YEAR = 5.3  # €/kW power/year
LIFETIME_BESS_YEARS = 15
WACC_BESS_REAL = 0.025  # 2.5% real WACC
BESS_REPLACEMENT_SHARE = 0.30  # Battery replacement at 30% of initial CAPEX

# =============================================================================
# Standalone BESS Scenario Parameters
# =============================================================================
# Updated from 13.72 EUR/kW-month average (164.63 EUR/kW-year) and normalized to
# 90% round-trip efficiency: 164.63 * (0.90 / 0.86) = 172.287 EUR/kW-year.
STANDALONE_BESS_ANNUAL_REVENUE_EUR_PER_MW = 172287.21
STANDALONE_BESS_POWER_MW = 10.0

# Annual full-equivalent discharge cycles cap (limits battery degradation)
BESS_MAX_ANNUAL_DISCHARGE_CYCLES = 300.0

# =============================================================================
# Project-Level Parameters
# =============================================================================
PROJECT_LIFETIME_YEARS = 30  # Project evaluation period

# =============================================================================
# EEG Innovationsausschreibung 2024 Parameters
# =============================================================================
# Floating Market Premium (FMP) for Innovation Tender 2024
# Strike price: 8.33 ct/kWh = 0.0833 EUR/kWh (matches the constant below)
INNOVATION_TENDER_STRIKE_PRICE_EUR_PER_KWH = 0.0833  # EUR/kWh
INNOVATION_TENDER_STRIKE_PRICE_EUR_PER_MWH = 83.3    # EUR/MWh (for convenience)
INNOVATION_TENDER_PREMIUM_YEARS = 20  # Market premium duration before merchant regime

# BESS sizing requirements for Innovation Tender eligibility:
# - P_BESS >= 0.25 × P_PV (battery power >= 25% of PV power)
# - E_BESS >= 2 × P_BESS (battery energy >= 2h duration at rated power)
INNOVATION_TENDER_MIN_BESS_POWER_RATIO = 0.25  # P_BESS / P_PV >= 0.25
INNOVATION_TENDER_MIN_BESS_DURATION_H = 2.0    # E_BESS / P_BESS >= 2.0 hours

# =============================================================================
# Default Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data_raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data_processed"

# Ensure directories exist
DATA_RAW_DIR.mkdir(exist_ok=True)
DATA_PROCESSED_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    """Main configuration class for simulation parameters."""
    
    # Grid connection capacity
    poi_capacity_mw: float = POI_CAPACITY_MW
    
    # Data paths
    data_raw_dir: Path = DATA_RAW_DIR
    data_processed_dir: Path = DATA_PROCESSED_DIR
    
    # Time zone settings
    timezone: str = "Europe/Berlin"
    
    # Year for analysis
    analysis_year: int = 2024  # Generation data year
    price_year: int = 2024     # Price data year
    
    # PV parameters
    capex_pv_eur_per_kw: float = CAPEX_PV_EUR_PER_KW
    opex_pv_eur_per_kw_per_year: float = OPEX_PV_EUR_PER_KW_PER_YEAR
    lifetime_pv_years: int = LIFETIME_PV_YEARS
    wacc_pv_real: float = WACC_PV_REAL
    degradation_pv_per_year: float = DEGRADATION_PV_PER_YEAR
    
    # Wind parameters
    capex_wind_eur_per_kw: float = CAPEX_WIND_EUR_PER_KW
    opex_wind_eur_per_kw_per_year: float = OPEX_WIND_EUR_PER_KW_PER_YEAR
    lifetime_wind_years: int = LIFETIME_WIND_YEARS
    wacc_wind_real: float = WACC_WIND_REAL
    degradation_wind_per_year: float = DEGRADATION_WIND_PER_YEAR
