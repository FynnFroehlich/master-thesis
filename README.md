# Cable Pooling and Co-located Renewables Simulation Model

A time series simulation model for analyzing renewable energy projects with cable pooling in Brandenburg, Germany.

## Project Structure

```
.
├── data_raw/              # Original CSV files from Renewables.ninja and SMARD
├── data_processed/        # Processed parquet files
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_data_sanity_check.ipynb
│   ├── 02_oversizing_analysis.ipynb
│   ├── 03_pv_bess_analysis.ipynb
│   ├── 04_pv_wind_bess_analysis.ipynb
│   ├── 05_wind_bess_analysis.ipynb
│   ├── 06_high_level_comparison.ipynb
│   └── 07_standalone_bess_analysis.ipynb
├── src/                   # Source code modules
│   ├── config.py              # Configuration constants
│   ├── data_io.py             # Data loading functions
│   ├── preprocess.py          # Data cleaning and alignment
│   ├── simulation.py          # Core simulation functions
│   ├── simulation_lp.py       # LP optimization for battery dispatch
│   ├── economics.py           # Economic evaluation (NPV, IRR, LCOE)
│   ├── scenarios.py           # Scenario definitions
│   ├── oversizing_analysis.py # Stand-alone PV/wind oversizing analysis
│   ├── hybrid_analysis.py     # PV+wind hybrid analysis
│   ├── pv_battery_analysis.py # PV+BESS greenfield analysis
│   └── hybrid_bess_analysis.py # PV+wind+BESS hybrid analysis
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Prepare Data

Place your raw data files in `data_raw/`:
- Wind generation data from Renewables.ninja (2019)
- PV generation data from Renewables.ninja (2019)
- Day-ahead prices from SMARD (2023, DE-LU bidding zone)

### 2. Preprocess Data

```python
from src.preprocess import build_clean_dataset, save_clean_dataset

df = build_clean_dataset(
    wind_path="data_raw/wind_data.csv",
    pv_path="data_raw/pv_data.csv",
    price_path="data_raw/price_data.csv"
)
save_clean_dataset(df)
```

### 3. Run Simulations

```python
from src.simulation import simulate_hybrid_year
from src.scenarios import generate_baseline_scenarios

scenarios = generate_baseline_scenarios()
scenario_h = scenarios[2]  # Hybrid scenario

result = simulate_hybrid_year(
    df=df,
    pv_capacity_mw=scenario_h.pv_capacity_mw,
    wind_capacity_mw=scenario_h.wind_capacity_mw
)
```

### 4. Economic Evaluation

```python
from src.economics import evaluate_project

metrics = evaluate_project(
    sim_df=sim_df,
    capex_eur=10_000_000.0,
    opex_eur_per_year=100_000.0,
    lifetime_years=20,
    discount_rate=0.05
)
```

## Scenarios

The model supports the following scenarios:
- **W**: Stand-alone wind (10 MW)
- **P**: Stand-alone PV (10 MW)
- **H**: Co-located wind and PV sharing 10 MW POI
- **PB**: PV + battery (green battery)
- **WB**: Wind + battery (green battery)

## Notes

- All simulations assume a 10 MW Point of Interconnection (POI) capacity constraint
- Battery scenarios use "green battery" logic (only charges from on-site renewables)
- LP optimization provides optimal battery dispatch with perfect price foresight





