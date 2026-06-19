# System Architecture Diagrams

## 1. High-Level Module Architecture

```mermaid
flowchart TB
    subgraph DataLayer["Data Layer"]
        RAW["Raw Data Files<br/>(Renewables.ninja, SMARD)"]
        DIO["data_io.py<br/>• load_wind_raw()<br/>• load_pv_raw()<br/>• load_prices_raw()"]
        PRE["preprocess.py<br/>• build_clean_dataset()<br/>• Align time series"]
        CLEAN["Cleaned Dataset<br/>(8760 hourly rows)"]
    end

    subgraph ConfigLayer["Configuration Layer"]
        CFG["config.py<br/>• Techno-economic parameters<br/>• POI capacity<br/>• WACC, CAPEX, OPEX<br/>• Innovation Tender params"]
        SCN["scenarios.py<br/>• Scenario dataclass<br/>• Baseline scenarios"]
    end

    subgraph SimulationLayer["Simulation Layer"]
        SIM["simulation.py<br/>• simulate_hybrid_year()<br/>• simulate_single_tech_year()<br/>• POI constraint logic"]
        LP["simulation_lp.py<br/>• LP optimization (PuLP)<br/>• Green battery dispatch<br/>• Innovation Tender FMP"]
    end

    subgraph AnalysisLayer["Analysis Layer"]
        OVS["oversizing_analysis.py<br/>Stand-alone PV/Wind"]
        HYB["hybrid_analysis.py<br/>Co-located PV+Wind"]
        BESS["hybrid_bess_analysis.py<br/>PV+Wind+BESS"]
        PVB["pv_battery_analysis.py<br/>Greenfield PV+BESS"]
    end

    subgraph EconomicsLayer["Economics Layer"]
        ECO["economics.py<br/>• NPV, IRR calculation<br/>• Cash flow builders<br/>• LCOE computation<br/>• PvBessNpvResult"]
    end

    subgraph OutputLayer["Output & Visualization"]
        NB["Jupyter Notebooks<br/>01-07 Analysis"]
        OUT["Results<br/>• NPV heatmaps<br/>• Optimal configs<br/>• Sensitivity plots"]
    end

    RAW --> DIO
    DIO --> PRE
    PRE --> CLEAN
    
    CFG --> SIM
    CFG --> LP
    CFG --> ECO
    SCN --> AnalysisLayer
    
    CLEAN --> SIM
    CLEAN --> LP
    
    SIM --> OVS
    SIM --> HYB
    LP --> BESS
    LP --> PVB
    
    OVS --> ECO
    HYB --> ECO
    BESS --> ECO
    PVB --> ECO
    
    ECO --> OUT
    AnalysisLayer --> NB
    NB --> OUT
```

## 2. Class/Dataclass Diagram

```mermaid
classDiagram
    class Config {
        +float poi_capacity_mw
        +Path data_raw_dir
        +Path data_processed_dir
        +str timezone
        +int analysis_year
        +int price_year
        +float capex_pv_eur_per_kw
        +float opex_pv_eur_per_kw_per_year
        +int lifetime_pv_years
        +float wacc_pv_real
        +float degradation_pv_per_year
        +float capex_wind_eur_per_kw
        +float opex_wind_eur_per_kw_per_year
        +int lifetime_wind_years
        +float wacc_wind_real
    }

    class Scenario {
        +str name
        +str description
        +float wind_capacity_mw
        +float pv_capacity_mw
        +float battery_power_mw
        +float battery_capacity_mwh
        +battery_duration_h() float
        +has_battery() bool
        +has_wind() bool
        +has_pv() bool
    }

    class OversizingResult {
        +str tech
        +float alpha
        +float capacity_mw
        +float npv_eur
        +float irr
        +float annual_revenue_eur
        +float annual_export_mwh
        +float annual_curtailment_mwh
        +float export_per_mw_poi_mwh
        +float poi_utilisation_factor
    }

    class HybridResult {
        +float alpha_tot
        +float s_wind
        +float pv_capacity_mw
        +float wind_capacity_mw
        +float wacc
        +float npv_eur
        +float irr
        +float annual_revenue_eur_total
        +float annual_export_mwh_total
        +float annual_curtailment_mwh_total
        +float poi_utilisation_factor
    }

    class HybridBessResult {
        +float alpha_tot
        +float s_wind
        +float gamma
        +float pv_capacity_mw
        +float wind_capacity_mw
        +float bess_power_mw
        +float bess_energy_mwh
        +float wacc
        +float npv_eur
        +float irr
        +float capex_total_eur
        +float annual_revenue_eur
        +float annual_export_mwh_total
        +float battery_cycles
        +float poi_utilisation
        +float curtailment_rate
    }

    class PvBessResult {
        +float alpha_pv
        +float gamma
        +float pv_capacity_mw
        +float batt_power_mw
        +float batt_energy_mwh
        +float wacc
        +float npv_eur
        +float annual_revenue_eur
        +float annual_export_mwh_total
        +float annual_curtailment_mwh
        +float poi_utilisation
        +float storage_throughput_mwh
        +float irr
    }

    class PvBessNpvResult {
        +float capex_pv_eur
        +float capex_bess_eur
        +float capex_total_eur
        +float annual_revenue_year1_eur
        +float annual_opex_total_eur
        +int bess_replacement_year
        +float bess_replacement_cost_eur
        +float npv_eur
        +float irr
        +float simple_payback_years
        +ndarray cash_flows
        +ndarray discounted_cash_flows
    }

    Config --> Scenario : configures
    Scenario --> OversizingResult : produces
    Scenario --> HybridResult : produces
    Scenario --> HybridBessResult : produces
    Scenario --> PvBessResult : produces
    PvBessResult ..> PvBessNpvResult : detailed_economics
```

## 3. Data Flow Diagram

```mermaid
flowchart LR
    subgraph Input["Input Data"]
        WIND["Wind Generation<br/>(Renewables.ninja)"]
        PV["PV Generation<br/>(Renewables.ninja)"]
        PRICE["Day-Ahead Prices<br/>(SMARD)"]
    end

    subgraph Processing["Data Processing"]
        LOAD["Load & Parse"]
        ALIGN["Temporal Alignment<br/>(8760 hours)"]
        MERGE["Merge to Single<br/>DataFrame"]
    end

    subgraph Simulation["Simulation Engine"]
        direction TB
        GEN["Generation<br/>Calculation"]
        POI["POI Constraint<br/>Application"]
        BATT["Battery Dispatch<br/>(LP Optimization)"]
        CURT["Curtailment<br/>Calculation"]
    end

    subgraph Economics["Economic Analysis"]
        REV["Revenue<br/>Calculation"]
        CF["Cash Flow<br/>Construction"]
        NPV["NPV/IRR<br/>Calculation"]
    end

    subgraph Output["Results"]
        OPT["Optimal<br/>Configuration"]
        HEAT["NPV<br/>Heatmaps"]
        SENS["Sensitivity<br/>Analysis"]
    end

    WIND --> LOAD
    PV --> LOAD
    PRICE --> LOAD
    LOAD --> ALIGN
    ALIGN --> MERGE
    MERGE --> GEN
    GEN --> POI
    POI --> BATT
    POI --> CURT
    BATT --> CURT
    CURT --> REV
    REV --> CF
    CF --> NPV
    NPV --> OPT
    NPV --> HEAT
    NPV --> SENS
```

## 4. LP Optimization Model Structure

```mermaid
flowchart TB
    subgraph Objective["Objective Function"]
        OBJ["Maximize: Σ export[t] × (price[t] + market_premium[t])"]
    end

    subgraph Variables["Decision Variables"]
        CH["charge[t] ≥ 0<br/>Battery charging power"]
        DIS["discharge[t] ≥ 0<br/>Battery discharging power"]
        SOC["soc[t] ≥ 0<br/>State of charge"]
        EXP["export[t] ≥ 0<br/>Grid export power"]
        CUR["curtailment[t] ≥ 0<br/>Curtailed power"]
    end

    subgraph Constraints["Constraints"]
        C1["Energy Balance:<br/>export = gen - charge - curtailment + discharge"]
        C2["POI Limit:<br/>export ≤ POI_MW"]
        C3["SOC Dynamics:<br/>soc[t] = soc[t-1] + charge×η_ch - discharge/η_dis"]
        C4["SOC Bounds:<br/>0 ≤ soc ≤ BESS_cap"]
        C5["Green Battery:<br/>charge ≤ renewable_gen"]
        C6["Discharge Limit:<br/>Σ discharge ≤ max_cycles × BESS_cap"]
        C7["Cyclic SOC:<br/>soc[end] = 0"]
    end

    Objective --> Variables
    Variables --> Constraints
```

## 5. Innovation Tender (Innovationsausschreibung) Logic

```mermaid
flowchart TB
    subgraph Eligibility["BESS Sizing Requirements"]
        E1["P_BESS ≥ 0.25 × P_PV"]
        E2["E_BESS ≥ 2.0 × P_BESS<br/>(min 2h duration)"]
    end

    subgraph FMP["Floating Market Premium Calculation"]
        MV["Monthly Market Value<br/>= Σ(gen × price) / Σ(gen)"]
        MP["Market Premium<br/>= max(0, Strike - MV)"]
        HP["Hourly Premium<br/>= 0 if price < 0<br/>= MP otherwise"]
    end

    subgraph CashFlow["Cash Flow Structure"]
        Y1["Years 1-20:<br/>Revenue with Premium"]
        Y2["Years 21-30:<br/>Merchant Revenue"]
        Y3["Year 15:<br/>BESS Replacement (30% CAPEX)"]
    end

    subgraph Comparison["Scenario Comparison"]
        INN["Innovation Tender<br/>(Strike: 83.3 €/MWh)"]
        MER["Merchant Only<br/>(Market prices)"]
    end

    Eligibility --> FMP
    FMP --> CashFlow
    CashFlow --> Comparison
```

## 6. Analysis Workflow

```mermaid
flowchart TB
    subgraph Setup["1. Setup"]
        LOAD["Load cleaned dataset"]
        PARAMS["Define parameter grid:<br/>• α_pv (oversizing)<br/>• s_wind (wind share)<br/>• γ (battery ratio)"]
    end

    subgraph GridEval["2. Grid Evaluation"]
        LOOP["For each configuration"]
        SIM["Run simulation"]
        ECON["Calculate economics"]
        STORE["Store result"]
    end

    subgraph Analysis["3. Analysis"]
        BEST["Find optimal by NPV"]
        PIVOT["Create pivot tables"]
        VIZ["Generate visualizations"]
    end

    subgraph Scenarios["Scenario Types"]
        S1["Stand-alone PV/Wind<br/>(oversizing_analysis)"]
        S2["Hybrid PV+Wind<br/>(hybrid_analysis)"]
        S3["PV+BESS<br/>(pv_battery_analysis)"]
        S4["PV+Wind+BESS<br/>(hybrid_bess_analysis)"]
    end

    Setup --> GridEval
    LOAD --> LOOP
    PARAMS --> LOOP
    LOOP --> SIM
    SIM --> ECON
    ECON --> STORE
    STORE --> LOOP
    GridEval --> Analysis
    STORE --> BEST
    BEST --> PIVOT
    PIVOT --> VIZ

    Scenarios -.-> GridEval
```

---

## Usage Notes

These diagrams are written in Mermaid format and can be rendered:

1. **GitHub/GitLab**: Directly in markdown files
2. **VS Code**: With Mermaid extension
3. **Mermaid Live Editor**: https://mermaid.live/
4. **Export to PDF/PNG**: Via Mermaid CLI or online tools
5. **LaTeX**: Convert to TikZ or include as image

### Export Commands (if mermaid-cli is installed)

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Export to PNG
mmdc -i architecture_diagram.md -o architecture.png -w 1600

# Export to PDF
mmdc -i architecture_diagram.md -o architecture.pdf
```
