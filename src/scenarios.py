"""
Scenario definitions for different renewable energy project configurations.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Scenario:
    """
    Data class representing a renewable energy project scenario.
    
    Attributes:
        name: Scenario identifier (e.g., 'W', 'P', 'H', 'PB', 'WB')
        description: Human-readable description
        wind_capacity_mw: Installed wind capacity in MW (0 if not applicable)
        pv_capacity_mw: Installed PV capacity in MW (0 if not applicable)
        battery_power_mw: Battery power rating in MW (0 if not applicable)
        battery_capacity_mwh: Battery energy capacity in MWh (0 if not applicable)
        battery_duration_h: Battery duration in hours (capacity/power ratio)
    """
    name: str
    description: str
    wind_capacity_mw: float = 0.0
    pv_capacity_mw: float = 0.0
    battery_power_mw: float = 0.0
    battery_capacity_mwh: float = 0.0
    
    @property
    def battery_duration_h(self) -> float:
        """Calculate battery duration from capacity and power."""
        if self.battery_power_mw > 0:
            return self.battery_capacity_mwh / self.battery_power_mw
        return 0.0
    
    def has_battery(self) -> bool:
        """Check if scenario includes battery storage."""
        return self.battery_power_mw > 0 and self.battery_capacity_mwh > 0
    
    def has_wind(self) -> bool:
        """Check if scenario includes wind generation."""
        return self.wind_capacity_mw > 0
    
    def has_pv(self) -> bool:
        """Check if scenario includes PV generation."""
        return self.pv_capacity_mw > 0


def generate_baseline_scenarios() -> List[Scenario]:
    """
    Generate baseline scenarios for testing and analysis.
    
    Returns:
        List of Scenario objects including:
        - W: Stand-alone wind (10 MW)
        - P: Stand-alone PV (10 MW)
        - H: Co-located wind and PV sharing 10 MW POI
        - PB: PV + battery (example: 10 MW PV + 5 MW / 10 MWh battery)
        - WB: Wind + battery (example: 10 MW wind + 5 MW / 10 MWh battery)
    """
    scenarios = [
        Scenario(
            name='W',
            description='Stand-alone wind behind 10 MW grid connection',
            wind_capacity_mw=10.0,
            pv_capacity_mw=0.0,
            battery_power_mw=0.0,
            battery_capacity_mwh=0.0
        ),
        Scenario(
            name='P',
            description='Stand-alone PV behind 10 MW grid connection',
            wind_capacity_mw=0.0,
            pv_capacity_mw=10.0,
            battery_power_mw=0.0,
            battery_capacity_mwh=0.0
        ),
        Scenario(
            name='H',
            description='Co-located wind and PV sharing 10 MW grid connection',
            wind_capacity_mw=5.0,  # Example: 5 MW wind + 5 MW PV
            pv_capacity_mw=5.0,
            battery_power_mw=0.0,
            battery_capacity_mwh=0.0
        ),
        Scenario(
            name='PB',
            description='PV plus battery behind 10 MW grid connection (green battery)',
            wind_capacity_mw=0.0,
            pv_capacity_mw=10.0,
            battery_power_mw=5.0,  # Example: 5 MW power rating
            battery_capacity_mwh=10.0  # Example: 10 MWh capacity (2h duration)
        ),
        Scenario(
            name='WB',
            description='Wind plus battery behind 10 MW grid connection (green battery)',
            wind_capacity_mw=10.0,
            pv_capacity_mw=0.0,
            battery_power_mw=5.0,  # Example: 5 MW power rating
            battery_capacity_mwh=10.0  # Example: 10 MWh capacity (2h duration)
        )
    ]
    
    return scenarios


def get_scenario_by_name(scenarios: List[Scenario], name: str) -> Optional[Scenario]:
    """
    Retrieve a scenario by its name.
    
    Args:
        scenarios: List of Scenario objects
        name: Scenario name to search for
        
    Returns:
        Scenario object if found, None otherwise
    """
    for scenario in scenarios:
        if scenario.name == name:
            return scenario
    return None






