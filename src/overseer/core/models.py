from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime

@dataclass
class InputPaths:
    fuels_and_topography_directory: str
    asp_filename: str
    cbd_filename: str
    cbh_filename: str
    cc_filename: str
    ch_filename: str
    dem_filename: str
    fbfm_filename: str
    slp_filename: str
    adj_filename: str
    phi_filename: str
    weather_directory: str
    ws_filename: str
    wd_filename: str
    m1_filename: str
    m10_filename: str
    m100_filename: str
    # Keep the existing fields
    fire: str
    vegetation: str
    elevation: str
    wind: str
    fuel_moisture: str



@dataclass
class OutputPaths:
    time_of_arrival: str
    fire_intensity: str
    flame_length: str
    spread_rate: str
    # Add any other output file paths here

@dataclass
class SimulationState:
    timestamp: datetime
    fire_intensity: np.ndarray
    burned_area: float
    fire_perimeter_length: float
    containment_percentage: float
    resources: Dict[str, int]
    weather: Dict[str, float]
    
    # Add any other relevant state information


@dataclass
class SimulationConfig:
    elmfire_data: Dict[str, Dict[str, Any]]
    input_paths: InputPaths
    output_paths: OutputPaths

    def get_parameter(self, section: str, key: str) -> Any:
        value = self.elmfire_data.get(section, {}).get(key)
        if value is not None:
            # Try to convert to float first, then to int if possible
            try:
                float_value = float(value)
                if float_value.is_integer():
                    return int(float_value)
                return float_value
            except ValueError:
                # If conversion fails, return the original string value
                return value
        return None
    

@dataclass
class SimulationResult:
    final_state: SimulationState
    performance_metrics: Dict[str, float]
    output_files: Dict[str, str]  # File type to file path mapping
    execution_time: float
    metadata: Dict[str, Any]


@dataclass
class Action:
    fireline_coordinates: List[Tuple[int, int]]

@dataclass
class EpisodeStep:
    step: int
    state: SimulationState
    action: Action
    reward: float
    next_state: SimulationState
    simulation_result: SimulationResult
    done: bool

@dataclass
class Episode:
    episode_id: int
    steps: List[EpisodeStep]
    total_reward: float
    total_steps: int
    final_burned_area: float
    final_containment_percentage: float
    execution_time: float

# Remove WeatherData, GeospatialData, and GeospatialPaths classes

# Keep other necessary classes like RLMetrics if needed