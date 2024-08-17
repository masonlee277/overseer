from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from datetime import datetime
from pathlib import Path
from enum import Enum


@dataclass
class InputPaths:
    fuels_and_topography_directory: Path
    asp_filename: Path
    cbd_filename: Path
    cbh_filename: Path
    cc_filename: Path
    ch_filename: Path
    dem_filename: Path
    fbfm_filename: Path
    slp_filename: Path
    adj_filename: Path
    phi_filename: Path
    weather_directory: Path
    ws_filename: Path
    wd_filename: Path
    m1_filename: Path
    m10_filename: Path
    m100_filename: Path

    @classmethod
    def get_absolute_path(cls, instance: 'InputPaths', filename: Union[str, Path]) -> Path:
        """
        Get the absolute path for a given filename, ensuring it has a .tif extension.

        Args:
            instance (InputPaths): An instance of the InputPaths class.
            filename (Union[str, Path]): The filename to get the absolute path for.

        Returns:
            Path: The absolute path for the given filename with .tif extension.

        Raises:
            ValueError: If the filename is not a valid attribute of InputPaths.
        """
        filename = Path(filename)
        if filename.name not in instance.__dict__:
            raise ValueError(f"{filename.name} is not a valid attribute of InputPaths")

        # Determine the correct directory
        if filename in [instance.ws_filename, instance.wd_filename, instance.m1_filename, instance.m10_filename, instance.m100_filename]:
            base_dir = instance.weather_directory
        else:
            base_dir = instance.fuels_and_topography_directory

        # Ensure the filename has .tif extension
        if filename.suffix.lower() != '.tif':
            filename = filename.with_suffix('.tif')

        return base_dir / filename


@dataclass
class OutputPaths:
    time_of_arrival: Path
    fire_intensity: Path
    flame_length: Path
    spread_rate: Path
    # Add any other output file paths here

@dataclass
class SimulationPaths:
    input_paths: InputPaths
    output_paths: OutputPaths

@dataclass
class SimulationConfig:
    sections: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __init__(self, sections: Dict[str, Dict[str, Any]] = None):
        self.sections = sections or {}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(sections=data)

    def to_dict(self) -> Dict[str, Any]:
        return self.sections

    def get_parameter(self, section: str, key: str) -> Any:
        return self.sections.get(section, {}).get(key)

    def set_parameter(self, section: str, key: str, value: Any) -> None:
        if section not in self.sections:
            self.sections[section] = {}
        self.sections[section][key] = value
    
@dataclass
class SimulationMetrics:
    burned_area: float
    fire_perimeter_length: float
    containment_percentage: float
    execution_time: float
    performance_metrics: Dict[str, float]
    fire_intensity: np.ndarray

@dataclass
class SimulationState:
    timestamp: datetime
    config: SimulationConfig
    paths: SimulationPaths
    metrics: SimulationMetrics
    save_path: Path
    resources: Dict[str, int]
    weather: Dict[str, float]

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
    done: bool

@dataclass
class Episode:
    episode_id: int
    steps: List[EpisodeStep]
    total_reward: float
    total_steps: int


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"



@dataclass
class SimulationResult:
    job_id: str
    status: JobStatus
    duration: Optional[float] = None
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None

