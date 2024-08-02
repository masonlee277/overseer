# src/overseer/core/state.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
import numpy as np
from datetime import datetime
from overseer.data.data_manager import DataManager
from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.core.models import SimulationState, InputPaths, OutputPaths

class State:
    """
    Represents a single state in the ELMFIRE simulation.

    This class is primarily a data container, storing raw state data and basic statistics.
    It delegates complex calculations to the DataManager.

    Attributes:
        raw_data (Dict[str, Any]): The raw state data from the simulation.
        data_manager (DataManager): Reference to the DataManager for complex calculations.
        simulation_state (SimulationState): The processed simulation state.
        input_paths (InputPaths): Paths to input data files.
        output_paths (OutputPaths): Paths to output data files.
        timestamp (float): The timestamp of this state.
    """

    def __init__(self, raw_data: Dict[str, Any], data_manager: DataManager):
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.raw_data = raw_data
        self.data_manager = data_manager
        self.timestamp = raw_data.get('timestamp', 0.0)
        self.simulation_state = self._process_raw_data()
        self.input_paths = self._extract_input_paths()
        self.output_paths = self._extract_output_paths()

    def _process_raw_data(self) -> SimulationState:
        """Process raw data into a SimulationState object."""
        self.logger.info(f"Processing raw data for state at timestamp {self.timestamp}")
        fire_intensity = self.raw_data['fire_intensity']
        return SimulationState(
            timestamp=datetime.fromtimestamp(self.timestamp),
            fire_intensity=fire_intensity,
            burned_area=self.data_manager.calculate_burned_area(fire_intensity),
            fire_perimeter_length=self.data_manager.calculate_fire_perimeter_length(fire_intensity),
            containment_percentage=self.data_manager.calculate_fire_containment_percentage(self.raw_data),
            resources=self.raw_data.get('resources_deployed', {}),
            weather={
                'wind_speed': self.raw_data.get('wind_speed'),
                'wind_direction': self.raw_data.get('wind_direction'),
                'temperature': self.raw_data.get('temperature'),
                'humidity': self.raw_data.get('humidity')
            }
        )

    def _extract_input_paths(self) -> InputPaths:
        """Extract input paths from raw data."""
        return InputPaths(
            fuels_and_topography_directory=self.raw_data.get('fuels_and_topography_directory', ''),
            asp_filename=self.raw_data.get('asp_filename', ''),
            cbd_filename=self.raw_data.get('cbd_filename', ''),
            cbh_filename=self.raw_data.get('cbh_filename', ''),
            cc_filename=self.raw_data.get('cc_filename', ''),
            ch_filename=self.raw_data.get('ch_filename', ''),
            dem_filename=self.raw_data.get('dem_filename', ''),
            fbfm_filename=self.raw_data.get('fbfm_filename', ''),
            slp_filename=self.raw_data.get('slp_filename', ''),
            adj_filename=self.raw_data.get('adj_filename', ''),
            phi_filename=self.raw_data.get('phi_filename', ''),
            weather_directory=self.raw_data.get('weather_directory', ''),
            ws_filename=self.raw_data.get('ws_filename', ''),
            wd_filename=self.raw_data.get('wd_filename', ''),
            m1_filename=self.raw_data.get('m1_filename', ''),
            m10_filename=self.raw_data.get('m10_filename', ''),
            m100_filename=self.raw_data.get('m100_filename', ''),
            fire=self.raw_data.get('fire_intensity_path', ''),
            vegetation=self.raw_data.get('vegetation_path', ''),
            elevation=self.raw_data.get('elevation_path', ''),
            wind=self.raw_data.get('wind_path', ''),
            fuel_moisture=self.raw_data.get('fuel_moisture_path', '')
        )

    def _extract_output_paths(self) -> OutputPaths:
        """Extract output paths from raw data."""
        return OutputPaths(
            time_of_arrival=self.raw_data.get('time_of_arrival_path', ''),
            fire_intensity=self.raw_data.get('fire_intensity_output_path', ''),
            flame_length=self.raw_data.get('flame_length_path', ''),
            spread_rate=self.raw_data.get('spread_rate_path', '')
        )

    def get_raw_data(self) -> Dict[str, Any]:
        """Return the raw state data."""
        return self.raw_data

    def get_simulation_state(self) -> SimulationState:
        """Return the processed simulation state."""
        return self.simulation_state

    def get_input_paths(self) -> InputPaths:
        """Return the input file paths."""
        return self.input_paths

    def get_output_paths(self) -> OutputPaths:
        """Return the output file paths."""
        return self.output_paths

    def get_fire_growth_rate(self, time_window: float) -> float:
        """Get the fire growth rate over a specified time window."""
        return self.data_manager.calculate_fire_growth_rate(self.raw_data, time_window)

    def get_resource_efficiency(self) -> float:
        """Calculate the resource efficiency for this state."""
        return self.data_manager.calculate_resource_efficiency(self.raw_data)

    def get_high_risk_areas(self) -> np.ndarray:
        """Identify high-risk areas based on the current state."""
        return self.data_manager.identify_high_risk_areas(
            self.simulation_state.fire_intensity,
            self.raw_data.get('elevation'),
            self.raw_data.get('fuel_type')
        )

    def __str__(self) -> str:
        """Return a string representation of the state."""
        return f"State(timestamp={self.timestamp}, burned_area={self.simulation_state.burned_area:.2f})"

    def __repr__(self) -> str:
        """Return a string representation of the state for debugging."""
        return self.__str__()


def main():
    # Setup mock config and data manager
    config = OverseerConfig()
    
    class MockDataManager:
        def calculate_burned_area(self, fire_intensity):
            return np.sum(fire_intensity > 0)
        
        def calculate_fire_perimeter_length(self, fire_intensity):
            return 15.2
        
        def calculate_fire_growth_rate(self, raw_data, time_window):
            return 2.5
        
        def calculate_resource_efficiency(self, raw_data):
            return 0.75
        
        def calculate_fire_containment_percentage(self, raw_data):
            return 60.0
        
        def identify_high_risk_areas(self, fire_intensity, elevation, fuel_type):
            return np.array([[True, False], [False, True]])

    data_manager = MockDataManager()

    # Create a sample raw data dictionary
    raw_data = {
        'timestamp': 3600.0,
        'fire_intensity': np.array([[0, 0, 1, 2], [0, 1, 3, 2], [0, 0, 1, 1]]),
        'resources_deployed': {'firefighters': 50, 'trucks': 10, 'aircraft': 2},
        'wind_speed': 15.5,
        'wind_direction': 225.0,
        'temperature': 30.5,
        'humidity': 20.0,
        'fuel_moisture': np.array([[10, 12, 8], [9, 11, 10], [8, 9, 11]]),
        'elevation': np.array([[100, 120, 110], [105, 115, 125], [110, 130, 140]]),
        'fuel_type': np.array([[1, 2, 1], [2, 3, 2], [1, 2, 3]]),
        'fire_intensity_path': '/path/to/fire_intensity.tif',
        'firelines_path': '/path/to/firelines.shp',
        'elevation_path': '/path/to/elevation.tif',
        'vegetation_path': '/path/to/vegetation.tif'
    }

    # Create a State object
    state = State(raw_data, data_manager)

    # Test the methods
    print(f"State: {state}")
    print(f"Raw data: {state.get_raw_data()}")
    print(f"Simulation state: {state.get_simulation_state()}")
    print(f"Input paths: {state.get_input_paths()}")
    print(f"Output paths: {state.get_output_paths()}")
    print(f"Fire growth rate: {state.get_fire_growth_rate(3600)}")
    print(f"Resource efficiency: {state.get_resource_efficiency()}")
    print(f"High risk areas: {state.get_high_risk_areas()}")


if __name__ == "__main__":
    main()