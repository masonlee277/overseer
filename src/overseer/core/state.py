# src/overseer/core/state.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any
import numpy as np
from overseer.data.data_manager import DataManager
from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig

class State:
    """
    Represents a single state in the ELMFIRE simulation.

    This class is primarily a data container, storing raw state data and basic statistics.
    It delegates complex calculations to the DataManager.

    Attributes:
        raw_data (Dict[str, Any]): The raw state data from the simulation.
        data_manager (DataManager): Reference to the DataManager for complex calculations.
        basic_stats (Dict[str, Any]): Pre-calculated basic statistics for quick access.
        timestamp (float): The timestamp of this state.

    Example of a state's raw_data:
    {
        'timestamp': 3600.0,  # Seconds since simulation start
        'fire_intensity': np.array([[0, 0, 1, 2], [0, 1, 3, 2], [0, 0, 1, 1]]),  # 2D array of fire intensity values
        'resources_deployed': {'firefighters': 50, 'trucks': 10, 'aircraft': 2},
        'wind_speed': 15.5,  # m/s
        'wind_direction': 225.0,  # degrees (0-360)
        'temperature': 30.5,  # Celsius
        'humidity': 20.0,  # Percentage
        'fuel_moisture': np.array([[10, 12, 8], [9, 11, 10], [8, 9, 11]]),  # 2D array of fuel moisture percentages
        'elevation': np.array([[100, 120, 110], [105, 115, 125], [110, 130, 140]]),  # 2D array of elevation in meters
        'fuel_type': np.array([[1, 2, 1], [2, 3, 2], [1, 2, 3]]),  # 2D array of fuel type codes
        'fire_intensity_path': '/path/to/fire_intensity.tif',
        'firelines_path': '/path/to/firelines.shp',
        'elevation_path': '/path/to/elevation.tif',
        'vegetation_path': '/path/to/vegetation.tif'
    }

    Example of basic_stats:
    {
        'burned_area': 1250.5,  # hectares
        'fire_perimeter_length': 15.2,  # kilometers
        'total_resources_deployed': 62,
        'max_fire_intensity': 3,
        'mean_fire_intensity': 1.375
    }
    """

    def __init__(self, raw_data: Dict[str, Any], data_manager: DataManager):
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.raw_data = raw_data
        self.data_manager = data_manager
        self.timestamp = raw_data.get('timestamp', 0.0)
        self.basic_stats = self._calculate_basic_stats()
        
        self.geospatial_data_paths = {
            'fire_intensity': raw_data.get('fire_intensity_path'),
            'firelines': raw_data.get('firelines_path'),
            'elevation': raw_data.get('elevation_path'),
            'vegetation': raw_data.get('vegetation_path'),
        }

    def _calculate_basic_stats(self) -> Dict[str, Any]:
        """Calculate and return basic statistics from the raw state data."""
        self.logger.info(f"Calculating basic stats for state at timestamp {self.timestamp}")
        return {
            'burned_area': self.data_manager.calculate_burned_area(self.raw_data['fire_intensity']),
            'fire_perimeter_length': self.data_manager.calculate_fire_perimeter_length(self.raw_data['fire_intensity']),
            'total_resources_deployed': sum(self.raw_data.get('resources_deployed', {}).values()),
            'max_fire_intensity': np.max(self.raw_data['fire_intensity']),
            'mean_fire_intensity': np.mean(self.raw_data['fire_intensity']),
        }

    def get_raw_data(self) -> Dict[str, Any]:
        """Return the raw state data."""
        return self.raw_data

    def get_geospatial_data_paths(self) -> Dict[str, str]:
        """Return the file paths for geospatial data."""
        return self.geospatial_data_paths
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Return the pre-calculated basic statistics."""
        return self.basic_stats

    def get_fire_growth_rate(self, time_window: float) -> float:
        """Get the fire growth rate over a specified time window."""
        return self.data_manager.calculate_fire_growth_rate(self.raw_data, time_window)

    def get_resource_efficiency(self) -> float:
        """Calculate the resource efficiency for this state."""
        return self.data_manager.calculate_resource_efficiency(self.raw_data)

    def get_fire_containment_percentage(self) -> float:
        """Calculate the fire containment percentage for this state."""
        return self.data_manager.calculate_fire_containment_percentage(self.raw_data)

    def get_high_risk_areas(self) -> np.ndarray:
        """Identify high-risk areas based on the current state."""
        return self.data_manager.identify_high_risk_areas(
            self.raw_data['fire_intensity'],
            self.raw_data.get('elevation', None),
            self.raw_data.get('fuel_type', None)
        )

    def __str__(self) -> str:
        """Return a string representation of the state."""
        return f"State(timestamp={self.timestamp}, burned_area={self.basic_stats['burned_area']:.2f})"

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
    print(f"Geospatial data paths: {state.get_geospatial_data_paths()}")
    print(f"Basic stats: {state.get_basic_stats()}")
    print(f"Fire growth rate: {state.get_fire_growth_rate(3600)}")
    print(f"Resource efficiency: {state.get_resource_efficiency()}")
    print(f"Fire containment percentage: {state.get_fire_containment_percentage()}")
    print(f"High risk areas: {state.get_high_risk_areas()}")


if __name__ == "__main__":
    main()