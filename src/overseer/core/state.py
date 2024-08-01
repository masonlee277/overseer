# src/overseer/core/state.py

from typing import Dict, Any
import numpy as np
from overseer.data.data_manager import DataManager
from overseer.utils.logging import OverseerLogger

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
    """

    def __init__(self, raw_data: Dict[str, Any], data_manager: DataManager):
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.raw_data = raw_data
        self.data_manager = data_manager
        self.timestamp = raw_data.get('timestamp', 0.0)
        self.basic_stats = self._calculate_basic_stats()
        
        # Add file paths for geospatial data
        # TODO: change where this occors 
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
        """
        Get the fire growth rate over a specified time window.

        This method delegates the calculation to the DataManager.

        Args:
            time_window (float): The time window to calculate the growth rate over.

        Returns:
            float: The fire growth rate in hectares per hour.
        """
        return self.data_manager.calculate_fire_growth_rate(self.raw_data, time_window)

    def get_resource_efficiency(self) -> float:
        """
        Calculate the resource efficiency for this state.

        This method delegates the calculation to the DataManager.

        Returns:
            float: The resource efficiency score.
        """
        return self.data_manager.calculate_resource_efficiency(self.raw_data)

    def get_fire_containment_percentage(self) -> float:
        """
        Calculate the fire containment percentage for this state.

        This method delegates the calculation to the DataManager.

        Returns:
            float: The fire containment percentage.
        """
        return self.data_manager.calculate_fire_containment_percentage(self.raw_data)

    def get_high_risk_areas(self) -> np.ndarray:
        """
        Identify high-risk areas based on the current state.

        This method delegates the calculation to the DataManager.

        Returns:
            np.ndarray: A boolean array indicating high-risk areas.
        """
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