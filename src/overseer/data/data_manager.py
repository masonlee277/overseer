# src/overseer/data/data_manager.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
import json


from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.core.state_manager import StateManager
from overseer.core.state import State
from overseer.data.geospatial_manager import GeoSpatialManager

class DataManager:
    """
    Manages all data operations for ELMFIRE simulations and the RL environment,
    including geospatial data handling.

    This class is responsible for:
    1. Saving and loading simulation states
    2. Managing geospatial data (saving, loading, and processing GeoTIFF files)
    3. Handling RL metrics
    4. Preprocessing data for the RL environment
    5. Managing intermediate states
    6. Aggregating and analyzing simulation results
    7. Coordinating geospatial operations through GeoSpatialManager

    It does NOT:
    1. Run simulations (handled by SimulationManager)
    2. Modify ELMFIRE configurations (handled by ConfigurationManager)
    3. Interface directly with the ELMFIRE simulator
    4. Implement RL algorithms or environments

    The DataManager interfaces with:
    - SimulationManager: Provides data storage and retrieval for simulation states
    - ElmfireGymEnv: Offers data preprocessing for the RL environment
    - ConfigurationManager: May load/save configuration-related data
    - GeoSpatialManager: Handles specific geospatial operations

    Attributes:
        config (OverseerConfig): The OverseerConfig instance.
        logger (logging.Logger): Logger for this class.
        data_dir (Path): Root directory for all data storage.
        current_episode (int): The current episode number.
        current_step (int): The current step number within the episode.
        geospatial_manager (GeoSpatialManager): Manager for geospatial operations.
    """


    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.geospatial_manager = GeoSpatialManager(self.config)
        self.state_manager = StateManager(self.config, self)
        self.current_episode = 0
        self.current_step = 0

    def __getattr__(self, name):
        # Delegate to GeoSpatialManager for undefined attributes
        return getattr(self.geospatial_manager, name)

    def close(self):
        """
        Close any open resources and perform cleanup operations.
        This method should be called when the environment is closed.
        """
        self.logger.info("Closing DataManager and cleaning up resources")

        
        # Perform any final data saving or cleanup
        self.cleanup_old_data(self.config.get('max_episodes_to_keep', 10))
        
        # Close the geospatial manager if it has a close method
        if hasattr(self.geospatial_manager, 'close'):
            self.geospatial_manager.close()
        
        self.logger.info("DataManager resources cleaned up and closed")

    def get_episode_data(self, episode: int) -> List[Dict[str, Any]]:
        """
        Retrieve all data for a specific episode.

        Args:
            episode (int): The episode number to retrieve data for.

        Returns:
            List[Dict[str, Any]]: A list of state dictionaries for the specified episode.
        """
        episode_data = []
        episode_dir = self.data_dir / 'simulations' / 'raw'
        for file in sorted(os.listdir(episode_dir)):
            if file.startswith(f'episode_{episode}_'):
                with open(episode_dir / file, 'r') as f:
                    episode_data.append(json.load(f))
        return episode_data

    def get_rl_metrics(self, episode: int) -> Dict[str, float]:
        """
        Retrieve RL metrics for a specific episode.

        Args:
            episode (int): The episode number to retrieve metrics for.

        Returns:
            Dict[str, float]: A dictionary of RL metrics for the specified episode.
        """
        metrics_file = self.data_dir / 'rl_metrics' / f'episode_{episode}_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        else:
            self.logger.warning(f"No RL metrics found for episode {episode}")
            return {}

    def update_rl_metrics(self, episode: int, step: int, metrics: Dict[str, Any]) -> None:
        """
        Update RL metrics for a specific episode and step.

        Args:
            episode (int): The current episode number.
            step (int): The current step number within the episode.
            metrics (Dict[str, Any]): The metrics to update.
        """
        metrics_dir = self.data_dir / 'rl_metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f'episode_{episode}_metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = {}
        
        existing_metrics[f'step_{step}'] = metrics
        
        with open(metrics_file, 'w') as f:
            json.dump(existing_metrics, f)

    def reset(self) -> None:
        """
        Reset the DataManager to its initial state.
        This method should be called when the environment is reset.
        """
        self.logger.info("Resetting DataManager")
        self.current_step = 0
        self.state_manager.reset()
        # Clear any temporary data or caches if necessary

    # ... (existing code) ...
    def save_simulation_state(self, state: Dict[str, Any]) -> None:
        """Save the current simulation state."""
        self.state_manager.update_state(state)
        self._save_state_to_disk(state)

    def load_simulation_state(self, episode: int, step: int) -> Dict[str, Any]:
        """Load a simulation state from disk."""
        return self._load_state_from_disk(episode, step)

    def get_current_state(self) -> Optional[State]:
        """Get the current state."""
        return self.state_manager.get_current_state()

    def get_state_at_time(self, timestamp: float) -> Optional[State]:
        """Get the state at a specific timestamp."""
        return self.state_manager.get_state_at_time(timestamp)

    def get_fire_growth_rate(self, time_window: float) -> float:
        """Get the fire growth rate over a specified time window."""
        return self.state_manager.get_fire_growth_rate(time_window)

    def get_resource_efficiency(self) -> float:
        """Calculate the resource efficiency for the current state."""
        return self.state_manager.get_resource_efficiency()

    def get_fire_containment_percentage(self) -> float:
        """Calculate the fire containment percentage for the current state."""
        return self.state_manager.get_fire_containment_percentage()

    def get_high_risk_areas(self) -> Optional[np.ndarray]:
        """Identify high-risk areas based on the current state."""
        return self.state_manager.get_high_risk_areas()

    def reset(self) -> None:
        """Reset the state manager."""
        self.state_manager.reset()

    def get_state_history(self) -> List[State]:
        """Get the history of states."""
        return self.state_manager.get_state_history()

    def preprocess_data_for_rl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess the simulation state for use in the RL environment."""
        try:
            processed_state = state.copy()
            
            fire_intensity = state['fire_intensity']
            processed_state['burned_area'] = self.calculate_burned_area(fire_intensity, threshold=0.5)
            processed_state['fire_perimeter'] = self.calculate_fire_perimeter(fire_intensity, threshold=0.5)
            processed_state['fire_spread_direction'] = self.compute_fire_spread_direction(fire_intensity)
            
            if 'elevation' in state:
                processed_state['terrain_effects'] = self.calculate_terrain_effects(state['elevation'], fire_intensity)
            
            if 'fuel_type' in state:
                processed_state['high_risk_areas'] = self.identify_high_risk_areas(fire_intensity, state['elevation'], state['fuel_type'])
            
            self.logger.info("Preprocessed simulation state for RL")
            return processed_state
        except Exception as e:
            self.logger.error(f"Failed to preprocess state for RL: {str(e)}")
            raise

    def calculate_burned_area(self, fire_intensity: np.ndarray, threshold: float) -> float:
        """Calculate the total burned area."""
        return self.geospatial_manager.calculate_burned_area(fire_intensity, threshold)

    def calculate_fire_perimeter(self, fire_intensity: np.ndarray, threshold: float) -> np.ndarray:
        """Calculate the fire perimeter."""
        return self.geospatial_manager.calculate_fire_perimeter(fire_intensity, threshold)

    def compute_fire_spread_direction(self, fire_intensity: np.ndarray) -> np.ndarray:
        """Compute the fire spread direction."""
        return self.geospatial_manager.compute_fire_spread_direction(fire_intensity)

    def calculate_terrain_effects(self, elevation: np.ndarray, fire_intensity: np.ndarray) -> np.ndarray:
        """Calculate terrain effects on fire spread."""
        return self.geospatial_manager.calculate_terrain_effects(elevation, fire_intensity)

    def identify_high_risk_areas(self, fire_intensity: np.ndarray, elevation: np.ndarray, fuel_type: np.ndarray) -> np.ndarray:
        """Identify areas at high risk for fire spread."""
        return self.geospatial_manager.identify_high_risk_areas(fire_intensity, elevation, fuel_type)

    def _save_state_to_disk(self, state: Dict[str, Any]) -> None:
        """Save the state to disk."""
        file_path = self.data_dir / 'simulations' / 'raw' / f'episode_{self.current_episode}_step_{self.current_step}.json'
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(state, f)

    def _load_state_from_disk(self, episode: int, step: int) -> Dict[str, Any]:
        """Load a state from disk."""
        file_path = self.data_dir / 'simulations' / 'raw' / f'episode_{episode}_step_{step}.json'
        with open(file_path, 'r') as f:
            return json.load(f)

    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.current_step += 1

    def increment_episode(self) -> None:
        """Increment the current episode counter and reset the step counter."""
        self.current_episode += 1
        self.current_step = 0

    def get_latest_episode_and_step(self) -> Tuple[int, int]:
        """Get the latest episode and step numbers from stored data."""
        try:
            episodes = [int(f.split('_')[1]) for f in os.listdir(self.data_dir / 'simulations' / 'raw') if f.startswith('episode_')]
            if not episodes:
                return 0, 0
            latest_episode = max(episodes)
            steps = [int(f.split('_')[3].split('.')[0]) for f in os.listdir(self.data_dir / 'simulations' / 'raw') if f.startswith(f'episode_{latest_episode}_')]
            latest_step = max(steps) if steps else 0
            return latest_episode, latest_step
        except Exception as e:
            self.logger.error(f"Failed to get latest episode and step: {str(e)}")
            return 0, 0

    def cleanup_old_data(self, max_episodes: int) -> None:
        """Remove old simulation data to free up storage space."""
        try:
            episodes = sorted([int(f.split('_')[1]) for f in os.listdir(self.data_dir / 'simulations' / 'raw') if f.startswith('episode_')], reverse=True)
            episodes_to_remove = episodes[max_episodes:]
            for episode in episodes_to_remove:
                for file in os.listdir(self.data_dir / 'simulations' / 'raw'):
                    if file.startswith(f'episode_{episode}_'):
                        os.remove(self.data_dir / 'simulations' / 'raw' / file)
            self.logger.info(f"Cleaned up old data, keeping the most recent {max_episodes} episodes")
        except Exception as e:
            self.logger.error(f"Failed to clean up old data: {str(e)}")

    # Add more methods as needed...

    def cleanup_old_data(self, max_episodes: int) -> None:
        """
        Remove old simulation data to free up storage space.

        Args:
            max_episodes (int): Maximum number of recent episodes to keep.
        """
        try:
            episodes = sorted([int(f.split('_')[1]) for f in os.listdir(self.data_dir / 'simulations' / 'raw') if f.startswith('episode_')], reverse=True)
            episodes_to_remove = episodes[max_episodes:]
            for episode in episodes_to_remove:
                for file in os.listdir(self.data_dir / 'simulations' / 'raw'):
                    if file.startswith(f'episode_{episode}_'):
                        os.remove(self.data_dir / 'simulations' / 'raw' / file)
            self.logger.info(f"Cleaned up old data, keeping the most recent {max_episodes} episodes")
        except Exception as e:
            self.logger.error(f"Failed to clean up old data: {str(e)}")

    # Additional geospatial methods delegating to GeoSpatialManager

    def calculate_burned_area(self, fire_intensity: np.ndarray, threshold: float) -> float:
        """
        Calculate the total burned area based on fire intensity.

        Args:
            fire_intensity (np.ndarray): The fire intensity array.
            threshold (float): The threshold for considering an area burned.

        Returns:
            float: The total burned area in hectares.
        """
        return self.geospatial_manager.calculate_burned_area(fire_intensity, threshold)

    def calculate_fire_perimeter(self, fire_intensity: np.ndarray, threshold: float) -> np.ndarray:
        """
        Calculate the fire perimeter based on fire intensity.

        Args:
            fire_intensity (np.ndarray): The fire intensity array.
            threshold (float): The threshold for considering an area part of the fire.

        Returns:
            np.ndarray: A boolean array representing the fire perimeter.
        """
        return self.geospatial_manager.calculate_fire_perimeter(fire_intensity, threshold)

    def compute_fire_spread_direction(self, fire_intensity: np.ndarray) -> np.ndarray:
        """
        Compute the predominant fire spread direction.

        Args:
            fire_intensity (np.ndarray): The fire intensity array.

        Returns:
            np.ndarray: An array representing the fire spread direction.
        """
        return self.geospatial_manager.compute_fire_spread_direction(fire_intensity)

    def calculate_terrain_effects(self, elevation: np.ndarray, fire_intensity: np.ndarray) -> np.ndarray:
        """
        Calculate the effects of terrain on fire spread.

        Args:
            elevation (np.ndarray): The elevation data.
            fire_intensity (np.ndarray): The fire intensity array.

        Returns:
            np.ndarray: An array representing the terrain effects on fire spread.
        """
        return self.geospatial_manager.calculate_terrain_effects(elevation, fire_intensity)

    def identify_high_risk_areas(self, fire_intensity: np.ndarray, elevation: np.ndarray, fuel_type: np.ndarray) -> np.ndarray:
        """
        Identify areas at high risk for fire spread.

        Args:
            fire_intensity (np.ndarray): The fire intensity array.
            elevation (np.ndarray): The elevation data.
            fuel_type (np.ndarray): The fuel type data.

        Returns:
            np.ndarray: A boolean array indicating high-risk areas.
        """
        return self.geospatial_manager.identify_high_risk_areas(fire_intensity, elevation, fuel_type)

    def calculate_fire_containment(self, fire_perimeter: np.ndarray, containment_lines: np.ndarray) -> float:
        """
        Calculate the percentage of fire containment.

        Args:
            fire_perimeter (np.ndarray): The fire perimeter array.
            containment_lines (np.ndarray): The containment lines array.

        Returns:
            float: The percentage of fire containment.
        """
        return self.geospatial_manager.calculate_fire_containment(fire_perimeter, containment_lines)

    def preprocess_data_for_rl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the simulation state for use in the RL environment.

        This method now includes geospatial preprocessing using GeoSpatialManager.

        Args:
            state (Dict[str, Any]): The raw simulation state.

        Returns:
            Dict[str, Any]: The preprocessed state suitable for the RL environment.
        """
        try:
            processed_state = state.copy()
            
            # Use GeoSpatialManager for geospatial calculations
            fire_intensity = state['fire_intensity']
            processed_state['burned_area'] = self.calculate_burned_area(fire_intensity, threshold=0.5)
            processed_state['fire_perimeter'] = self.calculate_fire_perimeter(fire_intensity, threshold=0.5)
            processed_state['fire_spread_direction'] = self.compute_fire_spread_direction(fire_intensity)
            
            if 'elevation' in state:
                processed_state['terrain_effects'] = self.calculate_terrain_effects(state['elevation'], fire_intensity)
            
            if 'fuel_type' in state:
                processed_state['high_risk_areas'] = self.identify_high_risk_areas(fire_intensity, state['elevation'], state['fuel_type'])
            
            # Add more preprocessing steps as needed
            
            self.logger.info("Preprocessed simulation state for RL")
            return processed_state
        except Exception as e:
            self.logger.error(f"Failed to preprocess state for RL: {str(e)}")
            raise
