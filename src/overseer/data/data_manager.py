import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import json
from datetime import datetime

from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.core.models import (
    InputPaths, OutputPaths, SimulationConfig, SimulationState, 
    SimulationResult, Action, EpisodeStep, Episode
)

class DataManager:
    """
    Manages all data operations for ELMFIRE simulations and the RL environment,
    including geospatial data handling and state management.

    This class is responsible for:
    1. Saving and loading simulation states
    2. Managing geospatial data (saving, loading, and processing GeoTIFF files)
    3. Handling RL metrics
    4. Preprocessing data for the RL environment
    5. Managing intermediate states
    6. Aggregating and analyzing simulation results
    7. Coordinating geospatial operations through GeoSpatialManager
    8. Managing the current state and history of states (previously StateManager functionality)

    Attributes:
        config (OverseerConfig): The OverseerConfig instance.
        logger (logging.Logger): Logger for this class.
        data_dir (Path): Root directory for all data storage.
        current_episode (int): The current episode number.
        current_step (int): The current step number within the episode.
        geospatial_manager (GeoSpatialManager): Manager for geospatial operations.
        current_state (Optional[SimulationState]): The current state of the simulation.
        state_history (List[SimulationState]): History of states for the current episode.
    """

    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.geospatial_manager = GeoSpatialManager(self.config)
        self.current_episode = 0
        self.current_step = 0
        self.current_state: Optional[SimulationState] = None
        self.state_history: List[SimulationState] = []
        self.rl_metrics: Dict[int, List[Dict[str, float]]] = {}

    def update_state(self, new_data: Dict[str, Any]) -> None:

        ## TODO: simulation state needs to be like high level for the fire perims
        
        """
        Update the current state with new data.

        Args:
            new_data (Dict[str, Any]): New state data to incorporate.
        """
        self.logger.info("Updating state")
        new_state = SimulationState(
            timestamp=datetime.now(),
            fire_intensity=new_data.get('fire_intensity', np.array([])),
            burned_area=self.calculate_burned_area(new_data.get('fire_intensity', np.array([])), threshold=0.5),
            fire_perimeter_length=self.calculate_fire_perimeter_length(new_data.get('fire_intensity', np.array([])), threshold=0.5),
            containment_percentage=self.calculate_fire_containment_percentage(new_data),
            resources=new_data.get('resources', {}),
            weather=new_data.get('weather', {})
        )
        self.current_state = new_state
        self.state_history.append(new_state)
        self._save_state_to_disk(new_data)

    def get_current_state(self) -> Optional[SimulationState]:
        """
        Get the current state.

        Returns:
            Optional[SimulationState]: The current state, or None if not initialized.
        """
        return self.current_state

    def get_state_at_time(self, timestamp: float) -> Optional[SimulationState]:
        """
        Get the state at a specific timestamp.

        Args:
            timestamp (float): The timestamp to retrieve the state for.

        Returns:
            Optional[SimulationState]: The state at the given timestamp, or None if not found.
        """
        state_data = self._load_state_from_disk(timestamp)
        if state_data:
            return SimulationState(
                timestamp=datetime.fromtimestamp(timestamp),
                fire_intensity=state_data.get('fire_intensity', np.array([])),
                burned_area=self.calculate_burned_area(state_data.get('fire_intensity', np.array([])), threshold=0.5),
                fire_perimeter_length=self.calculate_fire_perimeter_length(state_data.get('fire_intensity', np.array([])), threshold=0.5),
                containment_percentage=self.calculate_fire_containment_percentage(state_data),
                resources=state_data.get('resources', {}),
                weather=state_data.get('weather', {})
            )
        return None

    def get_fire_growth_rate(self, time_window: float) -> float:
        """
        Get the fire growth rate over a specified time window.

        Args:
            time_window (float): The time window to calculate the growth rate over.

        Returns:
            float: The fire growth rate in hectares per hour.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return 0.0
        
        past_state = self.get_state_at_time(self.current_state.timestamp.timestamp() - time_window)
        if past_state is None:
            return 0.0
        
        area_difference = self.current_state.burned_area - past_state.burned_area
        time_difference = (self.current_state.timestamp - past_state.timestamp).total_seconds() / 3600  # Convert to hours
        return area_difference / time_difference if time_difference > 0 else 0.0

    def get_resource_efficiency(self) -> float:
        """
        Get the resource efficiency for the current state.

        Returns:
            float: The resource efficiency score.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return 0.0
        
        # Implement resource efficiency calculation logic here
        # This is a placeholder implementation
        total_resources = sum(self.current_state.resources.values())
        return self.current_state.containment_percentage / total_resources if total_resources > 0 else 0.0

    def get_fire_containment_percentage(self) -> float:
        """
        Get the fire containment percentage for the current state.

        Returns:
            float: The fire containment percentage.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return 0.0
        return self.current_state.containment_percentage

    def get_high_risk_areas(self) -> Optional[np.ndarray]:
        """
        Get high-risk areas based on the current state.

        Returns:
            Optional[np.ndarray]: A boolean array indicating high-risk areas, or None if state is not initialized.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return None
        
        # Implement high-risk areas identification logic here
        # This is a placeholder implementation
        return self.geospatial_manager.identify_high_risk_areas(
            self.current_state.fire_intensity,
            np.array([]),  # Placeholder for elevation data
            np.array([])   # Placeholder for fuel type data
        )

    def reset(self) -> None:
        """
        Reset the data manager for a new episode.
        """
        self.logger.info("Resetting DataManager")
        self.current_state = None
        self.state_history.clear()
        self.current_episode += 1
        self.current_step = 0

    def get_state_history(self) -> List[SimulationState]:
        """
        Get the history of states for the current episode.

        Returns:
            List[SimulationState]: The history of states.
        """
        return self.state_history

    def _save_state_to_disk(self, state: Dict[str, Any]) -> None:
        """Save the state to disk."""
        file_path = self.data_dir / 'simulations' / 'raw' / f'episode_{self.current_episode}_step_{self.current_step}.json'
        with open(file_path, 'w') as f:
            json.dump(state, f)

    def _load_state_from_disk(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Load a state from disk based on timestamp."""
        for file in os.listdir(self.data_dir / 'simulations' / 'raw'):
            if file.startswith('episode_'):
                file_path = self.data_dir / 'simulations' / 'raw' / file
                #with open(file_path, 'r') as f:
        return self.geospatial_manager.calculate_fire_perimeter(fire_intensity, threshold)



    def generate_action_mask_from_episode(self, episode_step: EpisodeStep) -> np.ndarray:
        """
        Generate an action mask based on the simulation state contained in the EpisodeStep.

        Args:
            episode_step (EpisodeStep): The current episode step containing the simulation state.

        Returns:
            np.ndarray: Boolean mask where True indicates a valid action.
        """
        # Extract file paths from the episode step
        fire_intensity_path = episode_step.simulation_result.output_files['time_of_arrival']
        existing_firelines_path = episode_step.simulation_result.output_files['fire_intensity']
        elevation_path = episode_step.input_paths.dem_filename  # Using dem as elevation
        vegetation_path = episode_step.input_paths.cc_filename  # Using cc as vegetation

        # Generate action mask using file paths
        return self.geospatial_manager.generate_action_mask_from_files(
            fire_intensity_path=fire_intensity_path,
            existing_firelines_path=existing_firelines_path,
            elevation_path=elevation_path,
            vegetation_path=vegetation_path,
            min_distance=1,
            max_distance=10,
            max_slope=None,  # You can set this based on your requirements
            constructable_veg_types=None,  # You can set this based on your requirements
            min_effective_length=None  # You can set this based on your requirements
        )

    def get_existing_firelines(self, state: SimulationState) -> np.ndarray:
        # Implement this method to extract existing firelines from the state
        pass

    def get_elevation_data(self, state: SimulationState) -> np.ndarray:
        # Implement this method to extract elevation data from the state
        pass

    def get_vegetation_data(self, state: SimulationState) -> np.ndarray:
        # Implement this method to extract vegetation data from the state
        pass

    def calculate_fire_distance(self, fire_intensity: np.ndarray) -> np.ndarray:
        return self.geospatial_manager.calculate_fire_distance(fire_intensity)

    def calculate_slope(self, elevation: np.ndarray) -> np.ndarray:
        return self.geospatial_manager.calculate_slope(elevation)
    
    def check_action_constraints(self, state: SimulationState, fireline_coords: np.ndarray,
                                 max_fireline_distance: float, max_construction_slope: float,
                                 constructable_veg_types: List[int], min_effective_length: int) -> bool:
        """Check if the given action satisfies all constraints."""
        fire_intensity = state.fire_intensity
        existing_firelines = self.get_existing_firelines(state)
        elevation = self.get_elevation_data(state)
        vegetation = self.get_vegetation_data(state)
        fire_distance = self.calculate_fire_distance(fire_intensity)
        slope = self.calculate_slope(elevation)

        return self.geospatial_manager.check_fireline_constraints(
            fireline_coords, fire_intensity, existing_firelines, elevation, vegetation,
            fire_distance, slope, max_fireline_distance, max_construction_slope,
            constructable_veg_types, min_effective_length
        )


    def compute_fire_spread_direction(self, fire_intensity: np.ndarray) -> np.ndarray:
        """Compute the fire spread direction."""
        return self.geospatial_manager.compute_fire_spread_direction(fire_intensity)

    def calculate_terrain_effects(self, elevation: np.ndarray, fire_intensity: np.ndarray) -> np.ndarray:
        """Calculate terrain effects on fire spread."""
        return self.geospatial_manager.calculate_terrain_effects(elevation, fire_intensity)

    def identify_high_risk_areas(self, fire_intensity: np.ndarray, elevation: np.ndarray, fuel_type: np.ndarray) -> np.ndarray:
        """Identify areas at high risk for fire spread."""
        return self.geospatial_manager.identify_high_risk_areas(fire_intensity, elevation, fuel_type)

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

    def preprocess_data_for_rl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the simulation state for use in the RL environment.

        This method includes geospatial preprocessing using GeoSpatialManager.

        Args:
            state (Dict[str, Any]): The raw simulation state.

        Returns:
            Dict[str, Any]: The preprocessed state suitable for the RL environment.
        """
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

    def save_episode(self, episode: Episode) -> None:
        """
        Save an entire episode to disk.

        Args:
            episode (Episode): The episode to save.
        """
        episode_dir = self.data_dir / 'episodes' / f'episode_{episode.episode_id}'
        episode_dir.mkdir(parents=True, exist_ok=True)

        # Save episode metadata
        metadata = {
            'episode_id': episode.episode_id,
            'total_reward': episode.total_reward,
            'total_steps': episode.total_steps,
            'final_burned_area': episode.final_burned_area,
            'final_containment_percentage': episode.final_containment_percentage,
            'execution_time': episode.execution_time
        }
        with open(episode_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

        # Save each step
        for i, step in enumerate(episode.steps):
            step_data = {
                'step': step.step,
                'state': step.state.__dict__,
                'action': step.action.__dict__,
                'reward': step.reward,
                'next_state': step.next_state.__dict__,
                'simulation_result': step.simulation_result.__dict__,
                'done': step.done
            }
            with open(episode_dir / f'step_{i}.json', 'w') as f:
                json.dump(step_data, f)

        self.logger.info(f"Saved episode {episode.episode_id} to disk")

    def load_episode(self, episode_id: int) -> Optional[Episode]:
        """
        Load an entire episode from disk.

        Args:
            episode_id (int): The ID of the episode to load.

        Returns:
            Optional[Episode]: The loaded episode, or None if not found.
        """
        episode_dir = self.data_dir / 'episodes' / f'episode_{episode_id}'
        if not episode_dir.exists():
            self.logger.warning(f"Episode {episode_id} not found on disk")
            return None

        try:
            # Load episode metadata
            with open(episode_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            metadata = json.load(f)

            # Load steps
            steps = []
            step_files = sorted([f for f in os.listdir(episode_dir) if f.startswith('step_')])
            for step_file in step_files:
                with open(episode_dir / step_file, 'r') as f:
                    step_data = json.load(f)
                steps.append(EpisodeStep(
                    step=step_data['step'],
                    state=SimulationState(**step_data['state']),
                    action=Action(**step_data['action']),
                    reward=step_data['reward'],
                    next_state=SimulationState(**step_data['next_state']),
                    simulation_result=SimulationResult(**step_data['simulation_result']),
                    done=step_data['done']
                ))

            episode = Episode(
                episode_id=metadata['episode_id'],
                steps=steps,
                total_reward=metadata['total_reward'],
                total_steps=metadata['total_steps'],
                final_burned_area=metadata['final_burned_area'],
                final_containment_percentage=metadata['final_containment_percentage'],
                execution_time=metadata['execution_time']
            )

            self.logger.info(f"Loaded episode {episode_id} from disk")
            return episode
        except Exception as e:
            self.logger.error(f"Failed to load episode {episode_id}: {str(e)}")
            return None

    def _save_state_to_disk(self, state_data: Dict[str, Any]) -> None:
        """
        Save the current state to disk.

        Args:
            state_data (Dict[str, Any]): The state data to save.
        """
        state_dir = self.data_dir / 'states'
        state_dir.mkdir(parents=True, exist_ok=True)
        timestamp = state_data.get('timestamp', datetime.now().timestamp())
        with open(state_dir / f'state_{timestamp}.json', 'w') as f:
            json.dump(state_data, f)

    def _load_state_from_disk(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Load a state from disk.

        Args:
            timestamp (float): The timestamp of the state to load.

        Returns:
            Optional[Dict[str, Any]]: The loaded state data, or None if not found.
        """
        state_dir = self.data_dir / 'states'
        state_file = state_dir / f'state_{timestamp}.json'
        if not state_file.exists():
            return None
        with open(state_file, 'r') as f:
            return json.load(f)

    def get_resource_efficiency(self) -> float:
        """
        Calculate the resource efficiency based on the current state.

        Returns:
            float: The resource efficiency score (0.0 to 1.0).
        """
        if self.current_state is None:
            return 0.0

        # This is a simplified calculation and should be adjusted based on your specific requirements
        total_resources = sum(self.current_state.resources.values())
        if total_resources == 0:
            return 0.0

        containment_per_resource = self.current_state.containment_percentage / total_resources
        return min(containment_per_resource / 100, 1.0)  # Normalize to 0.0-1.0 range

    def get_high_risk_areas(self) -> Optional[np.ndarray]:
        """
        Identify high-risk areas based on the current state.

        Returns:
            Optional[np.ndarray]: A boolean array indicating high-risk areas, or None if not available.
        """
        if self.current_state is None:
            return None

        return self.identify_high_risk_areas(
            self.current_state.fire_intensity,
            self.current_state.elevation if hasattr(self.current_state, 'elevation') else np.array([]),
            self.current_state.fuel_type if hasattr(self.current_state, 'fuel_type') else np.array([])
        )

    def reset(self) -> None:
        """Reset the state manager."""
        self.current_state = None
        self.state_history = []
        self.current_episode += 1
        self.current_step = 0

    def get_state_history(self) -> List[SimulationState]:
        """Get the history of states."""
        return self.state_history
    
    def save_simulation_state(self, simulation_result: SimulationResult) -> None:
        """
        Save the simulation state and results.

        Args:
            simulation_result (SimulationResult): The simulation result to save.
        """
        try:
            self.logger.info(f"Saving simulation state for episode {self.current_episode}, step {self.current_step}")
            
            # Update the current state
            self.current_state = simulation_result.final_state
            self.state_history.append(self.current_state)

            # Save the simulation result
            result_dir = self.data_dir / 'simulations' / 'results'
            result_dir.mkdir(parents=True, exist_ok=True)
            result_file = result_dir / f'episode_{self.current_episode}_step_{self.current_step}.json'
            
            with open(result_file, 'w') as f:
                json.dump({
                    'final_state': self.current_state.__dict__,
                    'performance_metrics': simulation_result.performance_metrics,
                    'output_files': simulation_result.output_files,
                    'execution_time': simulation_result.execution_time,
                    'metadata': simulation_result.metadata
                }, f, default=self._json_serializer)

            self.logger.info(f"Simulation state saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save simulation state: {str(e)}")
            raise

    def update_rl_metrics(self, episode: int, step: int, metrics: Dict[str, float]) -> None:
        """
        Update the RL metrics for a given episode and step.

        Args:
            episode (int): The current episode number.
            step (int): The current step number within the episode.
            metrics (Dict[str, float]): The RL metrics to update.
        """
        try:
            self.logger.info(f"Updating RL metrics for episode {episode}, step {step}")
            
            if episode not in self.rl_metrics:
                self.rl_metrics[episode] = []
            
            self.rl_metrics[episode].append({
                'step': step,
                **metrics
            })

            # Save the updated metrics to disk
            metrics_dir = self.data_dir / 'rl_metrics'
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / f'episode_{episode}.json'
            
            with open(metrics_file, 'w') as f:
                json.dump(self.rl_metrics[episode], f)

            self.logger.info(f"RL metrics updated successfully")
        except Exception as e:
            self.logger.error(f"Failed to update RL metrics: {str(e)}")
            raise

    def _json_serializer(self, obj):
        """Custom JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, (datetime, np.ndarray)):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
def main():
    # This main function can be used for testing the DataManager
    config = OverseerConfig()  # Assuming you have a way to create this
    data_manager = DataManager(config)

    # Test updating state
    print("Testing state update:")
    initial_state = {
        'timestamp': datetime.now().timestamp(),
        'fire_intensity': np.array([[0, 0, 1], [0, 1, 2], [1, 2, 3]]),
        'resources': {'firefighters': 20, 'trucks': 5},
        'weather': {'wind_speed': 10.0, 'wind_direction': 180.0},
    }
    data_manager.update_state(initial_state)
    print(f"Current state after update: {data_manager.get_current_state()}")

    # Test getting current state
    print("\nTesting get_current_state:")
    current_state = data_manager.get_current_state()
    print(f"Current state: {current_state}")

    # Test getting fire growth rate
    print("\nTesting get_fire_growth_rate:")
    growth_rate = data_manager.get_fire_growth_rate(3600)
    print(f"Fire growth rate: {growth_rate}")

    # Test getting resource efficiency
    print("\nTesting get_resource_efficiency:")
    efficiency = data_manager.get_resource_efficiency()
    print(f"Resource efficiency: {efficiency}")

    # Test getting high risk areas
    print("\nTesting get_high_risk_areas:")
    high_risk_areas = data_manager.get_high_risk_areas()
    print(f"High risk areas: {high_risk_areas}")

    # Test resetting data manager
    print("\nTesting reset:")
    data_manager.reset()
    print(f"Data manager after reset: {data_manager.get_current_state()}")

    # Test getting state history
    print("\nTesting get_state_history:")
    data_manager.update_state(initial_state)
    data_manager.update_state({**initial_state, 'timestamp': datetime.now().timestamp()})
    history = data_manager.get_state_history()
    print(f"State history length: {len(history)}")
    print(f"State history: {history}")



if __name__ == "__main__":
    main()