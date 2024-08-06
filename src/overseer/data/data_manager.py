import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Any
import numpy as np
import json
from datetime import datetime

from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.data.state_manager import StateManager
from overseer.core.models import (
    SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics,
    InputPaths, OutputPaths, Action, EpisodeStep, Episode
)

class DataManager:
    """
    Manages all data operations for ELMFIRE simulations and the RL environment,
    including geospatial data handling and state management.

    This class is responsible for:
    1. Coordinating between StateManager and GeoSpatialManager
    2. Handling RL metrics
    3. Preprocessing data for the RL environment
    4. Aggregating and analyzing simulation results

    Attributes:
        config (OverseerConfig): The OverseerConfig instance.
        logger (logging.Logger): Logger for this class.
        data_dir (Path): Root directory for all data storage.
        geospatial_manager (GeoSpatialManager): Manager for geospatial operations.
        state_manager (StateManager): Manager for simulation states and episodes.
    """

    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.geospatial_manager = GeoSpatialManager(self.config)
        self.state_manager = StateManager(self.config)
        self.rl_metrics: Dict[int, List[Dict[str, float]]] = {}

    def update_state(self, state: SimulationState) -> None:
        self.logger.info(f"Updating state at timestamp: {state.timestamp}")
        self.state_manager.update_state(state)

    def get_current_state(self) -> Optional[SimulationState]:
        state = self.state_manager.get_current_state()
        self.logger.info(f"Retrieved current state: {'None' if state is None else state.timestamp}")
        return state

    def get_state_history(self) -> List[SimulationState]:
        """Get the state history from the StateManager."""
        return self.state_manager.get_state_history()

    def get_state_by_episode_step(self, episode_id: int, step: int) -> Optional[SimulationState]:
        """Get a specific state by episode ID and step number."""
        return self.state_manager.get_state_by_episode_step(episode_id, step)

    def start_new_episode(self) -> None:
        """Start a new episode using the StateManager."""
        self.state_manager.start_new_episode()

    def add_step_to_current_episode(self, state: SimulationState, action: Action, reward: float, next_state: SimulationState, done: bool) -> None:
        """Add a new step to the current episode using the StateManager."""
        self.state_manager.add_step_to_current_episode(state, action, reward, next_state, done)

    def get_current_episode(self) -> Optional[Episode]:
        """Get the current episode from the StateManager."""
        return self.state_manager.get_current_episode()

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """Get a specific episode by ID from the StateManager."""
        return self.state_manager.get_episode(episode_id)

    def get_fire_growth_rate(self, time_interval: float) -> float:
        """Calculate the fire growth rate."""
        current_state = self.get_current_state()
        if current_state is None:
            return 0.0
        return self.geospatial_manager.calculate_fire_growth_rate(current_state.metrics.fire_intensity, time_interval)

    def get_resource_efficiency(self) -> float:
        """Calculate the resource efficiency based on the current state."""
        current_state = self.get_current_state()
        if current_state is None:
            return 0.0
        total_resources = sum(current_state.resources.values())
        containment_percentage = current_state.metrics.containment_percentage
        return containment_percentage / total_resources if total_resources > 0 else 0.0

    def get_high_risk_areas(self) -> Optional[np.ndarray]:
        """Identify high-risk areas based on the current state."""
        current_state = self.get_current_state()
        if current_state is None:
            return None
        return self.geospatial_manager.identify_high_risk_areas(
            current_state.metrics.fire_intensity,
            current_state.paths.input_paths.elevation,
            current_state.paths.input_paths.fuel_moisture
        )

    def reset(self) -> None:
        """Reset the state manager."""
        self.state_manager.reset()

    def cleanup_old_data(self, max_episodes: int) -> None:
        """Remove old simulation data to free up storage space."""
        self.state_manager.cleanup_old_episodes(max_episodes)

    def save_state_to_disk(self, state: SimulationState) -> None:
        self.logger.info(f"Saving state to disk at timestamp: {state.timestamp}")
        save_filepath = self.state_manager._save_state_to_disk(state)
        self.logger.info(f"State saved to filepath: {save_filepath}")

    def load_state_from_disk(self, timestamp: str) -> Optional[SimulationState]:
        """Load a state from disk."""
        self.logger.info(f"Loading state from disk for timestamp {timestamp}")
        self.logger.info(f"State manager: {self.state_manager}")
        state = self.state_manager.load_state_from_disk(timestamp)
        if state:
            self.logger.info(f"Successfully loaded state from disk for timestamp {timestamp}")
            self.logger.debug(f"Loaded state details: timestamp={state.timestamp}, "
                              f"config sections={list(state.config.sections.keys())}, "
                              f"metrics={state.metrics}")
        else:
            self.logger.warning(f"Failed to load state from disk for timestamp {timestamp}")
        return state

    #########################################################################
    def get_episode_summary(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Get a summary of an episode."""
        return self.state_manager.get_episode_summary(episode_id)

    def get_all_episode_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries for all episodes."""
        return self.state_manager.get_all_episode_summaries()

    def export_episode_data(self, episode_id: int, export_path: Path) -> None:
        """Export all data for a specific episode to a file."""
        self.state_manager.export_episode_data(episode_id, export_path)

    def import_episode_data(self, import_path: Path) -> Optional[int]:
        """Import episode data from a file."""
        return self.state_manager.import_episode_data(import_path)

    def get_episode_statistics(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Calculate various statistics for an episode."""
        return self.state_manager.get_episode_statistics(episode_id)



    ####################################################################################33

    def update_rl_metrics(self, episode: int, step: int, metrics: Dict[str, float]) -> None:
        """Update RL metrics for a given episode and step."""
        self.logger.info(f"Updating RL metrics for episode {episode}, step {step}")
        if episode not in self.rl_metrics:
            self.rl_metrics[episode] = []
        self.rl_metrics[episode].append({'step': step, **metrics})
        self._save_rl_metrics_to_disk(episode)

    def get_rl_metrics(self, episode: int) -> List[Dict[str, float]]:
        """Get RL metrics for a specific episode."""
        return self.rl_metrics.get(episode, [])

    def _save_rl_metrics_to_disk(self, episode: int) -> None:
        """Save RL metrics for an episode to disk."""
        metrics_dir = self.data_dir / 'rl_metrics'
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_dir / f'episode_{episode}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.rl_metrics[episode], f)


    def _create_simulation_state(self, state_data: Dict) -> SimulationState:
        """Create a SimulationState object from a dictionary of state data."""
        config = SimulationConfig()
        for section, params in state_data.get('config', {}).items():
            for key, value in params.items():
                config.set_parameter(section, key, value)

        return SimulationState(
            timestamp=datetime.now(),
            config=config,
            paths=SimulationPaths(
                input_paths=InputPaths(**state_data.get('input_paths', {})),
                output_paths=OutputPaths(**state_data.get('output_paths', {}))
            ),
            metrics=SimulationMetrics(**state_data.get('metrics', {})),
            save_path=Path(state_data.get('save_path', '')),
            resources=state_data.get('resources', {}),
            weather=state_data.get('weather', {})
        )
    
    def clean_states(self):
        """Clean up the states directory."""
        self.state_manager.clean_states()

def main():
    # This main function can be used for testing the DataManager
    config = OverseerConfig()  # Assuming you have a way to create this
    data_manager = DataManager(config)

    # Create a fake SimulationState
    fake_state = SimulationState(
        timestamp=datetime.now(),
        config=SimulationConfig(sections={
            'INPUTS': {
                'FUELS_AND_TOPOGRAPHY_DIRECTORY': '/path/to/fuels',
                'WEATHER_DIRECTORY': '/path/to/weather'
            },
            'OUTPUTS': {
                'OUTPUTS_DIRECTORY': '/path/to/outputs'
            },
            'TIME_CONTROL': {
                'SIMULATION_TSTOP': '3600.0'
            }
        }),
        paths=SimulationPaths(
            input_paths=InputPaths(
                fuels_and_topography_directory=Path('/path/to/fuels'),
                asp_filename=Path('/path/to/fuels/asp.tif'),
                cbd_filename=Path('/path/to/fuels/cbd.tif'),
                cbh_filename=Path('/path/to/fuels/cbh.tif'),
                cc_filename=Path('/path/to/fuels/cc.tif'),
                ch_filename=Path('/path/to/fuels/ch.tif'),
                dem_filename=Path('/path/to/fuels/dem.tif'),
                fbfm_filename=Path('/path/to/fuels/fbfm.tif'),
                slp_filename=Path('/path/to/fuels/slp.tif'),
                adj_filename=Path('/path/to/fuels/adj.tif'),
                phi_filename=Path('/path/to/fuels/phi.tif'),
                weather_directory=Path('/path/to/weather'),
                ws_filename=Path('/path/to/weather/ws.tif'),
                wd_filename=Path('/path/to/weather/wd.tif'),
                m1_filename=Path('/path/to/weather/m1.tif'),
                m10_filename=Path('/path/to/weather/m10.tif'),
                m100_filename=Path('/path/to/weather/m100.tif'),
                fire=Path('/path/to/fire.shp'),
                vegetation=Path('/path/to/vegetation.tif'),
                elevation=Path('/path/to/elevation.tif'),
                wind=Path('/path/to/wind.tif'),
                fuel_moisture=Path('/path/to/fuel_moisture.tif')
            ),
            output_paths=OutputPaths(
                time_of_arrival=Path('/path/to/outputs/time_of_arrival.tif'),
                fire_intensity=Path('/path/to/outputs/fire_intensity.tif'),
                flame_length=Path('/path/to/outputs/flame_length.tif'),
                spread_rate=Path('/path/to/outputs/spread_rate.tif')
            )
        ),
        metrics=SimulationMetrics(
            burned_area=1000.0,
            fire_perimeter_length=500.0,
            containment_percentage=30.0,
            execution_time=120.0,
            performance_metrics={'cpu_usage': 80.0, 'memory_usage': 4000.0},
            fire_intensity=np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]])
        ),
        save_path=Path('/path/to/save'),
        resources={'firefighters': 20, 'trucks': 5},
        weather={'wind_speed': 10.0, 'wind_direction': 180.0}
    )

    # Test updating state
    print("Testing state update:")
    data_manager.update_state(fake_state)
    print(f"Current state after update: {data_manager.get_current_state()}")

    # Test getting current state
    print("\nTesting get_current_state:")
    current_state = data_manager.get_current_state()
    print(f"Current state: {current_state}")

    # Test resetting data manager
    print("\nTesting reset:")
    data_manager.reset()
    print(f"Data manager after reset: {data_manager.get_current_state()}")

    # Test getting state history
    print("\nTesting get_state_history:")
    data_manager.update_state(fake_state)
    data_manager.update_state(SimulationState(**{**fake_state.__dict__, 'timestamp': datetime.now()}))
    history = data_manager.get_state_history()
    print(f"State history length: {len(history)}")
    print(f"State history: {history}")

    # Test episode management
    print("\nTesting episode management:")
    data_manager.start_new_episode()
    action = Action(fireline_coordinates=[(1, 1), (2, 2), (3, 3)])
    data_manager.add_step_to_current_episode(fake_state, action, 1.0, fake_state, False)
    current_episode = data_manager.get_current_episode()
    print(f"Current episode: {current_episode}")

    # Test RL metrics
    print("\nTesting RL metrics:")
    data_manager.update_rl_metrics(1, 1, {'reward': 1.0, 'loss': 0.5})
    rl_metrics = data_manager.get_rl_metrics(1)
    print(f"RL metrics: {rl_metrics}")

if __name__ == "__main__":
    main()