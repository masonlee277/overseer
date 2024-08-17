import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Any
import numpy as np
import json
from datetime import datetime
import threading
from pathlib import Path
import traceback

from overseer.utils.logging import OverseerLogger
from overseer.utils import fix_path

from overseer.config.config import OverseerConfig
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.data.state_manager import StateManager
from overseer.core.models import (
    SimulationResult, SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics,
    InputPaths, OutputPaths, Action, EpisodeStep, Episode
)

class DataManager:
    """
    DataManager: Central hub for managing simulation data, states, and metrics in the Overseer system.

    Responsibilities:
    1. Coordinate data flow between StateManager and GeoSpatialManager
    2. Handle RL metrics collection and storage
    3. Preprocess and aggregate data for the RL environment
    4. Manage simulation states, episodes, and their persistence
    5. Provide high-level data analysis and retrieval methods

    Key Data Structures:
    1. SimulationState: Comprehensive snapshot of the simulation at a given time
       - timestamp: DateTime of the state
       - config: SimulationConfig object with ELMFIRE parameters
       - paths: SimulationPaths object with input/output file locations
       - metrics: SimulationMetrics object with performance and fire behavior data
       - resources: Dict of available firefighting resources
       - weather: Dict of current weather conditions
    2. Episode: Sequence of EpisodeSteps representing a complete simulation run
       - id: Unique identifier for the episode
       - steps: List of EpisodeStep objects
    3. EpisodeStep: Single step within an episode
       - state: SimulationState at this step
       - action: Action taken at this step
       - reward: Reward received for the action
       - next_state: Resulting SimulationState after the action
       - done: Boolean indicating if the episode is complete
    4. RL Metrics: Dict[int, List[Dict[str, float]]] storing metrics per episode and step

    Scope:
    The DataManager serves as an abstraction layer between the simulation logic and data storage/retrieval.
    It does not directly modify simulation states or apply actions; instead, it manages the flow of data
    and provides an interface for other components to access and update simulation information.

    Differences from ConfigManager:
    - ConfigManager focuses on ELMFIRE configuration management and input file generation
    - DataManager handles broader simulation data, including states, episodes, and RL metrics
    - DataManager does not interact with ELMFIRE directly or generate input files

    Interaction with StateManager and GeoSpatialManager:
    - StateManager: DataManager delegates state and episode management tasks to StateManager,
      including saving/loading states, managing episodes, and handling state history
    - GeoSpatialManager: DataManager uses GeoSpatialManager for spatial calculations and analysis,
      such as fire growth rate and high-risk area identification

    Key Methods:
    - update_state: Update the current simulation state
    - get_current_state: Retrieve the most recent simulation state
    - start_new_episode: Begin a new simulation episode
    - add_step_to_current_episode: Record a new step in the current episode
    - get_fire_growth_rate: Calculate the current fire growth rate
    - get_high_risk_areas: Identify areas at high risk in the current state
    - update_rl_metrics: Record metrics for reinforcement learning
    - export_episode_data: Save all data for a specific episode
    - get_episode_statistics: Calculate various statistics for an episode

    Thread Safety:
    The DataManager is designed with thread-safety in mind, utilizing thread-safe data structures
    and delegating to thread-safe sub-managers (StateManager and GeoSpatialManager).

    Performance Considerations:
    - Implements caching mechanisms to reduce redundant calculations
    - Uses efficient data structures for quick access to states and metrics
    - Delegates computationally intensive tasks to specialized managers

    Error Handling:
    Implements comprehensive error checking and logging for all data operations,
    ensuring data integrity and providing detailed feedback for debugging.

    Usage:
    Typically instantiated once at the start of the Overseer system and used throughout
    the simulation lifecycle to manage and provide access to all simulation-related data.

    Example:
        data_manager = DataManager(config)
        data_manager.update_state(new_state)
        current_state = data_manager.get_current_state()
        data_manager.start_new_episode()
        data_manager.add_step_to_current_episode(state, action, reward, next_state, done)
        fire_growth_rate = data_manager.get_fire_growth_rate(time_interval)

    The DataManager saves files in the following directory structure:

    File Structure:
    data/
    ├── episodes/
    │   ├── episode_1/
    │   │   ├── metadata.json
    │   │   ├── step_0/
    │   │   │   ├── state.json
    │   │   │   ├── action.json
    │   │   │   └── metrics.json
    │   │   ├── step_1/
    │   │   │   ├── state.json
    │   │   │   ├── action.json
    │   │   │   └── metrics.json
    │   │   └── ...
    │   ├── episode_2/
    │   │   └── ...
    │   └── ...
    └── global_metrics/
        └── overall_performance.json


    Note: This class is central to the Overseer system's data management and should be
    maintained with careful consideration of its impact on overall system performance and data integrity.
    """

    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.outputs_to_copy = config.get('data_management.outputs_to_copy', [])

        self.geospatial_manager = GeoSpatialManager(self.config)
        self.state_manager = StateManager(self.config)
        self.rl_metrics: Dict[int, List[Dict[str, float]]] = {}



    def update_state(self, state: SimulationState) -> None:
        self.logger.info("=" * 40)
        self.logger.info(f"Updating state at timestamp: {state.timestamp}")
        if state is None:
            self.logger.warning("State is None. Using Current State")
            state = self.get_current_state()
        state = self.calculate_state_metrics(state)
        self.logger.info(f"State metrics after calculation: {state.metrics}")
        self.state_manager.update_state(state)
        self.save_state_to_disk(state)
        self.logger.info(f"State updated and saved to disk")
        self.logger.info("=" * 40)

    def get_current_state(self) -> Optional[SimulationState]:
        self.logger.info("DataManager: Attempting to get current state")
        current_state = self.state_manager.get_current_state()
        if current_state is None:
            self.logger.warning("DataManager: Current state is None")
        else:
            self.logger.info(f"DataManager: Current state retrieved, timestamp: {current_state.timestamp}")
        return current_state

    def state_to_array(self, state: Optional[SimulationState] = None) -> np.ndarray:
        """
        Convert a SimulationState to a numpy array representation.

        This method takes a SimulationState object and converts its fire intensity data
        into a flattened numpy array. If no state is provided, it uses the current state.

        Args:
            state (Optional[SimulationState]): The state to convert. If None, use the current state.

        Returns:
            np.ndarray: A 1D numpy array representation of the state's fire intensity data.

        Raises:
            ValueError: If the fire intensity path does not exist or if the data cannot be read.
        """
        ##TODO:: USE INPUT AND OUTPUT DATA --> Encode 
        self.logger.info("Starting conversion of SimulationState to array")

        if state is None:
            self.logger.info("No state provided, using current state")
            state = self.get_current_state()

        flin_path = Path(state.paths.output_paths.fire_intensity)
        self.logger.debug(f"Fire intensity path: {flin_path}")

        if not flin_path.exists():
            self.logger.error(f"Fire intensity file does not exist: {flin_path}")
            raise ValueError(f"Fire intensity file does not exist: {flin_path}")

        try:
            flin_data = self.geospatial_manager.open_tiff(str(flin_path))
            flin_raster = flin_data['data']
            self.logger.debug(f"Fire intensity raster shape: {flin_raster.shape}")
        except Exception as e:
            self.logger.error(f"Failed to read fire intensity data: {str(e)}")
            raise ValueError(f"Failed to read fire intensity data: {str(e)}")

        # Flatten the raster to a 1D array
        state_array = flin_raster.flatten()

        self.logger.info(f"State array created with shape: {state_array.shape}")
        return state_array

    def get_state_history(self) -> List[SimulationState]:
        """Get the state history from the StateManager."""
        return self.state_manager.get_state_history()

    def get_state_by_episode_step(self, episode_id: int, step: int) -> Optional[SimulationState]:
        """Get a specific state by episode ID and step number."""
        return self.state_manager.get_state_by_episode_step(episode_id, step)

    def get_episode_id(self) -> int:
        """Get the current episode ID from the StateManager."""
        return self.state_manager.get_episode_id()
    
    def get_current_step(self) -> int:
        """Get the current step number from the StateManager."""
        return self.state_manager.get_current_step()
    
    def start_new_episode(self) -> None:
        """Start a new episode using the StateManager."""
        self.state_manager.start_new_episode()

    def add_step_to_current_episode(self, state: SimulationState, action: Action, reward: float, next_state: SimulationState, done: bool) -> None:
        """Add a new step to the current episode using the StateManager."""
        self.state_manager.add_step_to_current_episode(state, action, reward, next_state, done)

    def save_state_to_disk(self, state: SimulationState) -> None:
        self.logger.info(f"Saving state to disk at timestamp: {state.timestamp}")
        self.logger.info(f"Saving state to disk for episode {self.state_manager.current_episode_id}, step {self.state_manager.current_step}")
        save_filepath = self.state_manager._save_state_to_disk(state)
        self.logger.info(f"Saved state to disk at path: {save_filepath}")

    def get_current_episode(self) -> Optional[Episode]:
        """Get the current episode from the StateManager."""
        return self.state_manager.get_current_episode()

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """Get a specific episode by ID from the StateManager."""
        return self.state_manager.get_episode(episode_id)
    
    def add_step_to_current_episode(self, state: SimulationState, action: Action, reward: float, next_state: SimulationState, done: bool) -> None:
        step = EpisodeStep(
            step=self.state_manager.update_state_counter(),
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        #assert none of fields are None and add fail messages
        assert step.state is not None, "State is None"
        assert step.action is not None, "Action is None"    
        assert step.reward is not None, "Reward is None"
        assert step.next_state is not None, "Next state is None"
        assert step.done is not None, "Done is None"
        self.state_manager.add_step(step)




    def simresult_to_simstate(self, sim_result: SimulationResult, sim_config: SimulationConfig, sim_paths: SimulationPaths, timestamp: datetime) -> SimulationState:
        self.logger.info(f"Converting SimulationResult to SimulationState for job {sim_result.job_id}")

        try:
            # Update the output paths
            output_dir = Path(sim_paths.output_paths.time_of_arrival).parent
            flin_file = list(output_dir.glob("flin_*.tif"))
            toa_file = list(output_dir.glob("time_of_arrival_*.tif"))

            self.logger.debug(f"Found {len(flin_file)} fire intensity files and {len(toa_file)} time of arrival files")

            if not flin_file or not toa_file:
                self.logger.error(f"Required output files not found in {output_dir}")
                self.logger.debug(f"Contents of output directory: {list(output_dir.iterdir())}")
                return None

            new_output_paths = OutputPaths(
                time_of_arrival=toa_file[0],
                fire_intensity=flin_file[0],
                flame_length=None,
                spread_rate=None
            )
            self.logger.debug(f"New output paths: {new_output_paths}")

            new_paths = SimulationPaths(
                input_paths=sim_paths.input_paths,
                output_paths=new_output_paths
            )

            # Create a new SimulationState
            new_state = SimulationState(
                timestamp=timestamp,
                config=sim_config,
                paths=new_paths,
                metrics=SimulationMetrics(
                    burned_area=0.0,  # These will be updated in calculate_state_metrics
                    fire_perimeter_length=0.0,
                    containment_percentage=0.0,
                    execution_time=sim_result.duration,
                    performance_metrics={
                        'cpu_usage': sim_result.cpu_usage,
                        'memory_usage': sim_result.memory_usage
                    },
                    fire_intensity=None  # Placeholder, will be updated
                ),
                save_path=None,  # This can be set later if needed
                resources=None,  # This should be updated based on the action taken
                weather=None  # This should be updated based on the simulation results
            )

            # Calculate new metrics
            self.logger.info("Calculating new state metrics")
            new_state = self.calculate_state_metrics(new_state)



            self.logger.info(f"Successfully converted SimulationResult to SimulationState for job {sim_result.job_id}")
            return new_state

        except Exception as e:
            self.logger.error(f"Error in simresult_to_simstate: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            return None

    def calculate_state_metrics(self, state: SimulationState) -> SimulationState:
        """
        Calculate and update the simulation metrics for the given state.

        Args:
            state (SimulationState): The current simulation state.

        Returns:
            SimulationState: Updated simulation state with new metrics.
        """
        self.logger.info("Calculating state metrics")
        if state is None: # use the current state
            state = self.get_current_state()

        # Check if the required output files exist
        toa_path = Path(state.paths.output_paths.time_of_arrival)
        flin_path = Path(state.paths.output_paths.fire_intensity)

        if not toa_path.exists() or not flin_path.exists():
            self.logger.warning("[On Reset]: Required output files not found. Using default metrics.")
            new_metrics = SimulationMetrics(
                burned_area=0.0,
                fire_perimeter_length=0.0,
                containment_percentage=0.0,
                execution_time=0.0,
                performance_metrics={'cpu_usage': 0.0, 'memory_usage': 0.0},
                fire_intensity=None
            )
        else:
            new_metrics = self.geospatial_manager.calculate_state_metrics(state)
                
        # Create a new SimulationState with updated metrics
        updated_state = SimulationState(
            timestamp=state.timestamp,
            config=state.config,
            paths=state.paths,
            metrics=new_metrics,
            save_path=state.save_path,
            resources=state.resources,
            weather=state.weather
        )
        
        self.logger.info("State metrics calculated and updated")
        return updated_state
    

    def get_fire_growth_rate(self, time_interval: float) -> float:
        """Calculate the fire growth rate."""
        current_state = self.get_current_state()
        if current_state is None:
            return 0.0
        
        toa_path = fix_path(str(current_state.paths.output_paths.time_of_arrival), add_tif=True)
        if not toa_path:
            self.logger.error("Invalid or non-existent time of arrival path")
            return 0.0
        
        return self.geospatial_manager.calculate_fire_growth_rate(toa_path, time_interval)


    def reset(self) -> None:
        """Reset the state manager."""
        self.state_manager.reset()

    def cleanup_old_data(self, max_episodes: int) -> None:
        """Remove old simulation data to free up storage space."""
        self.state_manager.cleanup_old_episodes(max_episodes)

    def clear_all_data(self):
        self.state_manager.clear_all_data()

    def load_state_from_disk(self, episode_id: int, step: int) -> Optional[SimulationState]:
        self.logger.info(f"Loading state from disk for episode {episode_id}, step {step}")
        state = self.state_manager.load_state_from_disk(episode_id, step)
        if state:
            self.logger.info(f"Successfully loaded state from disk for episode {episode_id}, step {step}")
        else:
            self.logger.warning(f"Failed to load state from disk for episode {episode_id}, step {step}")
        return state

    def print_valid_episodes_and_steps(self):
        """Iterate through the episodes directory and print the valid episodes"""
        for episode_dir in self.data_dir.iterdir():
            if episode_dir.is_dir():
                self.logger.info(f"Valid episode: {episode_dir}")
                for step_dir in episode_dir.iterdir():
                    if step_dir.is_dir():
                        self.logger.info(f"Valid step: {step_dir}")

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

    def clear_all_episode_data(self):
        """Clean up the states directory."""
        self.state_manager.clear_all_episode_data()

    def update_fuel_file(self, filepath: str, fireline_coords: List[Tuple[int, int]]) -> None:
        """
        Update the fuel file with new fireline coordinates.
        
        Args:
            filepath (str): Path to the fuel file.
            fireline_coords (List[Tuple[int, int]]): List of fireline coordinates.
        """
        self.geospatial_manager.update_fuel_file(filepath, fireline_coords)


    def generate_action_mask_from_episode(self, episode_step: EpisodeStep) -> np.ndarray:
        """Generate an action mask based on the current episode step."""
        self.logger.info("Generating action mask from episode step")
        
        time_of_arrival_path = episode_step.state.paths.output_paths.time_of_arrival
        fbfm40_path = episode_step.state.paths.input_paths.fbfm_filename
        self.logger.info(f"Time of arrival path: {time_of_arrival_path}")
        self.logger.info(f"FBFM40 path: {fbfm40_path}")
        
        if not time_of_arrival_path.exists() or not fbfm40_path.exists():
            self.logger.warning("Required files for action mask generation not found")
            return np.ones((self.config.get('grid_size', 100), self.config.get('grid_size', 100)), dtype=bool)
        
        #TODO: add min and max distance if we need
        return self.geospatial_manager.generate_action_from_files(
            str(time_of_arrival_path),
            str(fbfm40_path)
        )
    

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