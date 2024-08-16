import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Any
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import threading
import shutil 

from overseer.config.config import OverseerConfig
from overseer.core.models import SimulationState, Episode, EpisodeStep, Action, SimulationConfig, SimulationPaths, SimulationMetrics, InputPaths, OutputPaths
from overseer.utils.logging import OverseerLogger

class StateManager:
    """
    Manages the state of the simulation, including current state, state history, and episodes.

    This class is responsible for:
    1. Maintaining the current simulation state
    2. Managing the history of states
    3. Handling episodes and their steps
    4. Saving and loading states and episodes to/from disk
    5. Providing access to states and episodes

    The StateManager works closely with SimulationState, Episode, and EpisodeStep objects:
    - SimulationState: Represents the state of the simulation at a given point in time
    - Episode: Represents a complete episode of the simulation, containing multiple EpisodeSteps
    - EpisodeStep: Represents a single step within an episode, containing state, action, and reward information

    Attributes:
        config (OverseerConfig): Configuration object for the Overseer system
        logger (logging.Logger): Logger for this class
        data_dir (Path): Directory for storing state and episode data
        current_state (Optional[SimulationState]): The current state of the simulation
        state_history (List[SimulationState]): History of all states in the current episode
        episodes (Dict[int, Episode]): Dictionary of all episodes, keyed by episode ID
        current_episode_id (int): ID of the current episode
        current_step (int): Current step number within the current episode
    """

    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_dir = Path(self.config.get('data_dir', 'data'))
        self.current_state: Optional[SimulationState] = None
        self.state_history: List[SimulationState] = []
        self.episodes: Dict[int, Episode] = {}
        self.current_episode_id: int = 0
        self.current_step: int = 0
        self.state_counter_lock = threading.Lock()

    def update_state(self, new_state: SimulationState) -> None:
        """
        Update the current state and add it to the state history.
        """
        self.logger.info("--------------------")
        state_counter = self.update_state_counter()
        self.logger.info(f"Updating state for episode {self.current_episode_id}, step {state_counter}")
        self.ensure_episode_exists(self.current_episode_id)
        self.logger.info(f"Episodes keys: {self.episodes.keys()}")

        self.current_state = new_state
        self.state_history.append(new_state)
        self._save_state_to_disk(new_state)
        self.logger.debug(f"State updated: {new_state}")
        self.logger.info("--------------------")


    def add_step(self, step: EpisodeStep) -> None:
        """
        Add a new step to the current episode and update the current state.
        """
        self.logger.info(f"Adding step to episode {self.current_episode_id}, step {step.step}")
        self.ensure_episode_exists(self.current_episode_id)
        episode = self.episodes[self.current_episode_id]
        episode.steps.append(step)
        episode.total_reward += step.reward
        episode.total_steps += 1
        self.current_state = step.next_state
        self.current_step = step.step
        self._save_episode_step_to_disk(step, self.current_step)


    def get_current_state(self) -> Optional[SimulationState]:
        self.logger.info(f"StateManager: Getting current state. Episode: {self.current_episode_id}, Step: {self.current_step}")
        if self.current_state is None:
            self.logger.warning("StateManager: Current state is None")
        else:
            self.logger.info(f"StateManager: Returning current state, timestamp: {self.current_state.timestamp}")
        return self.current_state
    

    def get_state_history(self) -> List[SimulationState]:
        """
        Get the history of states for the current episode.

        Returns:
            List[SimulationState]: List of all states in the current episode
        """
        return self.state_history

    def get_state_by_episode_step(self, episode_id: int, step: int) -> Optional[SimulationState]:
        """
        Get a specific state by episode ID and step number.

        Args:
            episode_id (int): The ID of the episode
            step (int): The step number within the episode

        Returns:
            Optional[SimulationState]: The requested state, or None if not found
        """
        self.logger.info(f"Retrieving state for episode {episode_id}, step {step}")
        episode = self.episodes.get(episode_id)
        if episode and 0 <= step < len(episode.steps):
            return episode.steps[step].state
        self.logger.warning(f"State not found for episode {episode_id}, step {step}")
        return None

    def update_state_counter(self) -> int:
        """
        Thread-safe method to update and return the current state counter.
        """
        with self.state_counter_lock:
            self.current_step += 1
            return self.current_step

    def ensure_episode_exists(self, episode_id: int) -> None:
        """
        Ensure that an episode with the given ID exists. If not, create it.
        """
        if episode_id not in self.episodes:
            self.logger.info(f"Creating new episode with ID {episode_id}")
            self.episodes[episode_id] = Episode(episode_id=episode_id, steps=[], total_reward=0.0, total_steps=0)
            self._create_episode_directory(episode_id)

    def reset_state_counter(self) -> None:
        """
        Thread-safe method to reset the state counter.
        """
        with self.state_counter_lock:
            self.current_step = 0


    def start_new_episode(self) -> None:
        self.reset_state_counter()
        self.current_episode_id += 1
        self.current_state = None
        self.current_step = 0  
        self.state_history.clear()

        self.ensure_episode_exists(self.current_episode_id)
        self.logger.debug(f"New episode started: {self.current_episode_id}")

    def add_step_to_current_episode(self, state: SimulationState, action: Action, reward: float, next_state: SimulationState, done: bool) -> None:
        """
        Add a new step to the current episode.

        Args:
            state (SimulationState): The current state
            action (Action): The action taken
            reward (float): The reward received
            next_state (SimulationState): The resulting state after taking the action
            done (bool): Whether the episode is finished
            state_counter (int): The current state counter from DataManager
        """
        state_counter = self.update_state_counter()
        self.logger.info(f"Adding step to episode {self.current_episode_id}, step {state_counter}")
        episode = self.episodes[self.current_episode_id]
        step = EpisodeStep(
            step=state_counter,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        episode.steps.append(step)
        episode.total_reward += reward
        episode.total_steps += 1
        self.current_step = state_counter
        self._save_episode_step_to_disk(step, state_counter)
        self.logger.debug(f"Step added: {step}")

    def get_current_episode(self) -> Optional[Episode]:
        """
        Get the current episode.

        Returns:
            Optional[Episode]: The current episode, or None if no episode is in progress
        """
        self.logger.info(f"Retrieving current episode (ID: {self.current_episode_id})")
        current_episode = self.episodes.get(self.current_episode_id)
        if current_episode:
            self.logger.info(f"Current episode found. Total steps: {current_episode.total_steps}")
            print(f"Current episode - ID: {self.current_episode_id}, Total steps: {current_episode.total_steps}")
        else:
            self.logger.warning("No current episode found")
            print("No current episode in progress")

        #log more about self.episodes (which is a dictionary)
        self.logger.info(f"Episodes: {self.episodes.keys()}")
        return current_episode

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """
        Get a specific episode by ID.

        Args:
            episode_id (int): The ID of the episode to retrieve

        Returns:
            Optional[Episode]: The requested episode, or None if not found
        """
        return self.episodes.get(episode_id)

    def _create_simulation_state(self, state_data: Dict) -> SimulationState:
        """
        Create a SimulationState object from a dictionary of state data.

        Args:
            state_data (Dict): Dictionary containing state data

        Returns:
            SimulationState: The created SimulationState object
        """
        self.logger.debug("Creating SimulationState from dictionary")
        return SimulationState(
            timestamp=datetime.now(),
            config=SimulationConfig(**state_data.get('config', {})),
            paths=SimulationPaths(
                input_paths=InputPaths(**state_data.get('input_paths', {})),
                output_paths=OutputPaths(**state_data.get('output_paths', {}))
            ),
            metrics=SimulationMetrics(**state_data.get('metrics', {})),
            save_path=Path(state_data.get('save_path', '')),
            resources=state_data.get('resources', {}),
            weather=state_data.get('weather', {})
        )


    def _parse_timestamp(self, timestamp: Union[str, datetime]) -> str:
        """
        Parse and format the timestamp consistently.

        Args:
            timestamp (Union[str, datetime]): The timestamp to parse

        Returns:
            str: The formatted timestamp string
        """
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                self.logger.error(f"Invalid timestamp format: {timestamp}")
                raise ValueError(f"Invalid timestamp format: {timestamp}")
        
        return timestamp.strftime("%Y%m%d_%H%M%S_%f")

    def _get_state_filename(self, timestamp: Union[str, datetime]) -> Path:
        """
        Get the filename for a state file based on the timestamp.

        Args:
            timestamp (Union[str, datetime]): The timestamp of the state

        Returns:
            Path: The path to the state file
        """
        formatted_timestamp = self._parse_timestamp(timestamp)
        return self.data_dir / 'states' / f'state_{formatted_timestamp}.json'

    def _save_state_to_disk(self, state: SimulationState) -> Path:
        """
        Save a state to disk.

        Args:
            state (SimulationState): The state to save

        Returns:
            Path: The path to the state file
        """
        state_counter = self.current_step
        self.logger.info(f"Saving state to disk for episode {self.current_episode_id}, step {state_counter}")
        episode_dir = self._get_episode_directory(self.current_episode_id)
        step_dir = episode_dir / f"step_{state_counter}"
        step_dir.mkdir(parents=True, exist_ok=True)
        state_file = step_dir / "state.json"
        self.logger.info(f"Saving state to {state_file}")

        try:
            with open(state_file, 'w') as f:
                json.dump(self._serialize_state(state), f, default=self._json_serializer, indent=2)
            self.logger.debug(f"State saved to {state_file}")
            return state_file
        except OSError as e:
            self.logger.error(f"Failed to save state to disk: {e}")
            raise

    def _get_episode_directory(self, episode_id: int) -> Path:
        return self.data_dir / "episodes" / f"episode_{episode_id}"

    def _create_episode_directory(self, episode_id: int) -> None:
        episode_dir = self._get_episode_directory(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = episode_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({"episode_id": episode_id, "start_time": datetime.now().isoformat()}, f)


    def load_state_from_disk(self, episode_id: int, step: int) -> Optional[SimulationState]:
        """Load a state from disk."""
        self.logger.info(f"Loading state from disk for episode {episode_id}, step {step}")
        episode_dir = self._get_episode_directory(episode_id)
        step_dir = episode_dir / f"step_{step}"
        state_file = step_dir / "state.json"
        self.logger.info(f"Trying loading state from {state_file}")
        if not state_file.exists():
            self.logger.warning(f"State file not found: {state_file}")
            return None

        try:
            with open(state_file, 'r') as f:
                state_data = json.load(f)
            return self._deserialize_state(state_data)
        except Exception as e:
            self.logger.error(f"Failed to load state from disk: {str(e)}")
            return None

    def _get_state_file_path(self, timestamp: str) -> Path:
        return self.data_dir / 'states' / f"state_{timestamp.replace(':', '-')}.json"
        
    
    def _save_episode_step_to_disk(self, step: EpisodeStep, state_counter: int) -> None:
        """
        Save an episode step to disk.

        Args:
            step (EpisodeStep): The episode step to save
            state_counter (int): The current state counter
        """
        self.logger.info(f"Saving episode step to disk for episode {self.current_episode_id}, step {state_counter}")
        episode_dir = self._get_episode_directory(self.current_episode_id)
        step_dir = episode_dir / f"step_{state_counter}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving episode step to {step_dir}")
        # Serialize and save the entire episode step
        step_file = step_dir / "episode_step.json"
        with open(step_file, 'w') as f:
            json.dump(self._serialize_episode_step(step), f, default=self._json_serializer, indent=2)

        self.logger.debug(f"Episode step saved to {step_file}")
    
    def clear_all_episode_data(self) -> None:
        self.logger.info("Clearing all episode data")
        episodes_dir = self.data_dir / "episodes"
        states_dir = self.data_dir / "states"
        
        for dir_to_clear in [episodes_dir, states_dir]:
            if dir_to_clear.exists():
                try:
                    shutil.rmtree(dir_to_clear)
                    dir_to_clear.mkdir(exist_ok=True)
                    self.logger.info(f"Cleared {dir_to_clear}")
                except Exception as e:
                    self.logger.error(f"Failed to clear {dir_to_clear}: {str(e)}")
            else:
                self.logger.warning(f"{dir_to_clear} does not exist")

        # Reset internal state
        self.episodes.clear()
        self.current_episode_id = 0
        self.current_step = 0
        self.current_state = None
        self.state_history.clear()

    def _serialize_state(self, state: SimulationState) -> Dict:
        """
        Serialize a SimulationState object to a dictionary.

        Args:
            state (SimulationState): The state to serialize

        Returns:
            Dict: The serialized state
        """
        #log the timestamp
        self.logger.info(f"Serializing state with timestamp: {state.timestamp}")
        return {
            'timestamp': state.timestamp.isoformat(),
            'config': self._serialize_config(state.config),
            'paths': {
                'input_paths': self._serialize_input_paths(state.paths.input_paths),
                'output_paths': self._serialize_output_paths(state.paths.output_paths)
            },
            'metrics': self._serialize_metrics(state.metrics),
            'save_path': str(state.save_path),
            'resources': state.resources,
            'weather': state.weather
        }

    def _serialize_episode_step(self, step: EpisodeStep) -> Dict:
        """
        Serialize an EpisodeStep object to a dictionary.

        Args:
            step (EpisodeStep): The episode step to serialize

        Returns:
            Dict: The serialized episode step
        """
        return {
            'step': step.step,
            'state': self._serialize_state(step.state),
            'action': {'fireline_coordinates': step.action.fireline_coordinates},
            'reward': step.reward,
            'next_state': self._serialize_state(step.next_state),
            'done': step.done
        }

    def _json_serializer(self, obj):
        """
        Custom JSON serializer for objects not serializable by default json code.

        Args:
            obj: The object to serialize

        Returns:
            A JSON serializable version of the object
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        self.logger.warning(f"Unserializable object encountered: {type(obj)}")
        return str(obj)

    def _serialize_config(self, config: SimulationConfig) -> Dict:
        if config is None:
            self.logger.warning("[_serialize_config] Config is None, returning empty dictionary")
            return {}
        else:
            return config.to_dict()

    def _serialize_action(self, action: Action) -> Dict[str, Any]:
        return {
            'fireline_coordinates': action.fireline_coordinates
        }

    def _deserialize_state(self, state_data: Dict) -> SimulationState:
        self.logger.debug("Deserializing state data to SimulationState")
        
        timestamp = datetime.fromisoformat(state_data['timestamp'])
        self.logger.debug(f"Deserialized timestamp: {timestamp}")
        
        config = SimulationConfig.from_dict(state_data['config'])
        self.logger.debug(f"Deserialized config: {config}")
        
        input_paths = InputPaths(**{k: Path(v) for k, v in state_data['paths']['input_paths'].items()})
        self.logger.debug(f"Deserialized input_paths: {input_paths}")
        
        output_paths = OutputPaths(**{k: Path(v) for k, v in state_data['paths']['output_paths'].items()})
        self.logger.debug(f"Deserialized output_paths: {output_paths}")
        
        paths = SimulationPaths(input_paths=input_paths, output_paths=output_paths)
        self.logger.debug(f"Deserialized paths: {paths}")
        
        metrics = SimulationMetrics(**state_data['metrics'])
        self.logger.debug(f"Deserialized metrics: {metrics}")
        
        save_path = Path(state_data['save_path'])
        self.logger.debug(f"Deserialized save_path: {save_path}")
        
        resources = state_data['resources']
        self.logger.debug(f"Deserialized resources: {resources}")
        
        weather = state_data['weather']
        self.logger.debug(f"Deserialized weather: {weather}")
        
        deserialized_state = SimulationState(
            timestamp=timestamp,
            config=config,
            paths=paths,
            metrics=metrics,
            save_path=save_path,
            resources=resources,
            weather=weather
        )
        self.logger.debug(f"Fully deserialized SimulationState: {deserialized_state}")
        
        return deserialized_state
    def _serialize_input_paths(self, input_paths: InputPaths) -> Dict:
        return {k: str(v) for k, v in input_paths.__dict__.items()}

    def _serialize_output_paths(self, output_paths: OutputPaths) -> Dict:
        return {k: str(v) for k, v in output_paths.__dict__.items()}

    def _serialize_metrics(self, metrics: SimulationMetrics) -> Dict:
        serialized = metrics.__dict__.copy()
        if serialized['fire_intensity'] is not None:
            serialized['fire_intensity'] = serialized['fire_intensity'].tolist()

        #if its none throw a warning
        if serialized['fire_intensity'] is None:
            self.logger.warning("[StateManager] Fire intensity is None")
        return serialized
    
    def clean_states(self):
        """
        Clean up the states directory by removing all state files.
        """
        self.logger.info("Cleaning up states directory")
        states_dir = self.data_dir / 'states'
        if states_dir.exists():
            for state_file in states_dir.glob('state_*.json'):
                try:
                    state_file.unlink()
                    self.logger.debug(f"Removed state file: {state_file}")
                except Exception as e:
                    self.logger.error(f"Failed to remove state file {state_file}: {str(e)}")
        self.logger.info("States directory cleanup completed")


    def state_to_array(self, state: Optional[SimulationState] = None) -> np.ndarray:
        """
        Convert a SimulationState to a numpy array.

        Args:
            state (Optional[SimulationState]): The state to convert. If None, use the current state.

        Returns:
            np.ndarray: A numpy array representation of the state.
        """
        if state is None:
            state = self.get_current_state()

            #log statemanger converying state to array
            self.logger.info(f"StateManager converting state to array: {state}")
            
        if state is None:
            self.logger.warning("No state available to convert to array.")
            return np.array([])

        # Example implementation (commented out):
        """
        layers = []
        for file_path in [state.paths.input_paths.fbfm_filename,
                          state.paths.input_paths.cbd_filename,
                          state.paths.input_paths.cbh_filename,
                          state.paths.input_paths.ch_filename,
                          state.paths.input_paths.cc_filename]:
            layer, _ = self.geospatial_manager.load_tiff(str(file_path))
            layers.append(layer)

        # Add fire intensity layer
        layers.append(state.metrics.fire_intensity)

        # Stack layers into a 3D array
        state_array = np.stack(layers, axis=-1)
        return state_array
        """

        # Mocked version
        return np.random.rand(100, 100, 6)  # 6 layers: 5 fuel layers + 1 fire intensity layer



    def load_episode_from_disk(self, episode_id: int) -> Optional[Episode]:
        """
        Load an episode from disk.

        Args:
            episode_id (int): The ID of the episode to load

        Returns:
            Optional[Episode]: The loaded episode, or None if not found
        """
        self.logger.info(f"Loading episode from disk for episode {episode_id}")

        # Log the contents of the data directory
        self.logger.info(f"Data directory contents: {os.listdir(self.data_dir)}")

        episodes_dir = self.data_dir / 'episodes'
        self.logger.info(f"Episodes directory contents: {os.listdir(episodes_dir)}")

        episode_dir = episodes_dir / f'episode_{episode_id}'
        self.logger.info(f"Episode directory: {episode_dir}")

        if not episode_dir.exists():
            self.logger.warning(f"Episode directory not found for episode {episode_id}")
            return None

        self.logger.info(f"Episode directory found for episode {episode_id}")
        self.logger.info(f"Episode directory contents: {os.listdir(episode_dir)}")

        steps = []
        for step_dir in sorted(episode_dir.glob('step_*')):
            self.logger.info(f"Processing step directory: {step_dir}")
            step_file = step_dir / 'episode_step.json'
            self.logger.info(f"Looking for step file: {step_file}")

            if step_file.exists():
                self.logger.info(f"Step file found: {step_file}")
                with open(step_file, 'r') as f:
                    step_data = json.load(f)
                    deserialized_step = self._deserialize_episode_step(step_data)
                    steps.append(deserialized_step)
                    self.logger.info(f"Loaded step {deserialized_step.step} for episode {episode_id}")
            else:
                self.logger.warning(f"Step file not found: {step_file}")

        self.logger.info(f"Loaded {len(steps)} steps for episode {episode_id}")

        if steps:
            episode = Episode(
                episode_id=episode_id,
                steps=steps,
                total_reward=sum(step.reward for step in steps),
                total_steps=len(steps)
            )
            self.episodes[episode_id] = episode
            self.logger.info(f"Episode {episode_id} loaded with {len(steps)} steps")
            return episode
        else:
            self.logger.warning(f"No steps found for episode {episode_id}")
            return None
        

    def _deserialize_episode_step(self, step_data: Dict) -> EpisodeStep:
        """
        Deserialize a dictionary to an EpisodeStep object.

        Args:
            step_data (Dict): The serialized episode step data

        Returns:
            EpisodeStep: The deserialized EpisodeStep object
        """
        self.logger.debug("Deserializing step data to EpisodeStep")
        return EpisodeStep(
            step=step_data['step'],
            state=self._deserialize_state(step_data['state']),
            action=Action(**step_data['action']),
            reward=step_data['reward'],
            next_state=self._deserialize_state(step_data['next_state']),
            done=step_data['done']
        )

    def cleanup_old_episodes(self, max_episodes: int) -> None:
        """
        Remove old episode data to free up storage space.

        Args:
            max_episodes (int): Maximum number of recent episodes to keep
        """
        self.logger.info(f"Cleaning up old episodes, keeping the most recent {max_episodes}")
        episode_dirs = sorted(self.data_dir.glob('episode_*'), key=lambda x: int(x.name.split('_')[1]), reverse=True)
        for old_dir in episode_dirs[max_episodes:]:
            self.logger.info(f"Removing old episode data: {old_dir}")
            for file in old_dir.glob('*'):
                file.unlink()
            old_dir.rmdir()
        self.logger.debug(f"Cleaned up {len(episode_dirs) - max_episodes} old episodes")

    def get_episode_summary(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a summary of an episode.

        Args:
            episode_id (int): The ID of the episode

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing episode summary, or None if not found
        """
        self.logger.info(f"Generating summary for episode {episode_id}")
        episode = self.episodes.get(episode_id)
        if episode is None:
            self.logger.warning(f"Episode {episode_id} not found")
            return None

        return {
            'episode_id': episode.episode_id,
            'total_reward': episode.total_reward,
            'total_steps': episode.total_steps,
            'start_time': episode.steps[0].state.timestamp if episode.steps else None,
            'end_time': episode.steps[-1].state.timestamp if episode.steps else None,
            'final_burned_area': episode.steps[-1].state.metrics.burned_area if episode.steps else None,
            'final_containment_percentage': episode.steps[-1].state.metrics.containment_percentage if episode.steps else None
        }

    def get_all_episode_summaries(self) -> List[Dict[str, Any]]:
        """
        Get summaries for all episodes.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing episode summaries
        """
        self.logger.info("Generating summaries for all episodes")
        return [self.get_episode_summary(episode_id) for episode_id in self.episodes.keys()]

    def export_episode_data(self, episode_id: int, export_path: Path) -> None:
        """
        Export all data for a specific episode to a file.

        Args:
            episode_id (int): The ID of the episode to export
            export_path (Path): The path to export the data to
        """
        self.logger.info(f"Exporting data for episode {episode_id} to {export_path}")
        episode = self.episodes.get(episode_id)
        if episode is None:
            self.logger.warning(f"Episode {episode_id} not found, cannot export")
            return

        export_data = {
            'episode_id': episode.episode_id,
            'total_reward': episode.total_reward,
            'total_steps': episode.total_steps,
            'steps': [self._serialize_episode_step(step) for step in episode.steps]
        }

        with open(export_path, 'w') as f:
            json.dump(export_data, f, default=self._json_serializer, indent=2)
        self.logger.debug(f"Episode {episode_id} data exported successfully")

    def import_episode_data(self, import_path: Path) -> Optional[int]:
        """
        Import episode data from a file.

        Args:
            import_path (Path): The path to import the data from

        Returns:
            Optional[int]: The imported episode ID, or None if import failed
        """
        self.logger.info(f"Importing episode data from {import_path}")
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            episode_id = import_data['episode_id']
            steps = [self._deserialize_episode_step(step_data) for step_data in import_data['steps']]

            episode = Episode(
                episode_id=episode_id,
                steps=steps,
                total_reward=import_data['total_reward'],
                total_steps=import_data['total_steps']
            )

            self.episodes[episode_id] = episode
            self._save_episode_to_disk(episode)
            self.logger.debug(f"Episode {episode_id} imported successfully")
            return episode_id
        except Exception as e:
            self.logger.error(f"Failed to import episode data: {str(e)}")
            return None

    def get_state_difference(self, state1: SimulationState, state2: SimulationState) -> Dict[str, Any]:
        """
        Calculate the difference between two simulation states.

        Args:
            state1 (SimulationState): The first state
            state2 (SimulationState): The second state

        Returns:
            Dict[str, Any]: A dictionary containing the differences between the two states
        """
        self.logger.info("Calculating state difference")
        diff = {
            'timestamp_diff': (state2.timestamp - state1.timestamp).total_seconds(),
            'burned_area_diff': state2.metrics.burned_area - state1.metrics.burned_area,
            'containment_percentage_diff': state2.metrics.containment_percentage - state1.metrics.containment_percentage,
            'resources_diff': {
                k: state2.resources.get(k, 0) - state1.resources.get(k, 0)
                for k in set(state1.resources.keys()) | set(state2.resources.keys())
            },
            'weather_diff': {
                k: state2.weather.get(k, 0) - state1.weather.get(k, 0)
                for k in set(state1.weather.keys()) | set(state2.weather.keys())
            }
        }
        self.logger.debug(f"State difference calculated: {diff}")
        return diff

    def get_episode_statistics(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """
        Calculate various statistics for an episode.

        Args:
            episode_id (int): The ID of the episode

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing episode statistics, or None if not found
        """
        self.logger.info(f"Calculating statistics for episode {episode_id}")
        episode = self.episodes.get(episode_id)
        if episode is None:
            self.logger.warning(f"Episode {episode_id} not found")
            return None

        rewards = [step.reward for step in episode.steps]
        burned_areas = [step.state.metrics.burned_area for step in episode.steps]
        containment_percentages = [step.state.metrics.containment_percentage for step in episode.steps]

        stats = {
            'episode_id': episode_id,
            'total_reward': episode.total_reward,
            'total_steps': episode.total_steps,
            'average_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'final_burned_area': burned_areas[-1],
            'max_burned_area': np.max(burned_areas),
            'average_burned_area': np.mean(burned_areas),
            'final_containment_percentage': containment_percentages[-1],
            'max_containment_percentage': np.max(containment_percentages),
            'average_containment_percentage': np.mean(containment_percentages),
        }
        self.logger.debug(f"Episode statistics calculated: {stats}")
        return stats

    def get_episode_id(self) -> int:
        return self.current_episode_id

    def get_current_step(self) -> int:
        return self.current_step

    def reset(self):
        self.logger.info("Resetting StateManager")
        self.current_state = None
        self.state_history.clear()
        self.current_episode_id += 1
        self.current_step = 0
        self.ensure_episode_exists(self.current_episode_id)
        self.logger.debug("StateManager reset complete")

    def clear_all_data(self):
        self.reset()
        self.clear_all_episode_data()  # This method should delete the files

