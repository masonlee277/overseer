import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import h5py
import json
import rasterio
from ..config.overseer_config import OverseerConfig
from ..utils.logging import OverseerLogger

class DataManager:
    """
    Manages data operations for ELMFIRE simulations and the RL environment.

    This class is responsible for:
    1. Saving and loading simulation states
    2. Managing geospatial data (saving and loading GeoTIFF files)
    3. Handling RL metrics
    4. Preprocessing data for the RL environment
    5. Managing intermediate states
    6. Aggregating and analyzing simulation results

    It does NOT:
    1. Run simulations (handled by SimulationManager)
    2. Modify ELMFIRE configurations (handled by ConfigurationManager)
    3. Interface directly with the ELMFIRE simulator
    4. Implement RL algorithms or environments

    The DataManager interfaces with:
    - SimulationManager: Provides data storage and retrieval for simulation states
    - ElmfireGymEnv: Offers data preprocessing for the RL environment
    - ConfigurationManager: May load/save configuration-related data

    Attributes:
        config (OverseerConfig): The OverseerConfig instance.
        logger (logging.Logger): Logger for this class.
        data_dir (Path): Root directory for all data storage.
        current_episode (int): The current episode number.
        current_step (int): The current step number within the episode.
    """

    def __init__(self, config: OverseerConfig):
        """
        Initialize the DataManager.

        Args:
            config (OverseerConfig): The OverseerConfig instance.
        """
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_dir = Path(config.get('data_directory'))
        self.current_episode = 0
        self.current_step = 0

    def save_simulation_state(self, state: Dict[str, Any]) -> None:
        """
        Save the current simulation state.

        Args:
            state (Dict[str, Any]): The current state of the ELMFIRE simulation.
        """
        filename = f"episode_{self.current_episode}_step_{self.current_step}.h5"
        filepath = self.data_dir / 'simulations' / 'raw' / filename
        
        try:
            with h5py.File(filepath, 'w') as hf:
                for key, value in state.items():
                    if isinstance(value, np.ndarray):
                        hf.create_dataset(key, data=value)
                    else:
                        hf.create_dataset(key, data=json.dumps(value))
            self.logger.info(f"Saved simulation state to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save simulation state: {str(e)}")
            raise

    def load_simulation_state(self, episode: int, step: int) -> Dict[str, Any]:
        """
        Load a simulation state from storage.

        Args:
            episode (int): The episode number.
            step (int): The step number within the episode.

        Returns:
            Dict[str, Any]: The loaded simulation state.
        """
        filename = f"episode_{episode}_step_{step}.h5"
        filepath = self.data_dir / 'simulations' / 'raw' / filename
        
        try:
            state = {}
            with h5py.File(filepath, 'r') as hf:
                for key in hf.keys():
                    if isinstance(hf[key][()], np.ndarray):
                        state[key] = hf[key][()]
                    else:
                        state[key] = json.loads(hf[key][()])
            self.logger.info(f"Loaded simulation state from {filepath}")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load simulation state: {str(e)}")
            raise

    def save_geospatial_data(self, data: np.ndarray, metadata: Dict[str, Any], filename: str) -> None:
        """
        Save geospatial data as a GeoTIFF file.

        Args:
            data (np.ndarray): The geospatial data to save.
            metadata (Dict[str, Any]): Metadata for the GeoTIFF file.
            filename (str): Name of the file to save.
        """
        filepath = self.data_dir / 'simulations' / 'processed' / filename
        
        try:
            with rasterio.open(filepath, 'w', **metadata) as dst:
                dst.write(data)
            self.logger.info(f"Saved geospatial data to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save geospatial data: {str(e)}")
            raise

    def load_geospatial_data(self, filename: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load geospatial data from a GeoTIFF file.

        Args:
            filename (str): Name of the file to load.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The loaded data and its metadata.
        """
        filepath = self.data_dir / 'simulations' / 'processed' / filename
        
        try:
            with rasterio.open(filepath) as src:
                data = src.read()
                metadata = src.meta
            self.logger.info(f"Loaded geospatial data from {filepath}")
            return data, metadata
        except Exception as e:
            self.logger.error(f"Failed to load geospatial data: {str(e)}")
            raise

    def save_rl_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Save metrics from the RL training process.

        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values.
        """
        filename = f"rl_metrics_episode_{self.current_episode}.json"
        filepath = self.data_dir / 'metrics' / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(metrics, f)
            self.logger.info(f"Saved RL metrics to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save RL metrics: {str(e)}")
            raise

    def load_rl_metrics(self, episode: int) -> Dict[str, float]:
        """
        Load metrics from a specific RL training episode.

        Args:
            episode (int): The episode number to load metrics for.

        Returns:
            Dict[str, float]: Dictionary of metric names and values.
        """
        filename = f"rl_metrics_episode_{episode}.json"
        filepath = self.data_dir / 'metrics' / filename
        
        try:
            with open(filepath, 'r') as f:
                metrics = json.load(f)
            self.logger.info(f"Loaded RL metrics from {filepath}")
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to load RL metrics: {str(e)}")
            raise

    def preprocess_state_for_rl(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess the simulation state for use in the RL environment.

        Args:
            state (Dict[str, Any]): The raw simulation state.

        Returns:
            np.ndarray: The preprocessed state suitable for the RL environment.
        """
        try:
            numeric_data = []
            for value in state.values():
                if isinstance(value, (int, float, np.ndarray)):
                    numeric_data.extend(np.array(value).flatten())
            preprocessed_state = np.array(numeric_data)
            self.logger.info("Preprocessed simulation state for RL")
            return preprocessed_state
        except Exception as e:
            self.logger.error(f"Failed to preprocess state for RL: {str(e)}")
            raise

    def save_intermediate_state(self, state: Dict[str, Any]) -> None:
        """
        Save an intermediate state of the simulation.

        Args:
            state (Dict[str, Any]): The intermediate state to save.
        """
        filename = f"intermediate_state_episode_{self.current_episode}_step_{self.current_step}.h5"
        filepath = self.data_dir / 'intermediate' / filename
        
        try:
            with h5py.File(filepath, 'w') as hf:
                for key, value in state.items():
                    if isinstance(value, np.ndarray):
                        hf.create_dataset(key, data=value)
                    else:
                        hf.create_dataset(key, data=json.dumps(value))
            self.logger.info(f"Saved intermediate state to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save intermediate state: {str(e)}")
            raise

    def load_intermediate_state(self, episode: int, step: int) -> Dict[str, Any]:
        """
        Load an intermediate state of the simulation.

        Args:
            episode (int): The episode number.
            step (int): The step number within the episode.

        Returns:
            Dict[str, Any]: The loaded intermediate state.
        """
        filename = f"intermediate_state_episode_{episode}_step_{step}.h5"
        filepath = self.data_dir / 'intermediate' / filename
        
        try:
            state = {}
            with h5py.File(filepath, 'r') as hf:
                for key in hf.keys():
                    if isinstance(hf[key][()], np.ndarray):
                        state[key] = hf[key][()]
                    else:
                        state[key] = json.loads(hf[key][()])
            self.logger.info(f"Loaded intermediate state from {filepath}")
            return state
        except Exception as e:
            self.logger.error(f"Failed to load intermediate state: {str(e)}")
            raise

    def aggregate_simulation_results(self, episode: int) -> Dict[str, Any]:
        """
        Aggregate results from all steps of a simulation episode.

        Args:
            episode (int): The episode number to aggregate results for.

        Returns:
            Dict[str, Any]: Aggregated simulation results.
        """
        try:
            results = []
            step = 0
            while True:
                try:
                    state = self.load_simulation_state(episode, step)
                    results.append(state)
                    step += 1
                except FileNotFoundError:
                    break
            
            aggregated_results = {
                'episode': episode,
                'total_steps': step,
                'final_state': results[-1] if results else None,
                'trajectory': results
            }
            self.logger.info(f"Aggregated results for episode {episode}")
            return aggregated_results
        except Exception as e:
            self.logger.error(f"Failed to aggregate simulation results: {str(e)}")
            raise

    def increment_step(self) -> None:
        """Increment the current step counter."""
        self.current_step += 1

    def increment_episode(self) -> None:
        """Increment the current episode counter and reset the step counter."""
        self.current_episode += 1
        self.current_step = 0

    def get_latest_episode_and_step(self) -> Tuple[int, int]:
        """
        Get the latest episode and step numbers from stored data.

        Returns:
            Tuple[int, int]: The latest episode and step numbers.
        """
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