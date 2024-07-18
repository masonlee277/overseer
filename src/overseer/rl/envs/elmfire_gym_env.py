import gym
import numpy as np
from typing import Dict, Any, Tuple

from overseer.config import OverseerConfig
from overseer.elmfire import SimulationManager
from overseer.data import DataManager
from overseer.rl.rewards import RewardManager
from overseer.rl.spaces import ActionSpace, ObservationSpace
from overseer.rl.utils import StateEncoder, ActionDecoder

class ElmfireGymEnv(gym.Env):
    """
    A Gym-like environment for the ELMFIRE simulator.
    
    This class wraps the ELMFIRE simulator and provides a standard Gym-like interface
    for reinforcement learning agents to interact with the simulator. It uses the
    SimulationManager to handle ELMFIRE simulations, the ConfigurationManager to
    manage ELMFIRE configurations, and the DataManager to handle data operations.
    
    Attributes:
        sim_manager (SimulationManager): Manages ELMFIRE simulations.
        config_manager (ConfigurationManager): Manages ELMFIRE configurations.
        data_manager (DataManager): Manages data operations.
        dim_reduction_model (ConvolutionalAutoencoder): The dimensionality reduction model.
        observation_space (gym.Space): The observation space of the environment.
        action_space (gym.Space): The action space of the environment.
        state_encoder (StateEncoder): Encodes ELMFIRE states into RL observations.
        action_decoder (ActionDecoder): Decodes RL actions into ELMFIRE inputs.
        reward_calculator (RewardCalculator): Calculates rewards based on ELMFIRE outputs.
    """
    
    def __init__(self, config_path: str, elmfire_path: str, output_path: str, data_path: str,
                 dim_reduction_model: ConvolutionalAutoencoder,
                 observation_space: gym.Space, action_space: gym.Space):
        """
        Initialize the ElmfireGymEnv.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            elmfire_path (str): Path to the ELMFIRE executable.
            output_path (str): Path where simulation outputs should be stored.
            data_path (str): Path where the DataManager should store data.
            dim_reduction_model (ConvolutionalAutoencoder): The dimensionality reduction model.
            observation_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
        """
        self.config_manager = ConfigurationManager(config_path)
        self.sim_manager = SimulationManager(self.config_manager, elmfire_path, output_path)
        self.data_manager = DataManager(data_path)
        
        self.dim_reduction_model = dim_reduction_model
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.state_encoder = StateEncoder(dim_reduction_model)
        self.action_decoder = ActionDecoder()
        self.reward_calculator = RewardCalculator()
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        
        This method resets the ELMFIRE simulator to its initial configuration,
        runs an initial simulation, and returns the encoded initial state.
        
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The initial encoded observation and info dictionary.
        """
        self.config_manager.reset_to_default()
        initial_state = self.sim_manager.run_simulation()
        
        self.data_manager.increment_episode()
        self.data_manager.save_simulation_state(initial_state)
        
        preprocessed_state = self.data_manager.preprocess_state_for_rl(initial_state)
        encoded_state = self.state_encoder.encode(preprocessed_state)
        
        return encoded_state, {}
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment given an action.
        
        This method applies the action to the ELMFIRE simulation, runs the simulation,
        and returns the next state, reward, and other information.
        
        Args:
            action: The action to be taken in the environment.
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: 
                The encoded next state, reward, done flag, truncated flag, and info dictionary.
        """
        decoded_action = self.action_decoder.decode(action)
        next_state = self.sim_manager.apply_action(decoded_action)
        
        self.data_manager.increment_step()
        self.data_manager.save_simulation_state(next_state)
        
        preprocessed_next_state = self.data_manager.preprocess_state_for_rl(next_state)
        encoded_next_state = self.state_encoder.encode(preprocessed_next_state)
        
        reward = self.reward_calculator.calculate_reward(next_state, action)
        
        done = self._check_termination(next_state)
        truncated = self._check_truncation(next_state)
        info = self._get_info(next_state)
        
        rl_metrics = {
            "reward": reward,
            "done": done,
            "truncated": truncated,
            # Add any other relevant metrics
        }
        self.data_manager.save_rl_metrics(rl_metrics)
        
        return encoded_next_state, reward, done, truncated, info
    
    # ... (other methods remain the same)

    def get_episode_data(self, episode: int) -> List[Dict[str, Any]]:
        """
        Retrieve data for a specific episode.

        Args:
            episode (int): The episode number to retrieve data for.

        Returns:
            List[Dict[str, Any]]: List of state dictionaries for the episode.
        """
        return self.data_manager.load_episode_data(episode)

    def get_rl_metrics(self, episode: int) -> Dict[str, float]:
        """
        Retrieve RL metrics for a specific episode.

        Args:
            episode (int): The episode number to retrieve metrics for.

        Returns:
            Dict[str, float]: Dictionary of metric names and values.
        """
        return self.data_manager.load_rl_metrics(episode)
    
