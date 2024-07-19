# src/overseer/rl/envs/elmfire_gym_env.py

import gym
import numpy as np
from typing import Dict, Any, Tuple, List

from overseer.config.config import OverseerConfig
from overseer.elmfire.simulation_manager import SimulationManager
from overseer.core.state_manager import StateManager
from overseer.rl.rewards.reward_manager import RewardManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.rl.spaces.observation_space import ObservationSpace
from overseer.rl.utils.state_encoder import StateEncoder
from overseer.rl.utils.action_decoder import ActionDecoder
from overseer.utils.logging import OverseerLogger
from overseer.elmfire.config_manager import ConfigurationManager

class ElmfireGymEnv(gym.Env):
    """
    A Gym-like environment for the ELMFIRE simulator.
    
    This class wraps the ELMFIRE simulator and provides a standard Gym-like interface
    for reinforcement learning agents to interact with the simulator.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the ElmfireGymEnv.
        """
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.logger.info("Initializing ElmfireGymEnv")
        
        self.config = OverseerConfig(config_path)
        self.config_manager = ConfigurationManager(self.config)
        self.state_manager = StateManager(self.config)
        
        self.sim_manager = SimulationManager(self.config)
        self.reward_manager = RewardManager(self.config)
        
        self.observation_space = ObservationSpace(self.config).space
        self.action_space = ActionSpace(self.config).space
        
        self.state_encoder = StateEncoder(self.config)
        self.action_decoder = ActionDecoder(self.config)
        
        self.current_episode = 0
        self.current_step = 0
        
        self.logger.info("ElmfireGymEnv initialized successfully")
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        """
        self.logger.info("Resetting environment")
        self.config_manager.reset_to_default()
        self.state_manager._initialize_from_config()
        
        initial_state = self.sim_manager.run_simulation()
        self.state_manager.update_state(initial_state)
        
        self.current_episode += 1
        self.current_step = 0
        
        encoded_state = self.state_encoder.encode(self.state_manager.get_current_state())
        
        self.logger.info("Environment reset complete")
        return encoded_state, {}
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment given an action.
        """
        self.logger.info(f"Taking step with action: {action}")
        
        # Decode the action
        decoded_action = self.action_decoder.decode(action)
        
        # Apply the action and run the simulation
        simulation_results = self.sim_manager.apply_action(decoded_action)
        
        # Process the simulation results and update the state
        self.data_manager.process_simulation_results(simulation_results)
        current_state = self.data_manager.get_current_state()
        self.state_manager.update_state(current_state)
        
        # Encode the state for the RL agent
        encoded_next_state = self.state_encoder.encode(current_state)
        
        # Calculate the reward
        reward = self.reward_manager.calculate_reward(self.state_manager)
        
        # Check for episode termination or truncation
        done = self._check_termination(current_state)
        truncated = self._check_truncation(current_state)
        
        # Get additional info
        info = self._get_info(current_state)
        
        # Update RL metrics
        rl_metrics = {
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }
        self.data_manager.update_rl_metrics(self.current_episode, self.current_step, rl_metrics)
        
        # Increment step counter
        self.current_step += 1
        
        self.logger.info(f"Step complete. Reward: {reward}, Done: {done}, Truncated: {truncated}")
        return encoded_next_state, reward, done, truncated, info

    
    def _check_termination(self, state: Dict[str, Any]) -> bool:
        """Check if the episode should terminate."""
        self.logger.debug("Checking termination condition")
        # Example termination condition: fire has burned more than 50% of the area
        total_area = self.config.get('total_area', 10000)  # Default 10000 hectares
        return state['burned_area'] > 0.5 * total_area
    
    def _check_truncation(self, state: Dict[str, Any]) -> bool:
        """Check if the episode should be truncated."""
        self.logger.debug("Checking truncation condition")
        # Example truncation condition: simulation has run for more than 24 hours
        return state['current_timestamp'] > 24 * 3600  # 24 hours in seconds
    
    def _get_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Get additional information about the current state."""
        self.logger.debug("Gathering additional state information")
        return {
            "fire_growth_rate": self.state_manager.get_fire_growth_rate(3600),  # Growth rate over last hour
            "resources_used": sum(state['resources_deployed'].values()),
            "current_wind_speed": state['wind_speed'],
            "current_wind_direction": state['wind_direction'],
        }

    def get_episode_data(self, episode: int) -> List[Dict[str, Any]]:
        """
        Retrieve data for a specific episode.
        """
        self.logger.info(f"Retrieving data for episode {episode}")
        # This would need to be implemented in StateManager
        return self.state_manager.get_episode_data(episode)

    def get_rl_metrics(self, episode: int) -> Dict[str, float]:
        """
        Retrieve RL metrics for a specific episode.
        """
        self.logger.info(f"Retrieving RL metrics for episode {episode}")
        # This would need to be implemented in StateManager
        return self.state_manager.get_rl_metrics(episode)
    
    def close(self):
        """
        Clean up resources used by the environment.
        """
        self.logger.info("Closing ElmfireGymEnv and cleaning up resources")
        self.state_manager._save_state()  # Ensure final state is saved
        super().close()