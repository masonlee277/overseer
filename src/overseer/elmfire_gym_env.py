# src/overseer/rl/envs/elmfire_gym_env.py

import gym
import numpy as np
from typing import Dict, Any, Tuple, List

from overseer.config.config import OverseerConfig
from overseer.elmfire.simulation_manager import SimulationManager
from overseer.data.data_manager import DataManager
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.rl.rewards.reward_manager import RewardManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.rl.spaces.observation_space import ObservationSpace
from overseer.rl.utils.state_encoder import StateEncoder
from overseer.rl.utils.action_decoder import ActionDecoder
from overseer.utils.logging import OverseerLogger
from overseer.core.state_manager import StateManager
from overseer.core.state import State


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
        self.data_manager = DataManager(self.config)
        self.geospatial_manager = GeoSpatialManager(self.config)
        self.config_manager = ElmfireConfigManager(self.config)
        self.state_manager = StateManager(self.config, self.data_manager)
        
        self.sim_manager = SimulationManager(self.config, self.config_manager, self.data_manager)
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
        self.config_manager.initialize()
        self.state_manager.reset()
        
        initial_state = self.sim_manager.run_simulation()
        self.state_manager.update_state(initial_state)
        
        self.current_episode += 1
        self.current_step = 0
        
        encoded_state = self.state_encoder.encode(self.state_manager.get_current_state().get_raw_data())
        
        self.logger.info("Environment reset complete")
        return encoded_state, {}

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.logger.info(f"Taking step with action: {action}")
        
        decoded_action = self.action_decoder.decode(action)
        simulation_results = self.sim_manager.apply_action(decoded_action)
        
        self.state_manager.update_state(simulation_results)
        current_state = self.state_manager.get_current_state()
        
        encoded_next_state = self.state_encoder.encode(current_state.get_raw_data())
        
        reward = self.reward_manager.calculate_reward(current_state)
        
        done = self._check_termination(current_state)
        truncated = self._check_truncation(current_state)
        
        info = self._get_info(current_state)
        
        rl_metrics = {
            "reward": reward,
            "done": done,
            "truncated": truncated,
        }
        self.data_manager.update_rl_metrics(self.current_episode, self.current_step, rl_metrics)
        
        self.current_step += 1
        
        self.logger.info(f"Step complete. Reward: {reward}, Done: {done}, Truncated: {truncated}")
        return encoded_next_state, reward, done, truncated, info


    def _check_termination(self, state: State) -> bool:
        self.logger.debug("Checking termination condition")
        total_area = self.config.get('total_area', 10000)
        return state.get_basic_stats()['burned_area'] > 0.5 * total_area
    
    def _check_truncation(self, state: State) -> bool:
        self.logger.debug("Checking truncation condition")
        return state.get_raw_data()['timestamp'] > 24 * 3600
    
    def _get_info(self, state: State) -> Dict[str, Any]:
        self.logger.debug("Gathering additional state information")
        return {
            "fire_growth_rate": self.state_manager.get_fire_growth_rate(3600),
            "resources_used": sum(state.get_raw_data()['resources_deployed'].values()),
            "current_wind_speed": state.get_raw_data()['wind_speed'],
            "current_wind_direction": state.get_raw_data()['wind_direction'],
        }

    def get_episode_data(self, episode: int) -> List[Dict[str, Any]]:
        self.logger.info(f"Retrieving data for episode {episode}")
        return self.data_manager.get_episode_data(episode)

    def get_rl_metrics(self, episode: int) -> Dict[str, float]:
        self.logger.info(f"Retrieving RL metrics for episode {episode}")
        return self.data_manager.get_rl_metrics(episode)
    
    def close(self):
        self.logger.info("Closing ElmfireGymEnv and cleaning up resources")
        self.data_manager.