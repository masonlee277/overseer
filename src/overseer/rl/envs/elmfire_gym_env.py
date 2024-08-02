import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List

from overseer.config.config import OverseerConfig
from overseer.elmfire.simulation_manager import SimulationManager
from overseer.data.data_manager import DataManager
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.rl.rewards.reward_manager import RewardManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.rl.spaces.observation_space import ObservationSpace
from overseer.rl.utils.state_encoder import StateEncoder
from overseer.utils.logging import OverseerLogger
from overseer.core.state import State
from overseer.core.models import SimulationState, EpisodeStep

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

        #####################################
        self.config_manager = ElmfireConfigManager(self.config)
        self.data_manager = DataManager(self.config)
        
        self.sim_manager = SimulationManager(
            self.config, 
            self.config_manager, 
            self.data_manager
        )
        #####################################



        self.reward_manager = RewardManager(self.config)
        
        self.observation_space = ObservationSpace(self.config).space
        self.action_space = ActionSpace(self.config).space
        
        self.state_encoder = StateEncoder(self.config)
        
        self.current_episode = 0
        self.current_step = 0
        
        self.logger.info("ElmfireGymEnv initialized successfully")
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to its initial state.
        """
        super().reset(seed=seed)
        self.logger.info("Resetting environment")
        self.config_manager.initialize()
        self.data_manager.reset()
        
        initial_state = self.sim_manager.run_simulation()
        self.data_manager.save_simulation_state(initial_state)
        
        self.current_episode += 1
        self.current_step = 0
        
        # Create an EpisodeStep for the initial state
        episode_step = EpisodeStep(
            step=self.current_step,
            state=initial_state,
            action=None,  # No action taken yet
            reward=0.0,
            next_state=initial_state,
            simulation_result=None,  # Will be filled after the first step
            done=False
        )
        
        encoded_state = self.state_encoder.encode(self.data_manager.get_current_state().get_raw_data())
        
        self.logger.info("Environment reset complete")
        return encoded_state, {}
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.logger.info(f"Taking step with action: {action}")
        
        # Create an EpisodeStep for the current action
        current_state = self.data_manager.get_current_state()
        episode_step = EpisodeStep(
            step=self.current_step,
            state=current_state,
            action=action,
            reward=0.0,  # Reward will be calculated after applying the action
            next_state=None,  # To be filled after applying the action
            simulation_result=None,  # To be filled after applying the action
            done=False
        )
        
        # Apply the action and get simulation results
        simulation_results = self.sim_manager.apply_action(action)
        
        # Update the episode step with the new state and simulation results
        episode_step.next_state = self.data_manager.get_current_state()
        episode_step.simulation_result = simulation_results
        
        # Save the simulation state
        self.data_manager.save_simulation_state(simulation_results)
        
        # Encode the next state
        encoded_next_state = self.state_encoder.encode(episode_step.next_state.get_raw_data())
        
        # Calculate the reward
        reward = self.reward_manager.calculate_reward(episode_step.next_state)
        
        # Check termination and truncation conditions
        terminated = self._check_termination(episode_step.next_state)
        truncated = self._check_truncation(episode_step.next_state)
        
        # Gather additional info
        info = self._get_info(episode_step.next_state)
        
        # Update RL metrics
        rl_metrics = {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        self.data_manager.update_rl_metrics(self.current_episode, self.current_step, rl_metrics)
        self.current_step += 1
        
        self.logger.info(f"Step complete. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        return encoded_next_state, reward, terminated, truncated, info


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
            "fire_growth_rate": self.data_manager.get_fire_growth_rate(3600),
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
        self.data_manager.close()
        self.sim_manager.cleanup()

    def render(self, mode='human'):
        """
        Render the environment.
        This method should be implemented to provide visualization of the environment state.
        """
        pass  # Implement rendering logic here

    def seed(self, seed=None):
        """
        Set the seed for this env's random number generator(s).
        """
        return []  # Return a list of seeds used in this env's random number generators
    