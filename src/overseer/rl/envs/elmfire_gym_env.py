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
        self.data_manager = DataManager(self.config)
        self.config_manager = ElmfireConfigManager(self.config, self.data_manager)

        self.sim_manager = SimulationManager(
            self.config, 
            self.config_manager, 
            self.data_manager
        )
        #####################################



        self.reward_manager = RewardManager(self.config, self.data_manager)
        
        self.observation_space = ObservationSpace(self.config).space
        self.action_space = ActionSpace(self.config, self.data_manager)

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
        
        initial_state = self.sim_manager.get_state()
        
        self.current_episode += 1
        self.current_step = 0
        
        # Create an EpisodeStep for the initial state
        episode_step = EpisodeStep(
            step=self.current_step,
            state=initial_state,
            action=None,  # No action taken yet
            reward=0.0,
            next_state=initial_state,
            done=False
        )
        
        #encoded_state = self.state_encoder.encode(self.data_manager.get_current_state().get_raw_data())
        encoded_state = self.data_manager.state_to_array()
        self.logger.info("Environment reset complete")
        #log the type of the encoded state
        self.logger.info(f"Type of encoded state: {type(encoded_state)}")
        return encoded_state
    

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.logger.info(f"Taking step with action: {action}")
        
        # Convert the raw action to an Action object
        action_obj = self.action_space.create_action(action)
        self.logger.info(f"Converted action: {action_obj}")

        # Create an EpisodeStep for the current action
        current_state = self.data_manager.get_current_state()
        
        # Apply the action and get simulation results
        next_state, done = self.sim_manager.apply_action(action_obj)



        
        # Encode the next state
        #encoded_next_state = self.state_encoder.encode(next_state.get_raw_data())
        #convert next state to array
        encoded_next_state = self.data_manager.state_to_array(next_state)
        # Calculate the reward
        reward = self.reward_manager.calculate_reward(next_state)
        

        self.data_manager.add_step_to_current_episode(
            state=current_state,
            action=action_obj,
            reward=reward,
            next_state=next_state,
            done=done
        )
        # Check termination and truncation conditions
        
        # Gather additional info
        info = self._get_info(next_state)
        
        # Update RL metrics
        rl_metrics = {
            "reward": reward,
            "terminated": done,
        }
        terminated=done

        self.data_manager.update_rl_metrics(self.current_episode, self.current_step, rl_metrics)
        self.current_step += 1
        
        self.logger.info(f"Step complete. Reward: {reward}, Terminated: {terminated}")
        return encoded_next_state, reward, terminated, info

    def _get_info(self, state: SimulationState) -> Dict[str, Any]:
        self.logger.debug("Gathering additional state information")
        current_state = self.data_manager.get_current_state()
        
        if current_state is None:
            self.logger.warning("No current state available for info gathering")
            return {}

        return {
            "burned_area": current_state.metrics.burned_area,
            "fire_perimeter_length": current_state.metrics.fire_perimeter_length,
            "containment_percentage": current_state.metrics.containment_percentage,
            "execution_time": current_state.metrics.execution_time,
            "fire_growth_rate": self.data_manager.get_fire_growth_rate(3600),
            "resources_used": sum(current_state.resources.values()),
            "current_wind_speed": current_state.weather.get('wind_speed', 0),
            "current_wind_direction": current_state.weather.get('wind_direction', 0),
        }

    def get_episode_data(self, episode: int) -> List[Dict[str, Any]]:
        self.logger.info(f"Retrieving data for episode {episode}")
        return self.data_manager.get_episode_data(episode)

    def get_rl_metrics(self, episode: int) -> Dict[str, float]:
        self.logger.info(f"Retrieving RL metrics for episode {episode}")
        return self.data_manager.get_rl_metrics(episode)
    
    def close(self):
        self.logger.info("Closing ElmfireGymEnv and cleaning up resources")
        self.data_manager.reset()
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
    