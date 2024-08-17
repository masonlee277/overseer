import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List
import traceback

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
        
        # Initialize/reset ConfigManager
        self.config_manager.initialize()
        
        # Create an initial SimulationState
        initial_state = self.sim_manager.create_initial_state()
        
        # Reset DataManager and update with initial state
        self.data_manager.reset()
        self.data_manager.update_state(initial_state)

        self.current_episode += 1
        self.current_step = 0
        
        encoded_state = self.data_manager.state_to_array()
        self.logger.info("Environment reset complete")
        self.logger.info(f"Type of encoded state: {type(encoded_state)}")
        return encoded_state, {}  # Return empty dict as info
        

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.logger.info(f"Taking step with action: {action}")
        
        # Convert the raw action to an Action object
        action_obj = self.action_space.create_action(action)
        self.logger.info(f"Converted action: {action_obj}")

        # Create an EpisodeStep for the current action
        current_state = self.data_manager.get_current_state()
        assert current_state is not None, "Current state is None"
        # Apply the action and get simulation results
        next_state, done = self.sim_manager.run_simulation(action_obj)


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
        }

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
    


def main():
    logger = OverseerLogger().get_logger("ElmfireGymEnv_Main")

    config_path = "src\overseer\config\elmfire_config.yaml"
    env = ElmfireGymEnv(config_path)

    try:
        logger.info("Initializing ElmfireGymEnv")

        logger.info("Initializing ElmfireGymEnv")
        env = ElmfireGymEnv(config_path)
        
        # Test reset functionality
        logger.info("Testing reset functionality")
        observation, _ = env.reset()
        logger.info(f"Initial observation shape: {observation.shape}")
        
        # Test action space
        logger.info("Testing action space")
        sample_action = env.action_space.sample()
        logger.info(f"Sample action: {sample_action}")
        
        # Test observation space
        logger.info("Testing observation space")
        logger.info(f"Observation space: {env.observation_space}")
        
        # Test single step
        logger.info("Testing single step")
        next_observation, reward, terminated, info = env.step(sample_action)
        logger.info(f"Step result - Reward: {reward}, Terminated: {terminated}")
        logger.info(f"Info: {info}")
        

        
        # Test RL metrics retrieval
        num_episodes = 5
        max_steps_per_episode = 5

        for episode in range(num_episodes):
            logger.info(f"Starting Episode {episode + 1}")
            try:
                observation, _ = env.reset()
                episode_reward = 0
                
                for step in range(max_steps_per_episode):
                    action = env.action_space.sample()  # Random action
                    logger.debug(f"Episode {episode + 1}, Step {step + 1}: Sampled action = {action}")
                    
                    try:
                        observation, reward, terminated, info = env.step(action)
                        episode_reward += reward
                        
                        logger.info(f"Episode {episode + 1}, Step {step + 1}: Reward = {reward}, Info = {info}")
                        
                        if terminated:
                            logger.info(f"Episode {episode + 1} finished after {step + 1} steps")
                            break
                    except Exception as step_error:
                        logger.error(f"Error during step {step + 1} of episode {episode + 1}")
                        logger.error(traceback.format_exc())
                        break
                
                logger.info(f"Episode {episode + 1} total reward: {episode_reward}")
                
                # Get and print episode data and RL metrics
                # episode_data = env.get_episode_data(episode)
                # rl_metrics = env.get_rl_metrics(episode)
                # logger.info(f"Episode {episode + 1} data: {episode_data}")
                # logger.info(f"Episode {episode + 1} RL metrics: {rl_metrics}")
            
            except Exception as episode_error:
                logger.error(f"Error during episode {episode + 1}")
                logger.error(traceback.format_exc())
        
        logger.info("Closing environment")
        env.close()

    except Exception as e:
        logger.error("Error in main execution")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()