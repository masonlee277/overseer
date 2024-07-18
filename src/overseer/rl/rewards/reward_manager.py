from typing import Dict, Any
import numpy as np
from overseer.config import OverseerConfig
from overseer.elmfire.data_manager import DataManager
from overseer.utils.logging import OverseerLogger
from .reward_strategies import RewardStrategy, SimpleAreaRewardStrategy, ResourceAwareRewardStrategy



class RewardManager:


    """
    Manages the calculation of rewards for the RL environment.

    This class acts as a facade for reward calculation, handling the interaction
    with the DataManager and applying the chosen reward strategy.

    Attributes:
        data_manager (DataManager): Instance of DataManager for data retrieval.
        config (OverseerConfig): Configuration object.
        logger (OverseerLogger): Logger instance.
        reward_strategy (RewardStrategy): The current reward calculation strategy.
    """

    def __init__(self, data_manager: DataManager, config: OverseerConfig):
        """
        Initialize the RewardManager.

        Args:
            data_manager (DataManager): Instance of DataManager.
            config (OverseerConfig): Configuration object.
        """
        self.data_manager = data_manager
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.reward_strategy = self._init_reward_strategy()

    def _init_reward_strategy(self) -> RewardStrategy:
        """
        Initialize the reward strategy based on configuration.

        Returns:
            RewardStrategy: The initialized reward strategy.
        """
        strategy_name = self.config.get('reward_strategy', 'simple')
        if strategy_name == 'simple':
            return SimpleAreaRewardStrategy()
        elif strategy_name == 'resource_aware':
            return ResourceAwareRewardStrategy()
        else:
            self.logger.warning(f"Unknown reward strategy '{strategy_name}'. Using SimpleAreaRewardStrategy.")
            return SimpleAreaRewardStrategy()

    def set_reward_strategy(self, strategy: RewardStrategy) -> None:
        """
        Set a new reward calculation strategy.

        Args:
            strategy (RewardStrategy): The new reward strategy to use.
        """
        self.reward_strategy = strategy
        self.logger.info(f"Reward strategy set to {strategy.__class__.__name__}")

    def calculate_reward(self, current_episode: int, current_step: int) -> float:
        """
        Calculate the reward for the current state.

        This method retrieves the necessary data from the DataManager and applies
        the current reward strategy.

        Args:
            current_episode (int): The current episode number.
            current_step (int): The current step number.

        Returns:
            float: The calculated reward.
        """
        try:
            current_state = self.data_manager.load_simulation_state(current_episode, current_step)
            previous_state = self.data_manager.load_simulation_state(current_episode, current_step - 1) if current_step > 0 else current_state

            reward = self.reward_strategy.calculate_reward(current_state, previous_state)
            self.logger.info(f"Calculated reward: {reward}")
            return reward
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0  # Return a default reward in case of error

    def get_state_info(self, episode: int, step: int) -> Dict[str, Any]:
        """
        Retrieve relevant state information for reward calculation.

        This method gets both the geospatial data and other state information
        from the DataManager.

        Args:
            episode (int): The episode number.
            step (int): The step number.

        Returns:
            Dict[str, Any]: A dictionary containing relevant state information.
        """
        try:
            state = self.data_manager.load_simulation_state(episode, step)
            geospatial_data, _ = self.data_manager.load_geospatial_data(f"fire_perimeter_episode_{episode}_step_{step}.tif")
            
            state_info = {
                'burned_area': np.sum(geospatial_data > 0),  # Assuming positive values indicate burned area
                'resources_used': state.get('resources_used', 0),
                'time_elapsed': state.get('time_elapsed', 0),
                # Add any other relevant information from the state
            }
            return state_info
        except Exception as e:
            self.logger.error(f"Error retrieving state info: {str(e)}")
            return {}