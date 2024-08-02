import os
import sys
from typing import Dict, Any
import random

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.rl.rewards.reward_strategies import RewardStrategy, SimpleAreaRewardStrategy, ResourceAwareRewardStrategy, MockRewardStrategy
from overseer.data.data_manager import DataManager
from overseer.core.models import SimulationState

class RewardManager:
    """
    Manages the calculation of rewards for the ELMFIRE Reinforcement Learning environment.

    This class acts as a facade for reward calculation, handling the interaction
    with the DataManager and applying the chosen reward strategy. It is designed
    to work seamlessly with the ElmfireGymEnv class, providing a flexible and
    modular approach to reward calculation in the RL training process.

    Attributes:
        config (OverseerConfig): Configuration object shared with ElmfireGymEnv.
        logger (OverseerLogger): Logger instance for tracking reward calculations.
        reward_strategy (RewardStrategy): The current reward calculation strategy.
        data_manager (DataManager): The data manager instance for accessing simulation states.

    Methods:
        calculate_reward(state: SimulationState) -> float:
            Called by ElmfireGymEnv in its step function to compute the reward.

        set_reward_strategy(strategy: RewardStrategy) -> None:
            Allows for dynamic switching of reward strategies during training.
    """

    def __init__(self, config: OverseerConfig, data_manager: DataManager):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_manager = data_manager
        self.reward_strategy = self._init_reward_strategy()

    def _init_reward_strategy(self) -> RewardStrategy:
        """Initialize the reward strategy based on configuration."""
        strategy_name = self.config.get('reward_strategy', 'simple')
        if strategy_name == 'simple':
            return SimpleAreaRewardStrategy()
        elif strategy_name == 'resource_aware':
            return ResourceAwareRewardStrategy()
        else:
            self.logger.warning(f"Unknown reward strategy '{strategy_name}'. Using SimpleAreaRewardStrategy.")
            return SimpleAreaRewardStrategy()

    def set_reward_strategy(self, strategy: RewardStrategy) -> None:
        """Set a new reward calculation strategy."""
        self.reward_strategy = strategy
        self.logger.info(f"Reward strategy set to {strategy.__class__.__name__}")

    def calculate_reward(self, state: SimulationState) -> float:
        """
        Calculate the reward for the current state.

        This method uses the SimulationState to access current state
        information and applies the current reward strategy.

        Args:
            state (SimulationState): The current simulation state.

        Returns:
            float: The calculated reward.
        """
        try:
            reward = self.reward_strategy.calculate_reward(state)
            self.logger.info(f"Calculated reward: {reward}")
            return reward
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0  # Return a default reward in case of error

def main():
    # Setup logging
    logger = OverseerLogger().get_logger("RewardManagerTest")
    logger.info("Starting RewardManager test")

    # Create a mock config
    class MockConfig:
        def get(self, key, default=None):
            config_dict = {
                'reward_strategy': 'mock',
                'logging_level': 'INFO'
            }
            return config_dict.get(key, default)

    config = MockConfig()

    # Create a mock DataManager
    class MockDataManager:
        def __init__(self):
            self.current_state = None

        def get_current_state(self):
            return self.current_state

        def update_state(self, state):
            self.current_state = state

    data_manager = MockDataManager()

    # Initialize RewardManager
    reward_manager = RewardManager(config, data_manager)
    logger.info(f"Initialized RewardManager with strategy: {reward_manager.reward_strategy.__class__.__name__}")

    # Test reward calculation with different strategies
    strategies_to_test = [
        MockRewardStrategy(fixed_reward=2.0),
    ]

    for strategy in strategies_to_test:
        reward_manager.set_reward_strategy(strategy)
        logger.info(f"Testing strategy: {strategy.__class__.__name__}")

        # Simulate a series of states and calculate rewards
        for i in range(5):
            timestamp = i * 3600  # 1-hour intervals
            burned_area = random.uniform(100, 1000)
            resources_deployed = {'firefighters': random.randint(10, 50), 'trucks': random.randint(2, 10)}
            
            mock_state = SimulationState(
                timestamp=timestamp,
                fire_intensity=None,  # Add a mock fire intensity array if needed
                burned_area=burned_area,
                fire_perimeter_length=random.uniform(50, 200),
                containment_percentage=random.uniform(0, 100),
                resources=resources_deployed,
                weather={'wind_speed': random.uniform(0, 20), 'wind_direction': random.uniform(0, 360)}
            )
            
            data_manager.update_state(mock_state)
            reward = reward_manager.calculate_reward(mock_state)
            logger.info(f"  Timestamp: {timestamp}, Burned Area: {burned_area:.2f}, "
                        f"Resources Deployed: {resources_deployed}, Reward: {reward:.4f}")

        logger.info(f"Finished testing {strategy.__class__.__name__}\n")

    # Test error handling
    logger.info("Testing error handling")

    def faulty_calculate_reward(self, state):
        raise Exception("Simulated error in reward calculation")

    faulty_strategy = MockRewardStrategy()
    faulty_strategy.calculate_reward = faulty_calculate_reward
    reward_manager.set_reward_strategy(faulty_strategy)
    reward = reward_manager.calculate_reward(data_manager.get_current_state())
    logger.info(f"Reward calculated with faulty strategy: {reward}")

    logger.info("RewardManager test completed")

if __name__ == "__main__":
    main()