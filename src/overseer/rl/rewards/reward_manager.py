# src/overseer/rl/rewards/reward_manager.py
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from typing import Dict, Any
import random

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.core.state_manager import StateManager
from overseer.rl.rewards.reward_strategies import RewardStrategy, SimpleAreaRewardStrategy, ResourceAwareRewardStrategy, MockRewardStrategy

class RewardManager:
    """
    Manages the calculation of rewards for the ELMFIRE Reinforcement Learning environment.

    This class acts as a facade for reward calculation, handling the interaction
    with the StateManager and applying the chosen reward strategy. It is designed
    to work seamlessly with the ElmfireGymEnv class, providing a flexible and
    modular approach to reward calculation in the RL training process.

    Interaction with ElmfireGymEnv:
    1. Initialization: The RewardManager is initialized in the ElmfireGymEnv constructor
       using the same configuration object. This ensures consistency in reward strategies
       across the environment.

    2. Step Function: In the ElmfireGymEnv's step function, after applying an action
       and updating the state, the calculate_reward method of RewardManager is called
       with the current StateManager. This calculates the reward for the current step.

    3. Flexibility: The RewardManager allows for dynamic switching of reward strategies
       during training, if needed, without modifying the core environment logic.

    4. Error Handling: The RewardManager includes error handling to ensure that the
       RL training process doesn't halt due to reward calculation errors, returning
       a default reward in case of exceptions.

    Attributes:
        config (OverseerConfig): Configuration object shared with ElmfireGymEnv.
        logger (OverseerLogger): Logger instance for tracking reward calculations.
        reward_strategy (RewardStrategy): The current reward calculation strategy.

    Methods:
        calculate_reward(state_manager: StateManager) -> float:
            Called by ElmfireGymEnv in its step function to compute the reward.

        set_reward_strategy(strategy: RewardStrategy) -> None:
            Allows for dynamic switching of reward strategies during training.

    Usage in ElmfireGymEnv:
        def step(self, action):
            # ... apply action and update state ...
            reward = self.reward_manager.calculate_reward(self.state_manager)
            # ... determine if episode is done, return step information ...

    This design allows for easy extension and modification of reward calculation
    logic without changing the core environment code, promoting modularity and
    flexibility in the RL training process.
    """

    def __init__(self, config: OverseerConfig):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
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

    def calculate_reward(self, state_manager: StateManager) -> float:
        """
        Calculate the reward for the current state.

        This method uses the StateManager to access current and historical state
        information and applies the current reward strategy.

        Args:
            state_manager (StateManager): The state manager containing current and historical state information.

        Returns:
            float: The calculated reward.
        """
        try:
            reward = self.reward_strategy.calculate_reward(state_manager)
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

    # Initialize RewardManager
    reward_manager = RewardManager(config)
    logger.info(f"Initialized RewardManager with strategy: {reward_manager.reward_strategy.__class__.__name__}")

    # Create a mock StateManager
    class MockStateManager:
        def __init__(self):
            self.current_timestamp = 0
            self.states = {}

        def get_current_state(self):
            return self.states.get(self.current_timestamp)

        def get_state_at_time(self, timestamp):
            return self.states.get(timestamp)

        def add_state(self, timestamp, burned_area, resources_deployed):
            class MockState:
                def get_basic_stats(self):
                    return {
                        'burned_area': burned_area,
                        'total_resources_deployed': resources_deployed
                    }
            self.states[timestamp] = MockState()
            self.current_timestamp = timestamp

    state_manager = MockStateManager()

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
            resources_deployed = random.uniform(10, 100)
            state_manager.add_state(timestamp, burned_area, resources_deployed)

            reward = reward_manager.calculate_reward(state_manager)
            logger.info(f"  Timestamp: {timestamp}, Burned Area: {burned_area:.2f}, "
                        f"Resources Deployed: {resources_deployed:.2f}, Reward: {reward:.4f}")

        logger.info(f"Finished testing {strategy.__class__.__name__}\n")

    # Test error handling
    logger.info("Testing error handling")

    def faulty_calculate_reward(self, state_manager):
        raise Exception("Simulated error in reward calculation")

    faulty_strategy = MockRewardStrategy()
    faulty_strategy.calculate_reward = faulty_calculate_reward
    reward_manager.set_reward_strategy(faulty_strategy)
    reward = reward_manager.calculate_reward(state_manager)
    logger.info(f"Reward calculated with faulty strategy: {reward}")

    logger.info("RewardManager test completed")

if __name__ == "__main__":
    main()