# src/overseer/rl/rewards/reward_manager.py

from typing import Dict, Any
from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.core.state_manager import StateManager
from .reward_strategies import RewardStrategy, SimpleAreaRewardStrategy, ResourceAwareRewardStrategy

class RewardManager:
    """
    Manages the calculation of rewards for the RL environment.

    This class acts as a facade for reward calculation, handling the interaction
    with the StateManager and applying the chosen reward strategy.

    Attributes:
        config (OverseerConfig): Configuration object.
        logger (OverseerLogger): Logger instance.
        reward_strategy (RewardStrategy): The current reward calculation strategy.
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