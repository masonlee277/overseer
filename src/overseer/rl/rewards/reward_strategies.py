from abc import ABC, abstractmethod
from typing import Dict, Any
from ..data.data_manager import DataManager
from ..config.overseer_config import OverseerConfig
from ..utils.logging import OverseerLogger

class RewardStrategy(ABC):
    """
    Abstract base class for reward calculation strategies.
    """
    @abstractmethod
    def calculate_reward(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> float:
        """
        Calculate the reward based on the current and previous states.

        Args:
            current_state (Dict[str, Any]): The current state of the simulation.
            previous_state (Dict[str, Any]): The previous state of the simulation.

        Returns:
            float: The calculated reward.
        """
        pass

class SimpleAreaRewardStrategy(RewardStrategy):
    """
    A simple reward strategy based on the change in burned area.
    """
    def calculate_reward(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> float:
        current_area = current_state.get('burned_area', 0)
        previous_area = previous_state.get('burned_area', 0)
        return -(current_area - previous_area)  # Negative reward for increase in burned area

class ResourceAwareRewardStrategy(RewardStrategy):
    """
    A more complex reward strategy that considers both burned area and resource usage.
    """
    def calculate_reward(self, current_state: Dict[str, Any], previous_state: Dict[str, Any]) -> float:
        area_change = current_state.get('burned_area', 0) - previous_state.get('burned_area', 0)
        resource_usage = current_state.get('resources_used', 0) - previous_state.get('resources_used', 0)
        return -(area_change + 0.1 * resource_usage)  # Penalize both area increase and resource usage
