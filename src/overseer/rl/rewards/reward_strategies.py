import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from abc import ABC, abstractmethod
from typing import Dict, Any
from overseer.core.models import SimulationState
import numpy as np


class RewardStrategy(ABC):
    """
    Abstract base class for reward calculation strategies.

    This class defines the interface for all reward strategies used in the ELMFIRE RL environment.
    """

    @abstractmethod
    def calculate_reward(self, state: SimulationState) -> float:
        """
        Calculate the reward based on the current simulation state.

        Args:
            state (SimulationState): The current state of the simulation.

        Returns:
            float: The calculated reward.
        """
        pass



class SimpleAreaRewardStrategy(RewardStrategy):
    def calculate_reward(self, state: SimulationState) -> float:
        # Negative reward based on burned area
        return -state.burned_area

class SimpleAreaRewardStrategy(RewardStrategy):
    """
    A simple reward strategy based on the burned area.

    This strategy calculates a negative reward proportional to the burned area.
    """

    def calculate_reward(self, state: SimulationState) -> float:
        """
        Calculate the reward based on the burned area.

        Args:
            state (SimulationState): The current state of the simulation.

        Returns:
            float: The calculated reward (negative value).
        """
        return -state.metrics.burned_area
    
class MockRewardStrategy(RewardStrategy):
    """
    A mock reward strategy for testing purposes.

    This strategy always returns a fixed reward value.
    """

    def __init__(self, fixed_reward: float = 1.0):
        """
        Initialize the MockRewardStrategy.

        Args:
            fixed_reward (float): The fixed reward value to return. Defaults to 1.0.
        """
        self.fixed_reward = fixed_reward

    def calculate_reward(self, state: SimulationState) -> float:
        """
        Return a fixed reward value, regardless of the state.

        Args:
            state (SimulationState): The current state of the simulation (unused).

        Returns:
            float: The fixed reward value.
        """
        return self.fixed_reward
    
class ResourceAwareRewardStrategy(RewardStrategy):
    """
    A reward strategy that considers both burned area and resource usage.

    This strategy calculates a negative reward based on the burned area and the total resources used.
    """

    def calculate_reward(self, state: SimulationState) -> float:
        """
        Calculate the reward based on burned area and resource usage.

        Args:
            state (SimulationState): The current state of the simulation.

        Returns:
            float: The calculated reward (negative value).
        """
        total_resources = sum(state.resources.values())
        return -(state.metrics.burned_area + 0.1 * total_resources)

class ComplexRewardStrategy(RewardStrategy):
    """
    A complex reward strategy that considers multiple factors.

    This strategy calculates a reward based on burned area, containment percentage,
    resource usage, and weather conditions.
    """

    def calculate_reward(self, state: SimulationState) -> float:
        """
        Calculate a complex reward based on multiple factors.

        Args:
            state (SimulationState): The current state of the simulation.

        Returns:
            float: The calculated reward.
        """
        burned_area_penalty = -state.metrics.burned_area
        containment_bonus = state.metrics.containment_percentage
        resource_penalty = -0.1 * sum(state.resources.values())
        
        # Penalize high wind speeds as they increase fire spread risk
        wind_penalty = -0.5 * state.weather.get('wind_speed', 0)
        
        return burned_area_penalty + containment_bonus + resource_penalty + wind_penalty
