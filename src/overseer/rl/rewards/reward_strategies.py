import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from abc import ABC, abstractmethod
from typing import Dict, Any
from overseer.core.models import SimulationState
import numpy as np

class RewardStrategy(ABC):
    @abstractmethod
    def calculate_reward(self, state: SimulationState) -> float:
        pass

class SimpleAreaRewardStrategy(RewardStrategy):
    def calculate_reward(self, state: SimulationState) -> float:
        # Negative reward based on burned area
        return -state.burned_area

class ResourceAwareRewardStrategy(RewardStrategy):
    def calculate_reward(self, state: SimulationState) -> float:
        # Negative reward based on burned area and resources used
        total_resources = sum(state.resources.values())
        return -(state.burned_area + 0.1 * total_resources)

class ComplexRewardStrategy(RewardStrategy):
    def calculate_reward(self, state: SimulationState) -> float:
        # More complex reward considering multiple factors
        burned_area_penalty = -state.burned_area
        containment_bonus = state.containment_percentage
        resource_penalty = -0.1 * sum(state.resources.values())
        
        # Penalize high wind speeds as they increase fire spread risk
        wind_penalty = -0.5 * state.weather['wind_speed']
        
        return burned_area_penalty + containment_bonus + resource_penalty + wind_penalty

class MockRewardStrategy(RewardStrategy):
    def __init__(self, fixed_reward: float = 1.0):
        self.fixed_reward = fixed_reward

    def calculate_reward(self, state: SimulationState) -> float:
        return self.fixed_reward