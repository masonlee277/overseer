# src/overseer/rl/rewards/reward_strategies.py
"""
Reward Strategies for the ELMFIRE Reinforcement Learning Environment

This module defines various reward calculation strategies for the ELMFIRE simulation environment.
These strategies determine how the agent is rewarded or penalized based on its actions and the
resulting state of the environment.

Interaction between StateManager, State, and GeoSpatialManager:

1. StateManager:
   - Manages the current state and history of states in the simulation.
   - Provides methods to retrieve states at specific timestamps.
   - Acts as an interface between the reward strategies and the underlying data.

2. State:
   - Represents a single state in the ELMFIRE simulation.
   - Contains raw data and pre-calculated basic statistics about the fire state.
   - Provides methods to access specific state information.

3. GeoSpatialManager:
   - Handles geospatial calculations and analysis.
   - Provides methods for complex spatial operations related to fire behavior and containment.

Reward Calculation Process:
1. The reward strategy receives a StateManager object.
2. It retrieves the current state and relevant historical states using StateManager methods.
3. Basic statistics are obtained directly from the State objects.
4. For more complex calculations, the reward strategy may use GeoSpatialManager methods,
   which are typically accessed through the DataManager (referenced in the State object).
5. The reward strategy combines various factors (e.g., area change, resource efficiency,
   containment progress) to calculate the final reward.

This architecture allows for flexible and modular reward calculation, where different
strategies can be easily implemented and swapped without changing the core simulation logic.
It also enables the incorporation of complex geospatial analysis into the reward calculation,
providing a more comprehensive evaluation of the agent's performance in wildfire management.
"""


from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from overseer.core.state_manager import StateManager
from overseer.core.state import State

class RewardStrategy(ABC):
    """Abstract base class for reward calculation strategies."""
    
    @abstractmethod
    def calculate_reward(self, state_manager: StateManager) -> float:
        """
        Calculate the reward based on the current and historical states.

        Args:
            state_manager (StateManager): The state manager containing current and historical state information.

        Returns:
            float: The calculated reward.
        """
        pass


#################################################
#################################################
class SimpleAreaRewardStrategy(RewardStrategy):
    def calculate_reward(self, state_manager: StateManager) -> float:
        current_state = state_manager.get_current_state()
        if current_state is None:
            return 0.0

        previous_state = state_manager.get_state_at_time(current_state.timestamp - 3600)  # 1 hour ago
        
        if previous_state is None:
            return 0.0  # No previous state, return neutral reward
        
        current_area = current_state.get_basic_stats()['burned_area']
        previous_area = previous_state.get_basic_stats()['burned_area']
        return -(current_area - previous_area)  # Negative reward for increase in burned area

class ResourceAwareRewardStrategy(RewardStrategy):
    def calculate_reward(self, state_manager: StateManager) -> float:
        current_state = state_manager.get_current_state()
        if current_state is None:
            return 0.0

        previous_state = state_manager.get_state_at_time(current_state.timestamp - 3600)  # 1 hour ago
        
        if previous_state is None:
            return 0.0  # No previous state, return neutral reward
        
        area_change = current_state.get_basic_stats()['burned_area'] - previous_state.get_basic_stats()['burned_area']
        resource_usage = (current_state.get_basic_stats()['total_resources_deployed'] - 
                          previous_state.get_basic_stats()['total_resources_deployed'])
        
        # Calculate fire growth rate
        growth_rate = state_manager.get_fire_growth_rate(3600)  # Over the last hour
        
        # Combine factors to calculate reward
        area_weight = -1.0
        resource_weight = -0.1
        growth_rate_weight = -10.0
        
        reward = (
            area_weight * area_change +
            resource_weight * resource_usage +
            growth_rate_weight * growth_rate
        )
        
        return reward

class ComplexRewardStrategy(RewardStrategy):
    def calculate_reward(self, state_manager: StateManager) -> float:
        current_state = state_manager.get_current_state()
        if current_state is None:
            return 0.0
        
        # Calculate short-term change (last hour)
        state_1h_ago = state_manager.get_state_at_time(current_state.timestamp - 3600)
        short_term_area_change = self._calculate_area_change(current_state, state_1h_ago)
        
        # Calculate medium-term change (last 6 hours)
        state_6h_ago = state_manager.get_state_at_time(current_state.timestamp - 21600)
        medium_term_area_change = self._calculate_area_change(current_state, state_6h_ago)
        
        # Calculate long-term change (last 24 hours)
        state_24h_ago = state_manager.get_state_at_time(current_state.timestamp - 86400)
        long_term_area_change = self._calculate_area_change(current_state, state_24h_ago)
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(current_state, state_1h_ago)
        
        # Calculate fire containment progress
        containment_progress = self._calculate_containment_progress(current_state, state_6h_ago)
        
        # Combine factors to calculate reward
        short_term_weight = -1.0
        medium_term_weight = -0.5
        long_term_weight = -0.2
        resource_efficiency_weight = 0.3
        containment_progress_weight = 0.5
        
        reward = (
            short_term_weight * short_term_area_change +
            medium_term_weight * medium_term_area_change +
            long_term_weight * long_term_area_change +
            resource_efficiency_weight * resource_efficiency +
            containment_progress_weight * containment_progress
        )
        
        return reward
    
    def _calculate_area_change(self, current_state: State, previous_state: Optional[State]) -> float:
        if previous_state is None:
            return 0.0
        return current_state.get_basic_stats()['burned_area'] - previous_state.get_basic_stats()['burned_area']
    
    def _calculate_resource_efficiency(self, current_state: State, previous_state: Optional[State]) -> float:
        if previous_state is None:
            return 0.0
        area_change = self._calculate_area_change(current_state, previous_state)
        resource_change = (current_state.get_basic_stats()['total_resources_deployed'] - 
                           previous_state.get_basic_stats()['total_resources_deployed'])
        return -area_change / (resource_change + 1)  # Add 1 to avoid division by zero
    
    def _calculate_containment_progress(self, current_state: State, previous_state: Optional[State]) -> float:
        if previous_state is None:
            return 0.0
        current_perimeter = current_state.get_basic_stats()['fire_perimeter_length']
        previous_perimeter = previous_state.get_basic_stats()['fire_perimeter_length']
        return (previous_perimeter - current_perimeter) / previous_perimeter
    

class MockRewardStrategy(RewardStrategy):
    """A mock reward strategy for testing and validation purposes."""

    def __init__(self, fixed_reward: float = 1.0):
        self.fixed_reward = fixed_reward

    def calculate_reward(self, state_manager: StateManager) -> float:
        current_state = state_manager.get_current_state()
        if current_state is None:
            return 0.0
        
        # Simulate some calculation based on the current state
        burned_area = current_state.get_basic_stats().get('burned_area', 0)
        resources_used = current_state.get_basic_stats().get('total_resources_deployed', 0)
        
        # Return a combination of the fixed reward and state-based calculation
        return self.fixed_reward - (burned_area * 0.01) + (resources_used * 0.1)