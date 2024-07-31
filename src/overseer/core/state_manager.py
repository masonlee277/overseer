# src/overseer/core/state_manager.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, Any, Optional, List
import numpy as np

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.data.data_manager import DataManager
from overseer.core.state import State

class StateManager:
    """
    Manages the persistence, updates, and retrieval of State objects.

    This class acts as an interface between the RL environment and the DataManager.
    It manages the current state and history of states, delegating complex
    calculations and data operations to the DataManager.

    Attributes:
        config (OverseerConfig): Configuration object.
        logger (OverseerLogger): Logger instance.
        data_manager (DataManager): DataManager instance for data operations.
        current_state (Optional[State]): The current state of the simulation.
        state_history (List[State]): History of states for the current episode.
    """

    def __init__(self, config: OverseerConfig, data_manager: DataManager):
        self.config = config
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_manager = data_manager
        self.current_state: Optional[State] = None
        self.state_history: List[State] = []

    def update_state(self, new_data: Dict[str, Any]) -> None:
        """
        Update the current state with new data.

        Args:
            new_data (Dict[str, Any]): New state data to incorporate.
        """
        self.logger.info("Updating state")
        new_state = State(new_data, self.data_manager)
        self.current_state = new_state
        self.state_history.append(new_state)
        self.data_manager.save_state(new_data)

    def get_current_state(self) -> Optional[State]:
        """
        Get the current state.

        Returns:
            Optional[State]: The current state, or None if not initialized.
        """
        return self.current_state

    def get_state_at_time(self, timestamp: float) -> Optional[State]:
        """
        Get the state at a specific timestamp.

        Args:
            timestamp (float): The timestamp to retrieve the state for.

        Returns:
            Optional[State]: The state at the given timestamp, or None if not found.
        """
        state_data = self.data_manager.load_state_at_time(timestamp)
        if state_data:
            return State(state_data, self.data_manager)
        return None

    def get_fire_growth_rate(self, time_window: float) -> float:
        """
        Get the fire growth rate over a specified time window.

        Args:
            time_window (float): The time window to calculate the growth rate over.

        Returns:
            float: The fire growth rate in hectares per hour.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return 0.0
        return self.current_state.get_fire_growth_rate(time_window)

    def get_resource_efficiency(self) -> float:
        """
        Get the resource efficiency for the current state.

        Returns:
            float: The resource efficiency score.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return 0.0
        return self.current_state.get_resource_efficiency()

    def get_fire_containment_percentage(self) -> float:
        """
        Get the fire containment percentage for the current state.

        Returns:
            float: The fire containment percentage.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return 0.0
        return self.current_state.get_fire_containment_percentage()

    def get_high_risk_areas(self) -> Optional[np.ndarray]:
        """
        Get high-risk areas based on the current state.

        Returns:
            Optional[np.ndarray]: A boolean array indicating high-risk areas, or None if state is not initialized.
        """
        if self.current_state is None:
            self.logger.error("Current state is not initialized")
            return None
        return self.current_state.get_high_risk_areas()

    def reset(self) -> None:
        """
        Reset the state manager for a new episode.
        """
        self.logger.info("Resetting StateManager")
        self.current_state = None
        self.state_history.clear()

    def get_state_history(self) -> List[State]:
        """
        Get the history of states for the current episode.

        Returns:
            List[State]: The history of states.
        """
        return self.state_history

    def __str__(self) -> str:
        """Return a string representation of the StateManager."""
        return f"StateManager(current_state={self.current_state}, history_length={len(self.state_history)})"

    def __repr__(self) -> str:
        """Return a string representation of the StateManager for debugging."""
        return self.__str__()