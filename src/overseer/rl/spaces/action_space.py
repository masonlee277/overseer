"""
Action Space for ELMFIRE Reinforcement Learning Environment
The challenge is that the action space continually changes due to progression of fire

Justification for the Action Space Design:

1. Discrete vs. Continuous:
   We use a discrete action space because fireline construction typically involves
   discrete decisions (where to start, which direction to build, how long to make it).
   This aligns well with many RL algorithms and makes the action space more interpretable.

2. Structure:
   The action space is structured as a MultiDiscrete space with four components:
   (x, y, direction, length). This allows for flexible fireline placement while
   maintaining a manageable action space size.

3. Spatial Awareness:
   By including x and y coordinates, we enable the agent to learn spatial strategies,
   such as creating firelines near valuable assets or in areas prone to rapid fire spread.

4. Directional Flexibility:
   The four cardinal directions (N, E, S, W) provide sufficient flexibility for fireline
   construction while keeping the action space reasonably sized. Diagonal directions
   were omitted to reduce complexity, as they can be approximated by combinations of
   cardinal directions.

5. Variable Length:
   Allowing variable fireline lengths enables the agent to learn efficient resource
   allocation, balancing between numerous short firelines and fewer long ones.

6. Constraints and Validity:
   We implement constraint checking in the `contains` method to ensure only valid
   actions are considered. This prevents impossible firelines (e.g., those extending
   beyond the grid) and guides the agent towards feasible solutions.

7. Compatibility:
   This design is compatible with many RL algorithms that work with discrete action
   spaces, including DQN, PPO, and A2C. It's also amenable to masking techniques for
   invalid actions.

8. Scalability:
   The action space scales well with grid size. For larger grids, we can adjust the
   `max_fireline_length` or introduce a coarser grid for action selection without
   changing the fundamental structure.

9. Interpretability:
   Each action directly translates to a specific fireline construction command,
   making it easy to interpret and visualize the agent's decisions.

10. Future Extensibility:
    This design allows for future extensions, such as adding new action types
    (e.g., creating firebreaks, deploying water drops) by expanding the MultiDiscrete
    space or introducing hierarchical action selection.

Trade-offs and Considerations:
- The current design may lead to a large action space for big grids, which could
  slow down learning. This can be mitigated by using action masking or hierarchical RL.
- The discrete nature might miss some optimal fireline placements. However, the
  increased tractability and interpretability outweigh this limitation for most scenarios.
- Future versions might consider incorporating continuous elements for more precise
  positioning, but this would require careful balancing against increased complexity.

11. Action Masking:
    We implement action masking to significantly reduce the effective action space
    and improve learning efficiency. This is crucial for large grid sizes where the
    number of possible actions can become overwhelming for the RL agent.

Constraints for Realistic Fireline Construction:

1. Fire Avoidance:
   Firelines cannot be constructed inside or leading into active fire areas. This
   constraint ensures that the agent learns to create preventive barriers rather
   than ineffective lines through burning areas.

2. Proximity to Fire:
   Firelines must be within a certain distance of the fire front or high-risk areas.
   This constraint prevents the agent from wasting resources on distant, irrelevant
   actions while keeping the state space manageable.

3. No Self-Intersections:
   Firelines should not intersect with existing firelines. This rule promotes
   efficient resource use and prevents redundant actions.

4. Terrain Considerations:
   Fireline construction should account for terrain features. For example, it may be
   invalid or less effective to construct firelines on very steep slopes or water bodies.

5. Resource Limitations:
   The total length of firelines constructed should be limited based on available
   resources (e.g., personnel, equipment). This constraint forces the agent to
   make strategic decisions about fireline placement.

6. Vegetation Type:
   Certain vegetation types may be more or less suitable for fireline construction.
   The action space should consider the underlying vegetation map when determining
   valid actions.

7. Access Constraints:
   Firelines should be constructable from accessible areas. This might mean starting
   from existing roads or previously constructed firelines.

8. Protection Priorities:
   Actions near high-value assets or vulnerable areas should be prioritized. This
   can be implemented as a soft constraint through the reward function or as part
   of the action masking process.

9. Minimum Effective Length:
   Firelines shorter than a certain length may be ineffective and should be disallowed.
   This prevents the agent from making many small, ineffective actions.

10. Time Constraints:
    In a time-stepped simulation, there may be limits on how much fireline can be
    constructed in a single time step, reflecting real-world construction rates.

"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import numpy as np
from typing import List, Tuple, Optional
from gym import Space
from datetime import datetime
import matplotlib.pyplot as plt

from overseer.config.config import OverseerConfig
from overseer.data.data_manager import DataManager
from overseer.core.models import Action, SimulationState, EpisodeStep
from overseer.utils.logging import OverseerLogger
from overseer.core.models import Action


class ActionSpace(Space):
    """
    Custom action space for the ELMFIRE simulation environment.

    This space represents actions as a combination of resource allocation and fireline construction.
    """

    def __init__(self, config: OverseerConfig, data_manager: DataManager):
        super().__init__(shape=(4,), dtype=np.int32)
        self.config = config
        self.data_manager = data_manager
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)

        self.grid_size = self.config.get('grid_size', 100)
        self.max_fireline_length = self.config.get('max_fireline_length', 10)
        self.max_fireline_distance = self.config.get('max_fireline_distance', 5)
        self.max_construction_slope = self.config.get('max_construction_slope', 30)
        self.min_effective_length = self.config.get('min_effective_length', 3)
        self.constructable_veg_types = self.config.get('constructable_veg_types', [1, 2, 3])

    def sample(self) -> np.ndarray:
        """Sample a random action from the action space and return raw action space."""
        x = np.random.randint(0, self.grid_size)
        y = np.random.randint(0, self.grid_size)
        direction = np.random.randint(0, 8)
        length = np.random.randint(1, self.max_fireline_length + 1)
        return np.array([x, y, direction, length], dtype=np.int32)

    def sample_action(self) -> Action:
        """Sample a random action and return it as an Action instance."""
        raw_action = self.sample()
        coordinates = self.action_to_coordinates(raw_action)
        return Action(fireline_coordinates=coordinates)
    
    def contains(self, x) -> bool:
        """Check if the given action is within the action space."""
        if not isinstance(x, np.ndarray) or x.shape != (4,):
            return False
        x, y, direction, length = x
        return (0 <= x < self.grid_size and
                0 <= y < self.grid_size and
                0 <= direction < 8 and
                1 <= length <= self.max_fireline_length)


    def action_to_coordinates(self, action: np.ndarray) -> List[Tuple[int, int]]:
        """Convert an action to a list of coordinates representing the fireline."""
        x, y, direction, length = action
        dx, dy = self._get_direction_offsets(direction)
        return [(x + i*dx, y + i*dy) for i in range(length)]
    
    def get_action_mask(self, episode_step: EpisodeStep) -> np.ndarray:
        """Generate an action mask based on the current episode step."""
        self.logger.info("Generating action mask")
        
        action_mask = self.data_manager.generate_action_mask_from_episode(episode_step)
        
        self.logger.info(f"Action mask generated with shape: {action_mask.shape}")
        return action_mask
    
    def _check_constraints(self, action: np.ndarray, state: SimulationState) -> bool:
        fireline_coords = self._get_fireline_coords(action)

        self.logger.debug(f"Checking constraints for action: {action}")

        return self.data_manager.check_action_constraints(
            state,
            fireline_coords,
            self.max_fireline_distance,
            self.max_construction_slope,
            self.constructable_veg_types,
            self.min_effective_length
        )
    
    def _get_fireline_coords(self, action: np.ndarray) -> np.ndarray:
        x, y, direction, length = action
        dx, dy = self._get_direction_offsets(direction)
        coords = np.array([(x + i*dx, y + i*dy) for i in range(length)])
        return coords

    def create_action(self, raw_action: np.ndarray) -> Action:
        """
        Create an Action instance from raw action data.

        Args:
            raw_action (np.ndarray): The raw action data [x, y, direction, length].

        Returns:
            Action: An Action instance with the corresponding fireline coordinates.
        """
        coordinates = self.action_to_coordinates(raw_action)
        return Action(fireline_coordinates=coordinates)
    
    def _get_direction_offsets(self, direction: int) -> Tuple[int, int]:
        return [
            (0, 1), (1, 1), (1, 0), (1, -1),
            (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ][direction]

    def visualize_action(self, action: np.ndarray) -> None:
        
        grid = np.zeros((self.grid_size, self.grid_size))
        coords = self._get_fireline_coords(action)
        for x, y in coords:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y, x] = 1
        
        plt.imshow(grid, cmap='binary')
        plt.title(f"Action: {action}")
        plt.show()

    def encode_action(self, x: int, y: int, direction: int, length: int) -> Action:
        return Action(
            fireline_coordinates=self._get_fireline_coords(np.array([x, y, direction, length]))
        )

    def decode_action(self, action: Action) -> dict:
        if not action.fireline_coordinates:
            return {}
        start = action.fireline_coordinates[0]
        end = action.fireline_coordinates[-1]
        direction = self._get_direction_from_coords(start, end)
        length = len(action.fireline_coordinates)
        return {
            "x": start[0],
            "y": start[1],
            "direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][direction],
            "length": length
        }

    def _get_direction_from_coords(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if dx == 0:
            return 0 if dy > 0 else 4
        elif dy == 0:
            return 2 if dx > 0 else 6
        elif dx > 0:
            return 1 if dy > 0 else 3
        else:
            return 7 if dy > 0 else 5
def main():
    config = OverseerConfig()
    data_manager = DataManager(config)
    action_space = ActionSpace(config, data_manager)
    logger = OverseerLogger().get_logger("ActionSpaceTest")

    logger.info("Starting comprehensive ActionSpace tests")

    # Test 1: Basic Functionality
    logger.info("Test 1: Basic Functionality - Sampling and Validation")
    for i in range(1, 6):
        raw_action = action_space.sample()
        action = action_space.create_action(raw_action)
        is_valid = action_space.contains(raw_action)
        logger.info(f"  1.{i}. Sampled action: {raw_action}")
        logger.info(f"      Action coordinates: {action.fireline_coordinates}")
        logger.info(f"      Is valid: {is_valid}")
        logger.info(f"      Expected: True")
        logger.info(f"      Result: {'PASS' if is_valid else 'FAIL'}")

    # Test 2: Edge Cases
    logger.info("Test 2: Edge Cases")
    edge_cases = [
        ([0, 0, 0, 1], True, "Minimum valid action"),
        ([action_space.grid_size-1, action_space.grid_size-1, 7, action_space.max_fireline_length], True, "Maximum valid action"),
        ([-1, 0, 0, 1], False, "Invalid x (negative)"),
        ([action_space.grid_size, 0, 0, 1], False, "Invalid x (too large)"),
        ([0, -1, 0, 1], False, "Invalid y (negative)"),
        ([0, action_space.grid_size, 0, 1], False, "Invalid y (too large)"),
        ([0, 0, 8, 1], False, "Invalid direction"),
        ([0, 0, 0, 0], False, "Invalid length (zero)"),
        ([0, 0, 0, action_space.max_fireline_length + 1], False, "Invalid length (too long)"),
    ]
    for i, (case, expected, description) in enumerate(edge_cases, 1):
        is_valid = action_space.contains(np.array(case))
        action = action_space.create_action(np.array(case))
        logger.info(f"  2.{i}. Testing: {description}")
        logger.info(f"      Raw action: {case}")
        logger.info(f"      Action coordinates: {action.fireline_coordinates}")
        logger.info(f"      Is valid: {is_valid}")
        logger.info(f"      Expected: {expected}")
        logger.info(f"      Result: {'PASS' if is_valid == expected else 'FAIL'}")

    # Test 3: Action Mask Generation
    logger.info("Test 3: Action Mask Generation")
    mock_state = SimulationState(
        timestamp=datetime.now(),
        fire_intensity=np.random.rand(100, 100),
        burned_area=1000.0,
        fire_perimeter_length=500.0,
        containment_percentage=20.0,
        resources={'firefighters': 50, 'trucks': 10},
        weather={'temperature': 25.0, 'wind_speed': 10.0, 'wind_direction': 180.0}
    )
    mock_episode_step = EpisodeStep(
        step=0,
        state=mock_state,
        action=None,
        reward=0,
        next_state=None,
        simulation_result=None,
        done=False
    )
    action_mask = action_space.get_action_mask(mock_episode_step)
    logger.info(f"  3.1. Action mask shape: {action_mask.shape}")
    logger.info(f"       Expected shape: ({action_space.grid_size}, {action_space.grid_size})")
    logger.info(f"       Result: {'PASS' if action_mask.shape == (action_space.grid_size, action_space.grid_size) else 'FAIL'}")
    
    valid_actions = np.sum(action_mask)
    logger.info(f"  3.2. Number of valid actions: {valid_actions}")
    logger.info(f"       Total possible actions: {action_space.grid_size * action_space.grid_size}")
    logger.info(f"       Percentage of valid actions: {valid_actions / (action_space.grid_size * action_space.grid_size) * 100:.2f}%")
    logger.info(f"       Result: {'PASS' if 0 < valid_actions < action_space.grid_size * action_space.grid_size else 'FAIL'}")

    # Test 4: Constraint Checking
    logger.info("Test 4: Constraint Checking")
    test_actions = [
        ([0, 0, 0, 3], True, "Valid action"),
        ([8, 8, 0, 3], False, "Invalid: in fire area"),
        ([2, 2, 0, 3], False, "Invalid: intersects existing fireline"),
        ([0, 0, 0, 1], False, "Invalid: too short"),
        ([0, 0, 0, 20], False, "Invalid: too long")
    ]

    for i, (raw_action, expected, description) in enumerate(test_actions, 1):
        action = action_space.create_action(np.array(raw_action))
        is_valid = action_space._check_constraints(np.array(raw_action), mock_state)
        logger.info(f"  4.{i}. Testing: {description}")
        logger.info(f"      Raw action: {raw_action}")
        logger.info(f"      Action coordinates: {action.fireline_coordinates}")
        logger.info(f"      Is valid: {is_valid}")
        logger.info(f"      Expected: {expected}")
        logger.info(f"      Result: {'PASS' if is_valid == expected else 'FAIL'}")

    logger.info("All tests completed")

if __name__ == "__main__":
    main()