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

import gym
import numpy as np
from typing import Tuple, List
from overseer.config import OverseerConfig
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.core.state import State
from overseer.utils.logging import OverseerLogger

class ActionSpace:
    """
    Defines the action space for the RL environment, focusing on fireline construction.
    """
    
    def __init__(self, config: OverseerConfig, geospatial_manager: GeoSpatialManager):
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.logger.info("Initializing ActionSpace")
        
        self.config = config
        self.geospatial_manager = geospatial_manager
        self._initialize_parameters()
        
        self.space = self.create()
        self.logger.info(f"Action space created with shape: {self.space.shape}")

    def _initialize_parameters(self):
        self.logger.info("Initializing ActionSpace parameters")
        
        # Spatial parameters
        self.grid_size_x = self.config.get('spatial.extent.xmax', 10000)
        self.grid_size_y = self.config.get('spatial.extent.ymax', 10000)
        self.resolution = self.config.get('spatial.resolution', 30)
        self.grid_size = max(self.grid_size_x, self.grid_size_y) // self.resolution
        self.logger.info(f"Grid size: {self.grid_size}, Resolution: {self.resolution}")

        # Action space parameters
        self.max_fireline_length = self.config.get('reinforcement_learning.action_space.max_fireline_length', 10)
        self.max_fireline_distance = self.config.get('reinforcement_learning.action_space.max_fireline_distance', 10)
        self.max_construction_slope = self.config.get('reinforcement_learning.action_space.max_construction_slope', 30)
        self.constructable_veg_types = self.config.get('reinforcement_learning.action_space.constructable_vegetation_types', [1, 2, 3])
        self.logger.info(f"Max fireline length: {self.max_fireline_length}, Max distance: {self.max_fireline_distance}")
        self.logger.info(f"Max construction slope: {self.max_construction_slope}")
        self.logger.info(f"Constructable vegetation types: {self.constructable_veg_types}")

        # Constraint parameters
        self.min_effective_length = self.config.get('reinforcement_learning.constraints.min_effective_fireline_length', 3)
        self.max_firelines_per_timestep = self.config.get('reinforcement_learning.constraints.max_firelines_per_timestep', 5)
        self.resource_limit_factor = self.config.get('reinforcement_learning.constraints.resource_limit_factor', 0.8)
        self.logger.info(f"Min effective length: {self.min_effective_length}")
        self.logger.info(f"Max firelines per timestep: {self.max_firelines_per_timestep}")
        self.logger.info(f"Resource limit factor: {self.resource_limit_factor}")

    def create(self) -> gym.spaces.MultiDiscrete:
        self.logger.debug("Creating MultiDiscrete action space")
        return gym.spaces.MultiDiscrete([
            self.grid_size,  # x-coordinate
            self.grid_size,  # y-coordinate
            8,  # direction (0: N, 1: NE, 2: E, 3: SE, 4: S, 5: SW, 6: W, 7: NW)
            self.max_fireline_length  # length of fireline
        ])

    def sample(self) -> np.ndarray:
        """Sample a random action from the action space."""
        self.logger.debug("Sampling a random action")
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            action = self.space.sample()
            if self.contains(action):
                self.logger.debug(f"Valid action sampled: {action}")
                return action
            attempts += 1
        self.logger.warning(f"Failed to sample a valid action after {max_attempts} attempts")
        return None

    def contains(self, action: np.ndarray) -> bool:
        """Check if an action is within the action space and valid."""
        if not self.space.contains(action):
            self.logger.debug(f"Action {action} is not within the action space")
            return False
        is_valid = self._is_valid_fireline(action)
        self.logger.debug(f"Action {action} validity: {is_valid}")
        return is_valid

    def _is_valid_fireline(self, action: np.ndarray) -> bool:
        x, y, direction, length = action
        end_x, end_y = x, y
        
        dx, dy = self._get_direction_offsets(direction)
        end_x += dx * length
        end_y += dy * length
        
        is_valid = 0 <= end_x < self.grid_size and 0 <= end_y < self.grid_size
        self.logger.debug(f"Fireline validity check for action {action}: {is_valid}")
        return is_valid

    def _get_direction_offsets(self, direction: int) -> Tuple[int, int]:
        offsets = [
            (0, 1),   # N
            (1, 1),   # NE
            (1, 0),   # E
            (1, -1),  # SE
            (0, -1),  # S
            (-1, -1), # SW
            (-1, 0),  # W
            (-1, 1)   # NW
        ]
        return offsets[direction]

    def _get_fireline_coords(self, action: np.ndarray) -> np.ndarray:
        x, y, direction, length = action
        dx, dy = self._get_direction_offsets(direction)
        coords = [(x + i*dx, y + i*dy) for i in range(length)]
        return np.array(coords)
    
    def get_action_mask(self, state: State) -> np.ndarray:
        """
        Generate an action mask based on the current state.

        Args:
            state (State): The current state of the environment.

        Returns:
            np.array: A boolean mask where True indicates a valid action.
        """
        self.logger.info("Generating action mask")
        geospatial_data_paths = state.get_geospatial_data_paths()
        
        self.logger.debug(f"Geospatial data paths: {geospatial_data_paths}")
        
        # Generate base action mask using GeoSpatialManager
        action_mask = self.geospatial_manager.generate_action_mask_from_files(
            fire_intensity_path=geospatial_data_paths.get('fire_intensity'),
            existing_firelines_path=geospatial_data_paths.get('firelines'),
            elevation_path=geospatial_data_paths.get('elevation'),
            vegetation_path=geospatial_data_paths.get('vegetation'),
            min_distance=1,
            max_distance=self.max_fireline_distance,
            max_slope=np.tan(np.radians(self.max_construction_slope)),
            constructable_veg_types=self.constructable_veg_types,
            min_effective_length=self.min_effective_length
        )
        
        self.logger.info(f"Action mask generated with shape: {action_mask.shape}")
        return action_mask

    def _check_constraints(self, action: np.ndarray, fire_intensity: np.ndarray, existing_firelines: np.ndarray, 
                           slope: np.ndarray, fire_distance: np.ndarray, vegetation: np.ndarray) -> bool:
        x, y, direction, length = action
        fireline_coords = self._get_fireline_coords(action)

        self.logger.debug(f"Checking constraints for action: {action}")

        # Check fire avoidance
        if np.any(fire_intensity[fireline_coords[:, 1], fireline_coords[:, 0]] > 0):
            self.logger.debug("Constraint violated: fire avoidance")
            return False

        # Check proximity to fire
        if np.all(fire_distance[fireline_coords[:, 1], fireline_coords[:, 0]] > self.max_fireline_distance):
            self.logger.debug("Constraint violated: proximity to fire")
            return False

        # Check terrain slope
        if np.any(slope[fireline_coords[:, 1], fireline_coords[:, 0]] > np.tan(np.radians(self.max_construction_slope))):
            self.logger.debug("Constraint violated: terrain slope")
            return False

        # Check vegetation type
        if not np.all(np.isin(vegetation[fireline_coords[:, 1], fireline_coords[:, 0]], self.constructable_veg_types)):
            self.logger.debug("Constraint violated: vegetation type")
            return False

        # Check for intersections with existing firelines
        if np.any(existing_firelines[fireline_coords[:, 1], fireline_coords[:, 0]] > 0):
            self.logger.debug("Constraint violated: intersection with existing firelines")
            return False

        # Check minimum effective length
        if length < self.min_effective_length:
            self.logger.debug("Constraint violated: minimum effective length")
            return False

        self.logger.debug("All constraints satisfied")
        return True

    def visualize_action(self, action: np.ndarray):
        import matplotlib.pyplot as plt
        
        grid = np.zeros((self.grid_size, self.grid_size))
        coords = self._get_fireline_coords(action)
        for x, y in coords:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                grid[y, x] = 1
        
        plt.imshow(grid, cmap='binary')
        plt.title(f"Action: {action}")
        plt.show()

    def encode_action(self, x: int, y: int, direction: int, length: int) -> np.ndarray:
        return np.array([x, y, direction, length])

    def decode_action(self, action: np.ndarray) -> dict:
        x, y, direction, length = action
        return {
            "x": x,
            "y": y,
            "direction": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][direction],
            "length": length
        }
    
    
def main():
    config = OverseerConfig()
    geospatial_manager = GeoSpatialManager(config)
    action_space = ActionSpace(config, geospatial_manager)
    logger = OverseerLogger().get_logger("ActionSpaceTest")

    logger.info("Starting comprehensive ActionSpace tests")

    # Test 1: Basic Functionality
    logger.info("Test 1: Basic Functionality - Sampling and Validation")
    for i in range(1, 6):
        action = action_space.sample()
        is_valid = action_space.contains(action)
        logger.info(f"  1.{i}. Sampled action: {action}")
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
        logger.info(f"  2.{i}. Testing: {description}")
        logger.info(f"      Action: {case}")
        logger.info(f"      Is valid: {is_valid}")
        logger.info(f"      Expected: {expected}")
        logger.info(f"      Result: {'PASS' if is_valid == expected else 'FAIL'}")

    # Test 3: Action Mask Generation
    logger.info("Test 3: Action Mask Generation")
    class MockState:
        def get_geospatial_data_paths(self):
            return {
                'fire_intensity': 'path/to/fire_intensity.tif',
                'firelines': 'path/to/firelines.tif',
                'elevation': 'path/to/elevation.tif',
                'vegetation': 'path/to/vegetation.tif'
            }
    
    mock_state = MockState()
    action_mask = action_space.get_action_mask(mock_state)
    logger.info(f"  3.1. Action mask shape: {action_mask.shape}")
    logger.info(f"       Expected shape: ({action_space.grid_size}, {action_space.grid_size})")
    logger.info(f"       Result: {'PASS' if action_mask.shape == (action_space.grid_size, action_space.grid_size) else 'FAIL'}")
    
    valid_actions = np.sum(action_mask)
    logger.info(f"  3.2. Number of valid actions: {valid_actions}")
    logger.info(f"       Total possible actions: {action_space.grid_size * action_space.grid_size}")
    logger.info(f"       Percentage of valid actions: {valid_actions / (action_space.grid_size * action_space.grid_size) * 100:.2f}%")
    logger.info(f"       Result: {'PASS' if 0 < valid_actions < action_space.grid_size * action_space.grid_size else 'FAIL'}")

    # Test 4: Directional Offsets
    logger.info("Test 4: Directional Offsets")
    expected_offsets = [
        (0, 1), (1, 1), (1, 0), (1, -1),
        (0, -1), (-1, -1), (-1, 0), (-1, 1)
    ]
    for i, direction in enumerate(range(8)):
        offset = action_space._get_direction_offsets(direction)
        logger.info(f"  4.{i+1}. Direction {direction}: {offset}")
        logger.info(f"       Expected: {expected_offsets[i]}")
        logger.info(f"       Result: {'PASS' if offset == expected_offsets[i] else 'FAIL'}")

    # Test 5: Fireline Coordinates Generation
    logger.info("Test 5: Fireline Coordinates Generation")
    test_actions = [
        ([5, 5, 0, 3], [(5, 5), (5, 6), (5, 7)]),  # North
        ([5, 5, 2, 3], [(5, 5), (6, 5), (7, 5)]),  # East
        ([5, 5, 4, 3], [(5, 5), (5, 4), (5, 3)]),  # South
        ([5, 5, 6, 3], [(5, 5), (4, 5), (3, 5)])   # West
    ]
    for i, (action, expected_coords) in enumerate(test_actions, 1):
        coords = action_space._get_fireline_coords(np.array(action))
        logger.info(f"  5.{i}. Action: {action}")
        logger.info(f"      Generated coordinates: {coords.tolist()}")
        logger.info(f"      Expected coordinates: {expected_coords}")
        logger.info(f"      Result: {'PASS' if np.array_equal(coords, np.array(expected_coords)) else 'FAIL'}")

    # Test 6: Constraint Checking
    logger.info("Test 6: Constraint Checking")
    # Create mock data for constraint checking
    mock_fire_intensity = np.zeros((10, 10))
    mock_fire_intensity[5:, 5:] = 100  # Fire in bottom-right quadrant
    mock_existing_firelines = np.zeros((10, 10))
    mock_existing_firelines[2:4, 2:4] = 1  # Existing fireline in top-left
    mock_slope = np.full((10, 10), 20)  # 20 degree slope everywhere
    mock_fire_distance = np.full((10, 10), 5)  # 5 units from fire everywhere
    mock_vegetation = np.full((10, 10), 2)  # Vegetation type 2 everywhere

    test_actions = [
        ([0, 0, 0, 3], True, "Valid action"),
        ([8, 8, 0, 3], False, "Invalid: in fire area"),
        ([2, 2, 0, 3], False, "Invalid: intersects existing fireline"),
        ([0, 0, 0, 1], False, "Invalid: too short"),
        ([0, 0, 0, 20], False, "Invalid: too long")
    ]

    for i, (action, expected, description) in enumerate(test_actions, 1):
        is_valid = action_space._check_constraints(
            np.array(action), mock_fire_intensity, mock_existing_firelines,
            mock_slope, mock_fire_distance, mock_vegetation
        )
        logger.info(f"  6.{i}. Testing: {description}")
        logger.info(f"      Action: {action}")
        logger.info(f"      Is valid: {is_valid}")
        logger.info(f"      Expected: {expected}")
        logger.info(f"      Result: {'PASS' if is_valid == expected else 'FAIL'}")

    logger.info("All tests completed")

if __name__ == "__main__":
    main()