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
"""
Action Space for ELMFIRE Reinforcement Learning Environment
[... existing justification and constraints ...]
"""

import gym
import numpy as np
from overseer.config import OverseerConfig
from overseer.data.geospatial_manager import GeoSpatialManager

class ActionSpace:
    """
    Defines the action space for the RL environment, focusing on fireline construction.
    """
    
    def __init__(self, config: OverseerConfig, geospatial_manager: GeoSpatialManager):
        self.config = config
        self.geospatial_manager = geospatial_manager
        self.grid_size = config.get('spatial.extent.xmax') // config.get('spatial.resolution')
        self.max_fireline_length = config.get('rl.max_fireline_length', 10)
        self.space = self.create()

    def create(self):
        """Create the action space for the ELMFIRE environment."""
        return gym.spaces.MultiDiscrete([
            self.grid_size,  # x-coordinate
            self.grid_size,  # y-coordinate
            4,  # direction (0: North, 1: East, 2: South, 3: West)
            self.max_fireline_length  # length of fireline
        ])

    def sample(self):
        """Sample a random action from the action space."""
        while True:
            action = self.space.sample()
            if self.contains(action):
                return action

    def contains(self, action):
        """Check if an action is within the action space and valid."""
        if not self.space.contains(action):
            return False
        return self._is_valid_fireline(action)

    def _is_valid_fireline(self, action):
        """Check if a fireline action is valid (doesn't go out of bounds)."""
        x, y, direction, length = action
        end_x, end_y = x, y
        
        if direction == 0:  # North
            end_y += length
        elif direction == 1:  # East
            end_x += length
        elif direction == 2:  # South
            end_y -= length
        elif direction == 3:  # West
            end_x -= length
        
        return 0 <= end_x < self.grid_size and 0 <= end_y < self.grid_size

    def get_action_mask(self, state):
        """
        Generate an action mask based on the current state.

        Args:
            state (Dict): The current state of the environment.

        Returns:
            np.array: A boolean mask where True indicates a valid action.
        """
        mask = np.zeros(self.space.nvec, dtype=bool)
        
        # Load necessary data using GeoSpatialManager
        fire_intensity = self.geospatial_manager.load_tiff(state['fire_intensity_path'])[0]
        existing_firelines = self.geospatial_manager.load_tiff(state['firelines_path'])[0]
        elevation = self.geospatial_manager.load_tiff(state['elevation_path'])[0]
        vegetation = self.geospatial_manager.load_tiff(state['vegetation_path'])[0]

        # Calculate additional constraints
        slope, _ = self.geospatial_manager.calculate_slope_aspect(elevation)
        fire_distance = self.geospatial_manager.calculate_distance_to_fire(fire_intensity)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if existing_firelines[y, x] == 0:  # Only allow new firelines where none exist
                    for direction in range(4):
                        for length in range(1, self.max_fireline_length + 1):
                            action = np.array([x, y, direction, length])
                            if self._is_valid_fireline(action):
                                if self._check_constraints(action, fire_intensity, existing_firelines, slope, fire_distance, vegetation):
                                    mask[x, y, direction, length-1] = True

        return mask

    def _check_constraints(self, action, fire_intensity, existing_firelines, slope, fire_distance, vegetation):
        x, y, direction, length = action
        fireline_coords = self._get_fireline_coords(action)

        # Check fire avoidance
        if np.any(fire_intensity[fireline_coords[:, 1], fireline_coords[:, 0]] > 0):
            return False

        # Check proximity to fire
        max_distance = self.config.get('rl.max_fireline_distance', 10)
        if np.all(fire_distance[fireline_coords[:, 1], fireline_coords[:, 0]] > max_distance):
            return False

        # Check terrain slope
        max_slope = self.config.get('rl.max_construction_slope', 30)  # in degrees
        if np.any(slope[fireline_coords[:, 1], fireline_coords[:, 0]] > np.tan(np.radians(max_slope))):
            return False

        # Check vegetation type
        constructable_veg_types = self.config.get('rl.constructable_vegetation_types', [1, 2, 3])
        if not np.all(np.isin(vegetation[fireline_coords[:, 1], fireline_coords[:, 0]], constructable_veg_types)):
            return False

        # Check for intersections with existing firelines
        if np.any(existing_firelines[fireline_coords[:, 1], fireline_coords[:, 0]] > 0):
            return False

        return True

    def _get_fireline_coords(self, action):
        x, y, direction, length = action
        coords = []
        for i in range(length):
            if direction == 0:  # North
                coords.append((x, y + i))
            elif direction == 1:  # East
                coords.append((x + i, y))
            elif direction == 2:  # South
                coords.append((x, y - i))
            elif direction == 3:  # West
                coords.append((x - i, y))
        return np.array(coords)