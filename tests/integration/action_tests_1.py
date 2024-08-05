import pytest
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Tuple

# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_dir))

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.elmfire.data_in_handler import ElmfireDataInHandler
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.data.data_manager import DataManager
from overseer.core.models import SimulationState, Action, EpisodeStep

class TestElmfireAction:
    @pytest.fixture(autouse=True)
    def setup_method(self, logger):
        self.logger = logger
        self.logger.info("Starting ELMFIRE action test")
        yield
        self.logger.info("ELMFIRE action test completed")

    def test_action_space_initialization(self, action_space, config):
        self.logger.info("Testing ActionSpace initialization")
        assert isinstance(action_space, ActionSpace)
        assert action_space.grid_size == config.get('grid_size', 100)
        assert action_space.max_fireline_length == config.get('max_fireline_length', 10)
        self.logger.info("ActionSpace initialization test passed")

    def test_action_sampling(self, action_space):
        self.logger.info("Testing action sampling")
        for _ in range(10):
            action = action_space.sample_action()
            assert isinstance(action, Action)
            assert isinstance(action.fireline_coordinates, list)
            assert len(action.fireline_coordinates) > 0
            assert all(isinstance(coord, tuple) and len(coord) == 2 for coord in action.fireline_coordinates)
        self.logger.info("Action sampling test passed")
    def test_action_creation(self, action_space):
        self.logger.info("Testing action creation")
        raw_action = np.array([10, 20, 2, 5])
        action = action_space.create_action(raw_action)
        assert isinstance(action, Action)
        assert len(action.fireline_coordinates) == 5
        assert action.fireline_coordinates[0] == (10, 20)
        self.logger.info("Action creation test passed")

    def test_action_validation(self, action_space):
        self.logger.info("Testing action validation")
        valid_action = np.array([10, 20, 2, 5])
        assert action_space.contains(valid_action)

        invalid_actions = [
            np.array([-1, 20, 2, 5]),  # Invalid x
            np.array([10, -1, 2, 5]),  # Invalid y
            np.array([10, 20, 8, 5]),  # Invalid direction
            np.array([10, 20, 2, 0]),  # Invalid length
            np.array([10, 20, 2, action_space.max_fireline_length + 1]),  # Too long
        ]
        for invalid_action in invalid_actions:
            assert not action_space.contains(invalid_action)
        self.logger.info("Action validation test passed")

    def test_action_mask_generation(self, action_space, mock_state):
        self.logger.info("Testing action mask generation")
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
        assert isinstance(action_mask, np.ndarray)
        assert action_mask.shape == (action_space.grid_size, action_space.grid_size)
        assert np.any(action_mask)  # Ensure at least some actions are valid
        assert not np.all(action_mask)  # Ensure not all actions are valid
        self.logger.info("Action mask generation test passed")

    def test_action_encoding_decoding(self, action_space):
        self.logger.info("Testing action encoding and decoding")
        original_action = np.array([10, 20, 2, 5])
        encoded_action = action_space.encode_action(*original_action)
        decoded_action = action_space.decode_action(encoded_action)
        
        assert decoded_action['x'] == original_action[0]
        assert decoded_action['y'] == original_action[1]
        assert decoded_action['direction'] == ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][original_action[2]]
        assert decoded_action['length'] == original_action[3]
        self.logger.info("Action encoding and decoding test passed")



    def test_action_space_edge_cases(self, action_space, mock_state):
        self.logger.info("Testing ActionSpace edge cases")

        # Test 1: Extremely large action values
        large_action = [1000000, 1000000, 8, 1000000]
        is_valid = action_space.contains(large_action)
        self.logger.info(f"Extremely large action validity: {is_valid}")
        assert not is_valid, "Extremely large action should be invalid"

        # Test 2: Negative action values
        negative_action = [-10, -10, 2, 5]
        is_valid = action_space.contains(negative_action)
        self.logger.info(f"Negative action validity: {is_valid}")
        assert not is_valid, "Negative action should be invalid"

        # Test 3: Action with invalid direction
        invalid_direction_action = [50, 50, 9, 5]
        is_valid = action_space.contains(invalid_direction_action)
        self.logger.info(f"Invalid direction action validity: {is_valid}")
        assert not is_valid, "Action with invalid direction should be invalid"

        # Test 4: Action with zero length
        zero_length_action = [50, 50, 0, 0]
        is_valid = action_space.contains(zero_length_action)
        self.logger.info(f"Zero length action validity: {is_valid}")
        assert not is_valid, "Action with zero length should be invalid"

    def test_long_running_simulation(self, config_manager, action_space, mock_state):

        self.logger.info("Testing long-running simulation scenario")

        start_time = datetime.now()
        end_time = start_time + timedelta(days=7)  # 7-day simulation
        time_step = timedelta(hours=1)

        current_time = start_time
        while current_time < end_time:
            # Update mock state
            mock_state.timestamp = current_time
            mock_state.burned_area += np.random.randint(10, 100)
            mock_state.fire_perimeter_length += np.random.randint(5, 50)
            mock_state.containment_percentage = min(100, mock_state.containment_percentage + np.random.random() * 2)

            # Sample and apply action

            action = action_space.sample_action()
            try:
                config_manager.apply_action(action)
            except Exception as e:
                self.logger.error(f"Error applying action at {current_time}: {str(e)}")
                assert False, f"Action application should not fail: {str(e)}"

            # Log the action
            self.logger.info(f"Applied action at {current_time}: {action.fireline_coordinates}")

            # Update simulation config
            try:
                sim_config = config_manager.get_config_for_simulation()
                config_manager.data_in_handler.set_parameter('TIME_CONTROL', 'SIMULATION_TSTOP', str((current_time - start_time).total_seconds()))
            except Exception as e:
                self.logger.error(f"Error updating simulation config at {current_time}: {str(e)}")
                assert False, f"Simulation config update should not fail: {str(e)}"

            current_time += time_step

        self.logger.info("Long-running simulation completed successfully")

    def test_rapid_config_changes(self, config_manager, action_space):
        self.logger.info("Testing rapid configuration changes")

        for i in range(100):
            self.logger.info(f"Testing rapid config change {i}")
            section = np.random.choice(['INPUTS', 'OUTPUTS', 'TIME_CONTROL', 'COMPUTATIONAL_DOMAIN'])
            key = f"TEST_PARAM_{np.random.randint(1, 100)}"
            value = str(np.random.random())

            try:
                config_manager.data_in_handler.set_parameter(section, key, value)
            except Exception as e:
                self.logger.error(f"Error setting parameter {section}.{key}: {str(e)}")
                assert False, f"Rapid config changes should not fail: {str(e)}"

            # Apply a random action after each config change
            action = action_space.sample_action()

            try:
                config_manager.apply_action(action)
                self.logger.info(f"Applied action: {action.fireline_coordinates}")
            except Exception as e:
                self.logger.error(f"Error applying action during rapid config changes: {str(e)}")
                assert False, f"Action application during rapid config changes should not fail: {str(e)}"

        try:
            config_manager.save_config()
        except Exception as e:
            self.logger.error(f"Error saving config after rapid changes: {str(e)}")
            assert False, f"Saving config after rapid changes should not fail: {str(e)}"

        self.logger.info("Rapid configuration changes test completed successfully")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])