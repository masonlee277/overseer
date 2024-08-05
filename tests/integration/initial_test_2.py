import pytest
from pathlib import Path
import sys

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
from overseer.core.models import SimulationState, Action



class TestElmfireSetup:
    @pytest.fixture(autouse=True)
    def setup_method(self, logger):
        self.logger = logger
        self.logger.info("Starting ELMFIRE setup test")
        yield
        self.logger.info("ELMFIRE setup test completed")

    def test_elmfire_data_in_existence(self, data_in_handler):
        data_in_path = data_in_handler.data_in_path
        self.logger.info(f"data_in_path from handler: {data_in_path}")
        assert data_in_path.exists(), f"elmfire.data.in file not found at: {data_in_path}"
        self.logger.info(f"elmfire.data.in file found at: {data_in_path}")

    def test_elmfire_data_in_structure(self, data_in_handler):
        self.logger.info("Validating structure of elmfire.data.in")
        assert data_in_handler.validate_structure(), "elmfire.data.in structure is invalid"
        self.logger.info("elmfire.data.in structure is valid")

    def test_get_parameter(self, data_in_handler):
        test_param = data_in_handler.get_parameter('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY')
        self.logger.info(f"Test parameter value: {test_param}")
        assert test_param is not None, "Failed to retrieve test parameter"

    def test_validate_input_files(self, data_in_handler):
        self.logger.info("Validating input files")
        files_valid = data_in_handler.validate_input_files()
        assert files_valid, "Not all required input files are present and valid"
        self.logger.info("All required input files are present and valid")

    def test_config_manager_prepare_simulation(self, config_manager):
        self.logger.info("Testing prepare_simulation method")
        config_manager.prepare_simulation()
        # Add assertions here to verify the simulation preparation

    def test_config_manager_apply_action(self, config_manager, action_space):
        fireline_coords = action_space.sample_action()
        self.logger.info(f"Testing apply_action method with action: {fireline_coords}")
        self.logger.info(f"Converted to fireline coordinates: {fireline_coords}")
        config_manager.apply_action(fireline_coords)
        # Add assertions here to verify the action application

    def test_get_config_for_simulation(self, config_manager):
        self.logger.info("Getting simulation configuration")
        sim_config = config_manager.get_config_for_simulation()
        assert sim_config.elmfire_data, "ELMFIRE Data is empty"
        assert sim_config.input_paths, "Input Paths are empty"
        assert sim_config.output_paths, "Output Paths are empty"
        
        self.logger.info("Some specific parameters:")
        cell_size = sim_config.get_parameter('COMPUTATIONAL_DOMAIN', 'COMPUTATIONAL_DOMAIN_CELLSIZE')
        sim_duration = sim_config.get_parameter('TIME_CONTROL', 'SIMULATION_TSTOP')
        time_step = sim_config.get_parameter('TIME_CONTROL', 'SIMULATION_DT')
        
        assert cell_size is not None, "Cell Size is None"
        assert sim_duration is not None, "Simulation Duration is None"
        assert time_step is not None, "Time Step is None"

        self.logger.info(f"Cell Size: {cell_size}")
        self.logger.info(f"Simulation Duration: {sim_duration}")
        self.logger.info(f"Time Step: {time_step}")

    def test_validate_config(self, config_manager):
        self.logger.info("Validating configuration")
        is_valid = config_manager.validate_config()
        assert is_valid, "Configuration is not valid"
        self.logger.info("Configuration is valid")

    def test_save_config(self, config_manager):
        self.logger.info("Saving configuration")
        config_manager.save_config()
        # Add assertions here to verify the configuration was saved correctly

    @pytest.mark.parametrize("section,key,expected_type", [
        ('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY', str),
        ('COMPUTATIONAL_DOMAIN', 'COMPUTATIONAL_DOMAIN_CELLSIZE', (int, float)),
        ('TIME_CONTROL', 'SIMULATION_TSTOP', (int, float)),
    ])
    def test_specific_parameters(self, config_manager, section, key, expected_type):
        sim_config = config_manager.get_config_for_simulation()
        value = sim_config.get_parameter(section, key)
        assert isinstance(value, expected_type), f"{section}.{key} should be of type {expected_type}, but is {type(value)}"
        self.logger.info(f"{section}.{key} = {value}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])