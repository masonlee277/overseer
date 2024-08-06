import pytest
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import sys

# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_dir))

from overseer.config.config import OverseerConfig
from overseer.data.data_manager import DataManager
from overseer.core.models import SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics, InputPaths, OutputPaths, Action
from overseer.utils.logging import OverseerLogger
from overseer.elmfire.config_manager import ElmfireConfigManager

class TestConfigManagerIntegration:
    @pytest.fixture(autouse=True)
    def setup_method(self, logger):
        self.logger = logger
        self.logger.info("Starting Config Manager Integration test")
        yield
        self.logger.info("Config Manager Integration test completed")

    def test_config_manager_initialization(self, config, data_manager):
        self.logger.info("Testing config manager initialization")
        config_manager = ElmfireConfigManager(config, data_manager)
        assert config_manager is not None
        assert isinstance(config_manager, ElmfireConfigManager)
        self.logger.info("Config manager initialization successful")

    def test_config_manager_get_config(self, config_manager):
        self.logger.info("Testing config manager get_config method")
        elmfire_config = config_manager.get_config()
        assert elmfire_config is not None
        assert isinstance(elmfire_config, dict)
        assert 'INPUTS' in elmfire_config
        assert 'OUTPUTS' in elmfire_config
        assert 'TIME_CONTROL' in elmfire_config
        self.logger.info("Config manager get_config method successful")

    def test_config_manager_update_config(self, config_manager):
        self.logger.info("Testing config manager update_config method")
        original_config = config_manager.get_config()
        
        # Update a specific parameter
        config_manager.update_config('TIME_CONTROL', 'SIMULATION_TSTOP', '7200.0')
        
        updated_config = config_manager.get_config()
        assert updated_config['TIME_CONTROL']['SIMULATION_TSTOP'] == '7200.0'
        assert original_config['TIME_CONTROL']['SIMULATION_TSTOP'] != updated_config['TIME_CONTROL']['SIMULATION_TSTOP']
        
        self.logger.info("Config manager update_config method successful")

    def test_config_manager_reset_config(self, config_manager):
        self.logger.info("Testing config manager reset_config method")
        original_config = config_manager.get_config()
        
        # Make some changes
        config_manager.update_config('TIME_CONTROL', 'SIMULATION_TSTOP', '7200.0')
        config_manager.update_config('INPUTS', 'NEW_PARAM', 'test_value')
        
        # Reset the config
        config_manager.reset_config()
        
        reset_config = config_manager.get_config()
        assert reset_config == original_config
        assert 'NEW_PARAM' not in reset_config['INPUTS']
        assert reset_config['TIME_CONTROL']['SIMULATION_TSTOP'] == original_config['TIME_CONTROL']['SIMULATION_TSTOP']
        
        self.logger.info("Config manager reset_config method successful")

    def test_config_manager_update_from_simulation_state(self, config_manager, create_mock_state, config):
        self.logger.info("Testing config manager update from simulation state")
        
        # Create a mock simulation state with custom config
        custom_config = {
            'INPUTS': {'FUELS_AND_TOPOGRAPHY_DIRECTORY': '/custom/path'},
            'TIME_CONTROL': {'SIMULATION_TSTOP': '5400.0'},
        }
        mock_state = create_mock_state(config, custom_config=custom_config)
        
        # Update config from the mock state
        config_manager.update_from_simulation_state(mock_state)
        
        updated_config = config_manager.get_config()
        assert updated_config['INPUTS']['FUELS_AND_TOPOGRAPHY_DIRECTORY'] == '/custom/path'
        assert updated_config['TIME_CONTROL']['SIMULATION_TSTOP'] == '5400.0'
        
        self.logger.info("Config manager update from simulation state successful")

    def test_config_manager_generate_elmfire_data_in(self, config_manager, tmp_path):
        self.logger.info("Testing config manager generate_elmfire_data_in method")
        
        output_file = tmp_path / "elmfire.data.in"
        config_manager.generate_elmfire_data_in(output_file)
        
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            content = f.read()
            assert '&INPUTS' in content
            assert '&OUTPUTS' in content
            assert '&TIME_CONTROL' in content
        
        self.logger.info("Config manager generate_elmfire_data_in method successful")

    def test_config_manager_complex_scenario(self, config_manager, create_mock_state, config, tmp_path):
        self.logger.info("Testing config manager in a complex scenario")
        
        # 1. Get the original config
        original_config = config_manager.get_config()
        
        # 2. Update multiple parameters
        config_manager.update_config('TIME_CONTROL', 'SIMULATION_TSTOP', 10800.0)
        config_manager.update_config('INPUTS', 'DT_METEOROLOGY', 1800.0)
        config_manager.update_config('OUTPUTS', 'DTDUMP', 900.0)
        
        # Verify the updates
        updated_config = config_manager.get_config()
        assert updated_config['TIME_CONTROL']['SIMULATION_TSTOP'] == 10800.0
        assert updated_config['INPUTS']['DT_METEOROLOGY'] == 1800.0
        assert updated_config['OUTPUTS']['DTDUMP'] == 900.0
        
        # 3. Create a mock state with different parameters
        custom_config = {
            'INPUTS': {'FUELS_AND_TOPOGRAPHY_DIRECTORY': '/complex/path', 'NEW_PARAM': 'new_value'},
            'TIME_CONTROL': {'SIMULATION_TSTOP': 14400.0},
            'OUTPUTS': {'NEW_OUTPUT': 'output.txt'},
        }
        mock_state = create_mock_state(config, custom_config=custom_config)
        
        # 4. Update config from the mock state
        config_manager.update_from_simulation_state(mock_state)
        
        # 5. Generate elmfire.data.in
        output_file = tmp_path / "complex_elmfire.data.in"
        config_manager.generate_elmfire_data_in(str(output_file))
        
        # 6. Verify the final config
        final_config = config_manager.get_config()
        assert final_config['INPUTS']['FUELS_AND_TOPOGRAPHY_DIRECTORY'] == '/complex/path'
        assert final_config['INPUTS']['NEW_PARAM'] == 'new_value'
        assert final_config['INPUTS']['DT_METEOROLOGY'] == 1800.0
        assert final_config['TIME_CONTROL']['SIMULATION_TSTOP'] == 14400.0
        assert final_config['OUTPUTS']['DTDUMP'] == 900.0
        assert final_config['OUTPUTS']['NEW_OUTPUT'] == 'output.txt'
        
        # 7. Verify the generated file
        assert output_file.exists()
        with open(output_file, 'r') as f:
            content = f.read()
            assert 'FUELS_AND_TOPOGRAPHY_DIRECTORY = \'/complex/path\'' in content
            assert 'NEW_PARAM = \'new_value\'' in content
            assert 'DT_METEOROLOGY = 1800.0' in content
            assert 'SIMULATION_TSTOP = 14400.0' in content
            assert 'DTDUMP = 900.0' in content
            assert 'NEW_OUTPUT = \'output.txt\'' in content

        self.logger.info("Config manager complex scenario test successful")
        
        # 8. Reset the config and verify
        config_manager.reset_config()
        reset_config = config_manager.get_config()
        assert reset_config == original_config
        
        self.logger.info("Config manager complex scenario test completed")
    def test_data_manager_state_operations(self, data_manager, mock_state):
        self.logger.info("Testing data manager state operations")
        data_manager.clear_all_episode_data()
        # Test updating state
        data_manager.update_state(mock_state)
        current_state = data_manager.get_current_state()
        assert current_state is not None
        assert isinstance(current_state, SimulationState)
        assert current_state.timestamp == mock_state.timestamp
        self.logger.info("State update and retrieval successful")

        # Test getting state history
        history = data_manager.get_state_history()
        assert len(history) == 1
        assert history[0] == current_state
        self.logger.info("State history retrieval successful")

        # Test resetting data manager
        data_manager.reset()
        assert data_manager.get_current_state() is None
        assert len(data_manager.get_state_history()) == 0
        self.logger.info("Data manager reset successful")

    def test_data_manager_episode_operations(self, data_manager, mock_state, mock_action):
        self.logger.info("Testing data manager episode operations")
        # Start a new episode
        data_manager.start_new_episode()
        self.logger.info("New episode started")

        # Add a step to the current episode
        data_manager.add_step_to_current_episode(mock_state, mock_action, 1.0, mock_state, False)
        self.logger.info("Step added to current episode")

        # Get the current episode
        current_episode = data_manager.get_current_episode()
        assert current_episode is not None
        assert len(current_episode.steps) == 1
        assert current_episode.steps[0].state == mock_state
        assert current_episode.steps[0].action == mock_action
        self.logger.info("Current episode retrieval and verification successful")

    def test_data_manager_with_real_data(self, data_manager, config):
        self.logger.info("Testing data manager with real data")
        
        # Get the data directory from the config, or use a default value
        data_dir = Path(config.get_config().get('data_management', {}).get('data_directory', 'data'))
        mock_input_dir = data_dir / 'mock' / 'inputs'
        mock_output_dir = data_dir / 'mock' / 'outputs'

        # Create a SimulationState with real data
        real_state = SimulationState(
            timestamp=datetime.now(),
            config=SimulationConfig(sections={
                'INPUTS': {'FUELS_AND_TOPOGRAPHY_DIRECTORY': str(mock_input_dir)},
                'TIME_CONTROL': {'SIMULATION_TSTOP': '3600'},
            }),
            paths=SimulationPaths(
                input_paths=InputPaths(
                    fuels_and_topography_directory=mock_input_dir,
                    asp_filename=mock_input_dir / 'asp.tif',
                    cbd_filename=mock_input_dir / 'cbd.tif',
                    cbh_filename=mock_input_dir / 'cbh.tif',
                    cc_filename=mock_input_dir / 'cc.tif',
                    ch_filename=mock_input_dir / 'ch.tif',
                    dem_filename=mock_input_dir / 'dem.tif',
                    fbfm_filename=mock_input_dir / 'fbfm40.tif',
                    slp_filename=mock_input_dir / 'slp.tif',
                    adj_filename=mock_input_dir / 'adj.tif',
                    phi_filename=mock_input_dir / 'new_phi.tif',
                    weather_directory=mock_input_dir,
                    ws_filename=mock_input_dir / 'ws.tif',
                    wd_filename=mock_input_dir / 'wd.tif',
                    m1_filename=mock_input_dir / 'm1.tif',
                    m10_filename=mock_input_dir / 'm10.tif',
                    m100_filename=mock_input_dir / 'm100.tif',
                    fire=mock_input_dir / 'fire.shp',
                    vegetation=mock_input_dir / 'veg.tif',
                    elevation=mock_input_dir / 'dem.tif',
                    wind=mock_input_dir / 'wind.tif',
                    fuel_moisture=mock_input_dir / 'fuel_moisture.tif'
                ),
                output_paths=OutputPaths(
                    time_of_arrival=mock_output_dir / 'time_of_arrival.tif',
                    fire_intensity=mock_output_dir / 'flin.tif',
                    flame_length=mock_output_dir / 'flame_length.tif',
                    spread_rate=mock_output_dir / 'spread_rate.tif'
                )
            ),
            metrics=SimulationMetrics(
                burned_area=1000.0,
                fire_perimeter_length=500.0,
                containment_percentage=20.0,
                execution_time=120.0,
                performance_metrics={'cpu_usage': 80.0, 'memory_usage': 4000.0},
                fire_intensity=np.random.rand(100, 100)
            ),
            save_path=data_dir / 'mock' / 'save',
            resources={'firefighters': 50, 'trucks': 10},
            weather={'temperature': 25.0, 'wind_speed': 10.0, 'wind_direction': 180.0}
        )

        data_manager.reset()

        # Test updating state with real data
        self.logger.info(f"Updating state at timestamp: {real_state.timestamp}")
        data_manager.update_state(real_state)
        current_state = data_manager.get_current_state()
        self.logger.info(f"Current state retrieved at timestamp: {current_state.timestamp}")
        assert current_state is not None
        assert isinstance(current_state, SimulationState)
        assert current_state.timestamp == real_state.timestamp

        # Test saving and retrieving state
        self.logger.info(f"Saving state to disk at timestamp: {current_state.timestamp}")
        data_manager.save_state_to_disk(current_state)
        self.logger.info(f"Loading state from disk for timestamp: {current_state.timestamp.isoformat()}")
        #retrieved_state = data_manager.load_state_from_disk(current_state.timestamp.isoformat())
        retrieved_state = data_manager.load_state_from_disk(0, 1)  # episode 0, step 1

        self.logger.info(f"Retrieved state from disk at timestamp: {retrieved_state.timestamp if retrieved_state else 'None'}")
        assert retrieved_state is not None
        assert retrieved_state.timestamp == current_state.timestamp
        assert retrieved_state.config.sections == current_state.config.sections
        assert retrieved_state.paths.input_paths.fuels_and_topography_directory == current_state.paths.input_paths.fuels_and_topography_directory
        self.logger.info("All assertions passed for state saving and retrieval")

        # Test getting state history
        history = data_manager.get_state_history()
        assert len(history) == 1
        assert history[0] == current_state
        self.logger.info("State history verified")

    def test_data_manager_save_and_retrieve_multiple_states(self, data_manager, config, create_mock_state):
        self.logger.info("Testing saving and retrieving multiple states")
        data_manager.reset()  # Add this line

        # Create multiple states
        states = []
        for i in range(5):
            custom_config = {'TEST': {'VALUE': f'test_{i}'}}
            custom_metrics = {
                'burned_area': 100.0 * i,
                'fire_perimeter_length': 50.0 * i,
                'containment_percentage': 10.0 * i,
                'execution_time': 60.0 * i,
                'performance_metrics': {'cpu_usage': 50.0 + i, 'memory_usage': 2000.0 + i * 100},
                'fire_intensity': np.random.rand(50, 50)
            }
            custom_resources = {'firefighters': 50 + i * 10, 'trucks': 10 + i}
            custom_weather = {'temperature': 25.0 + i, 'wind_speed': 10.0 + i * 0.5, 'wind_direction': 180.0 + i * 10}
            
            state = create_mock_state(
                config,
                timestamp=datetime.now() + timedelta(hours=i),
                custom_config=custom_config,
                custom_metrics=custom_metrics,
                custom_resources=custom_resources,
                custom_weather=custom_weather
            )
            states.append(state)
            data_manager.update_state(state)
            self.logger.info(f"Saved state {i}")

        # Retrieve and verify states
        for i, original_state in enumerate(states):
            retrieved_state = data_manager.load_state_from_disk(data_manager.get_episode_id(), i+1)
            assert retrieved_state is not None, f"Failed to retrieve state {i}"
            assert retrieved_state.timestamp == original_state.timestamp
            assert retrieved_state.config.sections == original_state.config.sections
            assert retrieved_state.paths.input_paths.fuels_and_topography_directory == original_state.paths.input_paths.fuels_and_topography_directory
            assert retrieved_state.metrics.burned_area == original_state.metrics.burned_area
            assert np.array_equal(retrieved_state.metrics.fire_intensity, original_state.metrics.fire_intensity)
            self.logger.info(f"Successfully verified state {i}")

        # Verify state history
        history = data_manager.get_state_history()
        #logg the len of history
        self.logger.info(f"State history length: {len(history)}")
        
        assert len(history) == 5, "Incorrect number of states in history"
        self.logger.info("State history verified")

    def test_data_manager_episode_management(self, data_manager, config, create_mock_state):
        self.logger.info("Testing episode management")
        
        # Start a new episode
        data_manager.start_new_episode()
        self.logger.info("Started new episode")

        # Create and add steps to the episode
        for i in range(3):
            custom_config = {'EPISODE_TEST': {'STEP': f'step_{i}'}}
            custom_metrics = {
                'burned_area': 100.0 * i,
                'fire_perimeter_length': 50.0 * i,
                'containment_percentage': 10.0 * i,
                'execution_time': 60.0 * i,
                'performance_metrics': {'cpu_usage': 50.0 + i, 'memory_usage': 2000.0 + i * 100},
                'fire_intensity': np.random.rand(30, 30)
            }
            custom_resources = {'firefighters': 50 + i * 10, 'trucks': 10 + i}
            custom_weather = {'temperature': 25.0 + i, 'wind_speed': 10.0 + i * 0.5, 'wind_direction': 180.0 + i * 10}
            
            state = create_mock_state(
                config,
                timestamp=datetime.now() + timedelta(hours=i),
                custom_config=custom_config,
                custom_metrics=custom_metrics,
                custom_resources=custom_resources,
                custom_weather=custom_weather
            )
            action = Action(fireline_coordinates=[(j, j) for j in range(i+1)])
            data_manager.add_step_to_current_episode(state, action, reward=i * 0.5, next_state=state, done=(i == 2))
            self.logger.info(f"Added step {i} to episode")

        # Verify current episode
        current_episode = data_manager.get_current_episode()
        assert current_episode is not None, "Failed to retrieve current episode"
        assert len(current_episode.steps) == 3, "Incorrect number of steps in episode"
        assert current_episode.total_steps == 3
        assert current_episode.total_reward == 0.0 + 0.5 + 1.0
        self.logger.info("Current episode verified")


    def test_data_manager_reset_and_state_persistence(self, config, data_manager, create_mock_state):
        self.logger.info("Testing data manager reset and state persistence")

        # Clear all data at the start of the test
        data_manager.clear_all_data()

        # Create and add a state
        state1 = create_mock_state(config, timestamp=datetime.now())
        data_manager.update_state(state1)
        assert data_manager.get_current_state() == state1
        assert len(data_manager.get_state_history()) == 1

        # Reset the data manager (should not delete files)
        data_manager.reset()
        assert data_manager.get_current_state() is None
        assert len(data_manager.get_state_history()) == 0

        # Verify that the previously saved state can still be loaded
        loaded_state = data_manager.load_state_from_disk(0, 1)
        assert loaded_state is not None
        assert loaded_state.timestamp == state1.timestamp

        self.logger.info("Data manager reset and state persistence test completed")

    def test_data_manager_multiple_episodes(self,config, data_manager, create_mock_state, mock_action):
        self.logger.info("Testing data manager with multiple episodes")
        
        # Reset the data manager
        data_manager.reset()

        # Create two episodes
        for episode in range(2):
            data_manager.start_new_episode()
            for step in range(3):
                state = create_mock_state(config, timestamp=datetime.now() + timedelta(hours=step))
                data_manager.add_step_to_current_episode(state, mock_action, reward=step * 0.5, next_state=state, done=(step == 2))

        # Verify episodes
        assert len(data_manager.state_manager.episodes) == 2
        for episode_id, episode in data_manager.state_manager.episodes.items():
            assert episode.total_steps == 3
            assert episode.total_reward == 0.0 + 0.5 + 1.0

        # Verify that we can retrieve states from both episodes
        first_episode_state = data_manager.get_state_by_episode_step(1, 0)
        second_episode_state = data_manager.get_state_by_episode_step(2, 0)
        assert first_episode_state is not None
        assert second_episode_state is not None
        assert first_episode_state != second_episode_state

        self.logger.info("Multiple episodes test completed")

    def test_data_manager_edge_cases(self, config, data_manager, create_mock_state, mock_action):
        self.logger.info("Testing data manager edge cases")
        
        # Reset the data manager
        data_manager.reset()

        # Test adding a step without starting an episode
        state = create_mock_state(config, timestamp=datetime.now())
        with pytest.raises(Exception):  # Adjust the exception type if needed
            data_manager.add_step_to_current_episode(state, mock_action, reward=1.0, next_state=state, done=False)

        # Start an episode and add a step
        data_manager.start_new_episode()
        data_manager.add_step_to_current_episode(state, mock_action, reward=1.0, next_state=state, done=False)

        # Test retrieving a non-existent state
        non_existent_state = data_manager.get_state_by_episode_step(999, 0)
        assert non_existent_state is None

        # Test loading a non-existent state from disk
        non_existent_loaded_state = data_manager.load_state_from_disk(999, 0)
        assert non_existent_loaded_state is None

        # Test starting a new episode without finishing the current one
        data_manager.start_new_episode()
        assert len(data_manager.state_manager.episodes) == 2
        assert data_manager.state_manager.current_episode_id == 2  # Assuming it starts from 0

        self.logger.info("Edge cases test completed")


    def test_data_manager_save_and_retrieve_multiple_states(self, data_manager, config, create_mock_state):
        self.logger.info("Testing saving and retrieving multiple states")
        data_manager.reset()  # Add this line

        # Create multiple states
        states = []
        for i in range(5):
            custom_config = {'TEST': {'VALUE': f'test_{i}'}}
            custom_metrics = {
                'burned_area': 100.0 * i,
                'fire_perimeter_length': 50.0 * i,
                'containment_percentage': 10.0 * i,
                'execution_time': 60.0 * i,
                'performance_metrics': {'cpu_usage': 50.0 + i, 'memory_usage': 2000.0 + i * 100},
                'fire_intensity': np.random.rand(50, 50)
            }
            custom_resources = {'firefighters': 50 + i * 10, 'trucks': 10 + i}
            custom_weather = {'temperature': 25.0 + i, 'wind_speed': 10.0 + i * 0.5, 'wind_direction': 180.0 + i * 10}
            
            state = create_mock_state(
                config,
                timestamp=datetime.now() + timedelta(hours=i),
                custom_config=custom_config,
                custom_metrics=custom_metrics,
                custom_resources=custom_resources,
                custom_weather=custom_weather
            )
            states.append(state)
            data_manager.update_state(state)
            self.logger.info(f"Saved state {i}")

        # Retrieve and verify states
        for i, original_state in enumerate(states):
            retrieved_state = data_manager.load_state_from_disk(data_manager.get_episode_id(), i+1)
            assert retrieved_state is not None, f"Failed to retrieve state {i}"
            assert retrieved_state.timestamp == original_state.timestamp
            assert retrieved_state.config.sections == original_state.config.sections
            assert retrieved_state.paths.input_paths.fuels_and_topography_directory == original_state.paths.input_paths.fuels_and_topography_directory
            assert retrieved_state.metrics.burned_area == original_state.metrics.burned_area
            assert np.array_equal(retrieved_state.metrics.fire_intensity, original_state.metrics.fire_intensity)
            self.logger.info(f"Successfully verified state {i}")

        # Verify state history
        history = data_manager.get_state_history()
        #logg the len of history
        self.logger.info(f"State history length: {len(history)}")
        
        assert len(history) == 5, "Incorrect number of states in history"
        self.logger.info("State history verified")

    def test_data_manager_episode_management(self, data_manager, config, create_mock_state):
        self.logger.info("Testing episode management")
        
        # Start a new episode
        data_manager.start_new_episode()
        self.logger.info("Started new episode")

        # Create and add steps to the episode
        for i in range(3):
            custom_config = {'EPISODE_TEST': {'STEP': f'step_{i}'}}
            custom_metrics = {
                'burned_area': 100.0 * i,
                'fire_perimeter_length': 50.0 * i,
                'containment_percentage': 10.0 * i,
                'execution_time': 60.0 * i,
                'performance_metrics': {'cpu_usage': 50.0 + i, 'memory_usage': 2000.0 + i * 100},
                'fire_intensity': np.random.rand(30, 30)
            }
            custom_resources = {'firefighters': 50 + i * 10, 'trucks': 10 + i}
            custom_weather = {'temperature': 25.0 + i, 'wind_speed': 10.0 + i * 0.5, 'wind_direction': 180.0 + i * 10}
            
            state = create_mock_state(
                config,
                timestamp=datetime.now() + timedelta(hours=i),
                custom_config=custom_config,
                custom_metrics=custom_metrics,
                custom_resources=custom_resources,
                custom_weather=custom_weather
            )
            action = Action(fireline_coordinates=[(j, j) for j in range(i+1)])
            data_manager.add_step_to_current_episode(state, action, reward=i * 0.5, next_state=state, done=(i == 2))
            self.logger.info(f"Added step {i} to episode")

        # Verify current episode
        current_episode = data_manager.get_current_episode()
        assert current_episode is not None, "Failed to retrieve current episode"
        assert len(current_episode.steps) == 3, "Incorrect number of steps in episode"
        assert current_episode.total_steps == 3
        assert current_episode.total_reward == 0.0 + 0.5 + 1.0
        self.logger.info("Current episode verified")

        # Load episode from disk and verify
        loaded_episode = data_manager.state_manager.load_episode_from_disk(data_manager.state_manager.current_episode_id)
        assert loaded_episode is not None, "Failed to load episode from disk"
        assert len(loaded_episode.steps) == 3, "Incorrect number of steps in loaded episode"
        assert loaded_episode.total_steps == current_episode.total_steps
        assert loaded_episode.total_reward == current_episode.total_reward
        self.logger.info("Loaded episode verified")

        # Verify episode summary
        episode_summary = data_manager.get_episode_summary(data_manager.state_manager.current_episode_id)
        assert episode_summary is not None, "Failed to retrieve episode summary"
        assert episode_summary['total_steps'] == 3
        assert episode_summary['total_reward'] == 0.0 + 0.5 + 1.0
        self.logger.info("Episode summary verified")



if __name__ == "__main__":
    pytest.main([__file__, "-v"])