import pytest
from pathlib import Path
from datetime import datetime
import numpy as np
# Add the src directory to the Python path
import sys
src_dir = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_dir))

import os
from overseer.config.config import OverseerConfig
from overseer.data.data_manager import DataManager
from overseer.core.models import SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics, InputPaths, OutputPaths, Action

@pytest.fixture
def data_manager(config):
    return DataManager(config)

def test_data_manager_state_operations(data_manager, mock_state):
    # Test updating state
    data_manager.update_state(mock_state)
    current_state = data_manager.get_current_state()
    assert current_state is not None
    assert isinstance(current_state, SimulationState)
    assert current_state.timestamp == mock_state.timestamp

    # Test getting state history
    history = data_manager.get_state_history()
    assert len(history) == 1
    assert history[0] == current_state

    # Test resetting data manager
    data_manager.reset()
    assert data_manager.get_current_state() is None
    assert len(data_manager.get_state_history()) == 0

def test_data_manager_episode_operations(data_manager, mock_state, mock_action):
    # Start a new episode
    data_manager.start_new_episode()

    # Add a step to the current episode
    data_manager.add_step_to_current_episode(mock_state, mock_action, 1.0, mock_state, False)

    # Get the current episode
    current_episode = data_manager.get_current_episode()
    assert current_episode is not None
    assert len(current_episode.steps) == 1
    assert current_episode.steps[0].state == mock_state
    assert current_episode.steps[0].action == mock_action


def test_data_manager_with_real_data(data_manager, config):
    print(config.get_config())
    print('$$' * 40)

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

    data_manager.clean_states()
    
    # Test updating state with real data
    print(f"Updating state at timestamp: {real_state.timestamp}")
    data_manager.update_state(real_state)
    current_state = data_manager.get_current_state()
    print(f"Current state retrieved at timestamp: {current_state.timestamp}")
    assert current_state is not None
    assert isinstance(current_state, SimulationState)
    assert current_state.timestamp == real_state.timestamp

    # Test saving and retrieving statein
    print(f"Saving state to disk at timestamp: {current_state.timestamp}")
    data_manager.save_state_to_disk(current_state)
    print(f"Loading state from disk for timestamp: {current_state.timestamp.isoformat()}")
    retrieved_state = data_manager.load_state_from_disk(current_state.timestamp.isoformat())
    print(f"Retrieved state from disk at timestamp: {retrieved_state.timestamp if retrieved_state else 'None'}")
    assert retrieved_state is not None
    assert retrieved_state.timestamp == current_state.timestamp
    assert retrieved_state.config.sections == current_state.config.sections
    assert retrieved_state.paths.input_paths.fuels_and_topography_directory == current_state.paths.input_paths.fuels_and_topography_directory
    print("All assertions passed for state saving and retrieval")

    # Test getting state history
    history = data_manager.get_state_history()
    assert len(history) == 1
    assert history[0] == current_state

if __name__ == "__main__":
    pytest.main([__file__, "-v"])