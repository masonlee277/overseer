import pytest
import numpy as np
from datetime import datetime
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import sys

# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.append(str(src_dir))
import pytest
import numpy as np
from datetime import datetime
from pathlib import Path

from overseer.config.config import OverseerConfig
from overseer.data.data_manager import DataManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.rl.rewards.reward_manager import RewardManager
from overseer.rl.envs.elmfire_gym_env import ElmfireGymEnv
from overseer.core.models import SimulationState, Action, SimulationMetrics
from overseer.elmfire.simulation_manager import SimulationManager

@pytest.fixture
def config():
    return OverseerConfig()

@pytest.fixture
def data_manager(config):
    return DataManager(config)

@pytest.fixture
def action_space(config, data_manager):
    return ActionSpace(config, data_manager)
a

@pytest.fixture
def reward_manager(config, data_manager):
    return RewardManager(config, data_manager)

@pytest.fixture
def gym_env(config, data_manager, action_space, reward_manager):
    return ElmfireGymEnv(config)

def test_gym_initialization(gym_env):
    assert isinstance(gym_env, ElmfireGymEnv)
    assert gym_env.action_space is not None
    assert gym_env.observation_space is not None
    assert gym_env.reward_range == (-float('inf'), float('inf'))

def test_action_space(action_space):
    assert isinstance(action_space, ActionSpace)
    sample_action = action_space.sample()
    assert action_space.contains(sample_action)

def test_reward_manager(reward_manager, data_manager, create_mock_state, config):
    mock_state = create_mock_state(config)
    data_manager.update_state(mock_state)
    reward = reward_manager.calculate_reward(mock_state)
    assert isinstance(reward, float)

def test_gym_reset(gym_env):
    observation = gym_env.reset()
    assert observation is not None
    assert isinstance(observation, np.ndarray)

def test_gym_step_without_simulation(gym_env, action_space):
    gym_env.reset()
    sample_action = action_space.sample()
    observation, reward, done, info = gym_env.step(sample_action)
    assert observation is not None
    assert isinstance(observation, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_gym_env_apply_action(gym_env, action_space):
    """
    Test the apply_action method of ElmfireGymEnv.
    This method should apply an action and return a new state.
    """
    gym_env.reset()
    initial_state = gym_env.data_manager.get_current_state()
    action = action_space.sample()
    new_state, reward, done, _ = gym_env.step(action)
    
    assert new_state is not None
    assert isinstance(new_state, np.ndarray)
    assert reward is not None
    assert isinstance(done, bool)
    assert gym_env.data_manager.get_current_state() != initial_state

def test_data_manager_episode_management(config, create_mock_state):
    """
    Test the episode management functionality of DataManager.
    This includes starting a new episode, adding steps, and retrieving episode data.
    """
    data_manager = DataManager(config)
    data_manager.start_new_episode()
    
    # Create a mock state and action using the fixture
    mock_state = create_mock_state(config)
    mock_action = Action(fireline_coordinates=[(1, 1), (2, 2)])
    
    data_manager.add_step_to_current_episode(mock_state, mock_action, 1.0, mock_state, False)
    
    current_episode = data_manager.get_current_episode()
    assert current_episode is not None
    assert len(current_episode.steps) == 1
    assert current_episode.total_reward == 1.0

def test_simulation_manager_reset(config, config_manager, data_manager):
    """
    Test the reset_simulation method of SimulationManager.
    This method should reset the simulation to its initial state, but metrics may be updated.
    """
    sim_manager = SimulationManager(config, config_manager, data_manager)
    initial_state = sim_manager.reset_simulation()
    
    assert initial_state is not None
    current_state = sim_manager.get_state()
    print(f"Current state TEST TEST: {current_state}")
    assert current_state is not None
    
    # Compare non-metric attributes
    assert current_state.timestamp == initial_state.timestamp
    assert current_state.config == initial_state.config
    assert current_state.paths == initial_state.paths
    assert current_state.resources == initial_state.resources
    assert current_state.weather == initial_state.weather
    
    # Check that metrics are present but may have been updated
    assert hasattr(current_state, 'metrics')
    assert isinstance(current_state.metrics, SimulationMetrics)
    
    # Optionally, you can add more specific checks for metrics if there are
    # certain values or ranges you expect after a reset
    
    assert sim_manager.config_manager.get_config_for_simulation() == initial_state.config


def test_data_manager_state_persistence(config, create_mock_state):
    """
    Test the state persistence functionality of DataManager.
    This includes saving a state, loading it, and verifying its contents.
    """
    data_manager = DataManager(config)
    
    # Create a mock state
    mock_state = create_mock_state(config)
    
    # Start a new episode and add the mock state as a step
    data_manager.start_new_episode()
    data_manager.add_step_to_current_episode(mock_state, Action(fireline_coordinates=[]), 0.0, mock_state, False)
    
    # Get the current episode and step
    current_episode = data_manager.get_current_episode()
    assert current_episode is not None
    episode_id = current_episode.episode_id
    step = 0  # First step in the episode
    
    # Save the state to disk
    data_manager.save_state_to_disk(mock_state)
    
    # Load the state from disk
    loaded_state = data_manager.load_state_from_disk(episode_id, step)
    
    # Verify the loaded state matches the original
    assert loaded_state is not None
    assert loaded_state.timestamp == mock_state.timestamp
    assert loaded_state.config.sections == mock_state.config.sections
    assert loaded_state.paths == mock_state.paths
    assert loaded_state.metrics.__dict__ == mock_state.metrics.__dict__
    assert loaded_state.resources == mock_state.resources
    assert loaded_state.weather == mock_state.weather

def test_simulation_manager_apply_action(config, config_manager, data_manager, create_mock_state):
    """
    Test the apply_action method of SimulationManager.
    This method should apply an action and return a new state.
    """
    sim_manager = SimulationManager(config, config_manager, data_manager)
    
    # Create an initial state
    initial_state = create_mock_state(config)
    sim_manager.data_manager.update_state(initial_state)
    
    # Create a mock action
    action = Action(fireline_coordinates=[(1, 1), (2, 2), (3, 3)])
    
    # Apply the action
    new_state, done = sim_manager.apply_action(action)
    
    # Verify the new state
    assert new_state is not None
    assert isinstance(new_state, SimulationState)
    assert new_state != initial_state
    assert done is False  # Assuming the simulation is not complete after one action

def test_gym_env_episode_management(gym_env, action_space):
    """
    Test the episode management functionality of ElmfireGymEnv.
    This includes resetting the environment, taking steps, and verifying episode data.
    """
    # Reset the environment to start a new episode
    initial_obs = gym_env.reset()
    assert initial_obs is not None
    
    # Take a few steps
    for _ in range(3):
        action = action_space.sample()
        obs, reward, done, info = gym_env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    # Verify episode data
    episode = gym_env.data_manager.get_current_episode()
    assert episode is not None
    assert episode.episode_id == 0  # Assuming this is the first episode
    assert len(episode.steps) == 3
    assert episode.total_steps == 3
    assert episode.total_reward == sum(step.reward for step in episode.steps)

    # Verify that each step in the episode has the correct structure
    for step in episode.steps:
        assert isinstance(step.state, SimulationState)
        assert isinstance(step.action, Action)
        assert isinstance(step.reward, float)
        assert isinstance(step.next_state, SimulationState)
        assert isinstance(step.done, bool)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])