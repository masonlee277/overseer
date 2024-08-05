import pytest
from pathlib import Path
import sys
from datetime import datetime
import numpy as np

# Add the src directory to the Python path
src_dir = Path(__file__).resolve().parent.parent / 'src'
sys.path.append(str(src_dir))

from overseer.config.config import OverseerConfig
from overseer.utils.logging import OverseerLogger
from overseer.elmfire.data_in_handler import ElmfireDataInHandler
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.data.geospatial_manager import GeoSpatialManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.data.data_manager import DataManager
from overseer.core.models import SimulationState, Action

@pytest.fixture(scope="session")
def config():
    return OverseerConfig()

@pytest.fixture(scope="session")
def logger():
    return OverseerLogger().get_logger('TestElmfireSetup')

@pytest.fixture(scope="session")
def data_manager(config):
    return DataManager(config)

@pytest.fixture(scope="session")
def action_space(config, data_manager):
    return ActionSpace(config, data_manager)

@pytest.fixture(scope="session")
def geospatial_manager(config):
    return GeoSpatialManager(config)

@pytest.fixture(scope="session")
def data_in_handler(config, logger):
    handler = ElmfireDataInHandler(config, logger)
    handler.load_elmfire_data_in()
    return handler

@pytest.fixture(scope="session")
def config_manager(config, data_manager):
    return ElmfireConfigManager(config, data_manager)

@pytest.fixture
def mock_state():
    return SimulationState(
        timestamp=datetime.now(),
        fire_intensity=np.random.rand(100, 100),
        burned_area=1000.0,
        fire_perimeter_length=500.0,
        containment_percentage=20.0,
        resources={'firefighters': 50, 'trucks': 10},
        weather={'temperature': 25.0, 'wind_speed': 10.0, 'wind_direction': 180.0}
    )

@pytest.fixture(scope="function")
def mock_episode_step(mock_state):
    from overseer.core.models import EpisodeStep, SimulationState
    return EpisodeStep(
        step=0,
        state=SimulationState(**mock_state),
        action=None,
        reward=0,
        next_state=None,
        simulation_result=None,
        done=False
    )


@pytest.fixture(scope="function")
def mock_simulation_config():
    from overseer.core.models import SimulationConfig, InputPaths, OutputPaths
    return SimulationConfig(
        elmfire_data={
            'INPUTS': {'FUELS_AND_TOPOGRAPHY_DIRECTORY': '/data/mock/inputs'},
            'TIME_CONTROL': {'SIMULATION_TSTOP': '3600'},
        },
        input_paths=InputPaths(
            fuels_and_topography_directory='/data/mock/inputs',
            asp_filename='asp.tif',
            cbd_filename='cbd.tif',
            cbh_filename='cbh.tif',
            cc_filename='cc.tif',
            ch_filename='ch.tif',
            dem_filename='dem.tif',
            fbfm_filename='fbfm.tif',
            slp_filename='slp.tif',
            adj_filename='adj.tif',
            phi_filename='phi.tif',
            weather_directory='/data/mock/inputs',
            ws_filename='ws.tif',
            wd_filename='wd.tif',
            m1_filename='m1.tif',
            m10_filename='m10.tif',
            m100_filename='m100.tif',
            fire='fire.shp',
            vegetation='veg.tif',
            elevation='dem.tif',
            wind='wind.tif',
            fuel_moisture='fuel_moisture.tif'
        ),
        output_paths=OutputPaths(
            time_of_arrival='toa.tif',
            fire_intensity='intensity.tif',
            flame_length='flame_length.tif',
            spread_rate='spread_rate.tif'
        )
    )

@pytest.fixture(scope="function")
def mock_action():
    return [50, 50, 0, 5]  # x, y, direction, length