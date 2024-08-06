import pytest
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import os
import rasterio
from rasterio.transform import from_origin


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
from overseer.core.models import (
    SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics,
    InputPaths, OutputPaths, Action, EpisodeStep, Episode
)

def create_mock_data():
    base_dir = os.path.join('data', 'mock')
    input_dir = os.path.join(base_dir, 'inputs')
    output_dir = os.path.join(base_dir, 'outputs')
    scratch_dir = './scratch'

    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scratch_dir, exist_ok=True)

    # Define common parameters for all GeoTIFF files
    width, height = 100, 100
    transform = from_origin(0, 0, 60, 60)  # 60m resolution as per COMPUTATIONAL_DOMAIN_CELLSIZE
    crs = rasterio.crs.CRS.from_epsg(32610)  # EPSG:32610 - UTM Zone 10N

    # Create input files
    input_files = ['asp', 'cbd', 'cbh', 'cc', 'ch', 'dem', 'fbfm40', 'slp', 'adj', 'new_phi', 'ws', 'wd', 'm1', 'm10', 'm100']
    for filename in input_files:
        data = np.ones((height, width))
        output_path = os.path.join(input_dir, f'{filename}.tif')
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data, 1)
        print(f"Created mock input file: {output_path}")

    # Create output files
    output_files = ['flin', 'spread_rate', 'time_of_arrival']
    for filename in output_files:
        data = np.ones((height, width))
        output_path = os.path.join(output_dir, f'{filename}.tif')
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
            dst.write(data, 1)
        print(f"Created mock output file: {output_path}")

    # Create elmfire.data.in file
    elmfire_data_in_path = os.path.join(base_dir, 'elmfire.data.in')
    with open(elmfire_data_in_path, 'w') as f:
        f.write("&COMPUTATIONAL_DOMAIN\n")
        f.write("  COMPUTATIONAL_DOMAIN_CELLSIZE = 60\n")
        f.write("/\n\n")
        
        f.write("&INPUTS\n")
        f.write(f"  FUELS_AND_TOPOGRAPHY_DIRECTORY = '{input_dir}'\n")
        f.write(f"  WEATHER_DIRECTORY = '{input_dir}'\n")
        f.write("  DT_METEOROLOGY = 3600.0\n")
        f.write("  LH_MOISTURE_CONTENT = 30.0\n")
        f.write("  LW_MOISTURE_CONTENT = 60.0\n")
        for filename in input_files:
            upper_filename = filename.upper()
            if filename == 'fbfm40':
                upper_filename = 'FBFM'
            elif filename == 'new_phi':
                upper_filename = 'PHI'
            f.write(f"  {upper_filename}_FILENAME = '{filename}.tif'\n")
        f.write("/\n\n")
        
        f.write("&OUTPUTS\n")
        f.write(f"  OUTPUTS_DIRECTORY = '{output_dir}'\n")
        f.write("  DTDUMP = 3600.\n")
        f.write("  DUMP_FLIN = .TRUE.\n")
        f.write("  DUMP_SPREAD_RATE = .TRUE.\n")
        f.write("  DUMP_TIME_OF_ARRIVAL = .TRUE.\n")
        f.write("  CONVERT_TO_GEOTIFF = .TRUE.\n")
        f.write("  DUMP_SPOTTING_OUTPUTS = .TRUE.\n")
        f.write("/\n\n")
        
        f.write("&MISCELLANEOUS\n")
        f.write(f"  SCRATCH = '{scratch_dir}'\n")
        f.write("/\n\n")
        
        f.write("&SIMULATOR\n")
        f.write("  NUM_IGNITIONS = 2\n")
        f.write("/\n\n")
        
        f.write("&TIME_CONTROL\n")
        f.write("  SIMULATION_TSTOP = 43200.0\n")
        f.write("/\n")
    print(f"Created mock elmfire.data.in file: {elmfire_data_in_path}")


@pytest.fixture(scope="session", autouse=True)
def create_fake_data():
    create_mock_data()

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

@pytest.fixture
def config_manager(config, data_manager):
    return ElmfireConfigManager(config, data_manager)

@pytest.fixture
def mock_simulation_config(config):
    data_dir = Path(config.get_config().get('data_management', {}).get('data_directory', 'data'))
    mock_input_dir = data_dir / 'mock' / 'inputs'
    return SimulationConfig(sections={
        'INPUTS': {'FUELS_AND_TOPOGRAPHY_DIRECTORY': str(mock_input_dir)},
        'TIME_CONTROL': {'SIMULATION_TSTOP': '3600'},
    })

# ... (keep existing imports)


# ... (keep existing imports and other fixtures)

@pytest.fixture
def create_mock_state():
    def _create_mock_state(config, timestamp=None, custom_config=None, custom_metrics=None, custom_resources=None, custom_weather=None):
        data_dir = Path(config.get_config().get('data_management', {}).get('data_directory', 'data'))
        mock_input_dir = data_dir / 'mock' / 'inputs'
        mock_output_dir = data_dir / 'mock' / 'outputs'

        if timestamp is None:
            timestamp = datetime.now()

        if custom_config is None:
            custom_config = {
                'INPUTS': {'FUELS_AND_TOPOGRAPHY_DIRECTORY': str(mock_input_dir)},
                'TIME_CONTROL': {'SIMULATION_TSTOP': '3600'},
            }

        if custom_metrics is None:
            custom_metrics = {
                'burned_area': 1000.0,
                'fire_perimeter_length': 500.0,
                'containment_percentage': 20.0,
                'execution_time': 120.0,
                'performance_metrics': {'cpu_usage': 80.0, 'memory_usage': 4000.0},
                'fire_intensity': np.random.rand(100, 100)
            }

        if custom_resources is None:
            custom_resources = {'firefighters': 50, 'trucks': 10}

        if custom_weather is None:
            custom_weather = {'temperature': 25.0, 'wind_speed': 10.0, 'wind_direction': 180.0}

        return SimulationState(
            timestamp=timestamp,
            config=SimulationConfig(sections=custom_config),
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
            metrics=SimulationMetrics(**custom_metrics),
            save_path=data_dir / 'mock' / 'save',
            resources=custom_resources,
            weather=custom_weather
        )
    
    return _create_mock_state

@pytest.fixture
def mock_state(config, create_mock_state):
    return create_mock_state(config)

@pytest.fixture(scope="function")
def mock_episode_step(config, create_mock_state):
    mock_state = create_mock_state(config)
    return EpisodeStep(
        step=0,
        state=mock_state,
        action=Action(fireline_coordinates=[(1, 1), (2, 2), (3, 3)]),
        reward=0,
        next_state=mock_state,
        done=False
    )

# ... (keep other fixtures)
@pytest.fixture(scope="function")
def mock_episode(mock_episode_step):
    return Episode(
        episode_id=1,
        steps=[mock_episode_step],
        total_reward=0,
        total_steps=1
    )

@pytest.fixture(scope="function")
def mock_action():
    return Action(fireline_coordinates=[(50, 50), (55, 55), (60, 60)])

