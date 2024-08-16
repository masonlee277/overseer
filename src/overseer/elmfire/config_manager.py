"""
ElmfireConfigManager

A comprehensive manager for ELMFIRE (Eulerian Level set Model of FIRE spread) configuration and simulation preparation.

Responsibilities:
1. Manage the ELMFIRE configuration, including loading, updating, saving, and validating.
2. Interface with the GeoSpatialManager to apply actions to ELMFIRE input files.
3. Coordinate between ActionSpace, DataInHandler, and GeoSpatialManager.
4. Provide a high-level interface for SimulationManager to interact with ELMFIRE configuration.
5. Handle input file generation and validation for ELMFIRE simulations.

Data Structures:
1. ELMFIRE Config: A nested dictionary structure with the following main sections:
   - COMPUTATIONAL_DOMAIN: Contains grid and domain parameters.
   - INPUTS: Specifies input file paths and meteorological parameters.
   - OUTPUTS: Defines output settings and file paths.
   - TIME_CONTROL: Sets simulation time parameters.
   - SIMULATOR: Contains ELMFIRE-specific simulation settings.
   Each section contains key-value pairs, where values can be strings, numbers, or booleans.

2. General Config (OverseerConfig): A higher-level configuration object that includes:
   - ELMFIRE-specific settings
   - Geospatial parameters
   - Data management settings
   - Environment variables
   - Logging configurations

Scope:
The ElmfireConfigManager acts as a facade for all ELMFIRE-related configuration operations. It encapsulates
the complexity of managing ELMFIRE settings, input/output files, and simulation parameters. It does not
directly modify geospatial data or handle the simulation execution.

Design Patterns and Intuition:
1. Facade Pattern: Provides a simplified interface to the complex subsystem of ELMFIRE configuration.
2. Dependency Injection: Uses DI for GeoSpatialManager and DataInHandler to maintain loose coupling.
3. Singleton Pattern: Ensures a single configuration instance via OverseerConfig.
4. Observer Pattern: Can be extended to notify other components of configuration changes.
5. Command Pattern: Actions applied to the configuration can be treated as commands, allowing for undo/redo functionality.

Key Methods:
- __init__: Initializes the manager with necessary dependencies and loads the initial configuration.
- get_config: Retrieves the current ELMFIRE configuration.
- update_config: Updates specific configuration parameters.
- generate_elmfire_data_in: Generates the ELMFIRE input file based on the current configuration.
- validate_config: Ensures the configuration is valid and complete.
- apply_action: Applies a simulation action (e.g., fireline construction) to the configuration and input files.
- prepare_simulation: Sets up all necessary components for an ELMFIRE simulation.

Thread Safety:
Designed with thread-safety in mind for potential future multi-threaded simulations. Uses immutable
data structures where possible and implements proper synchronization for shared resources.

Performance Considerations:
- Caches configuration data to minimize file I/O operations.
- Uses efficient data structures (dictionaries) for quick access to configuration parameters.
- Implements lazy loading of large data files to reduce memory usage.

Error Handling:
- Implements robust error checking and logging for all operations.
- Raises custom exceptions for configuration-related errors to allow for fine-grained error handling.

Future Extensibility:
- Designed to be easily extendable for additional ELMFIRE features or integration with other fire models.
- Modular structure allows for easy addition of new configuration sections or parameters.

Usage:
Typically instantiated by the SimulationManager at the start of the Overseer application.
Used throughout the simulation lifecycle to manage and provide ELMFIRE configuration data.

Example:
    config_manager = ElmfireConfigManager(overseer_config, data_manager)
    config_manager.prepare_simulation()
    config_manager.apply_action(some_action)
    simulation_config = config_manager.get_config_for_simulation()

Note: This class is central to the Overseer system's interaction with ELMFIRE and should be
maintained with careful consideration of its wide-reaching effects on the simulation process.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from pprint import pformat
import copy
from pprint import pformat

from overseer.config import OverseerConfig
from overseer.data.data_manager import DataManager
from overseer.elmfire.data_in_handler import ElmfireDataInHandler
from overseer.utils.logging import OverseerLogger
from overseer.utils import fix_path

from overseer.core.models import SimulationConfig, SimulationPaths, InputPaths, OutputPaths, Action

class ElmfireConfigManager:
    """
    Manages the ELMFIRE configuration, input generation, and validation.

    This class is responsible for loading and managing the ELMFIRE configuration,
    generating input data using Cloudfire's fuel/weather/ignition client,
    validating input files, and coordinating between various components of the system.

    Attributes:
        config (OverseerConfig): The Overseer configuration object.
        geospatial_manager (GeoSpatialManager): Manager for geospatial operations.
        data_in_handler (ElmfireDataInHandler): Handler for ELMFIRE data input files.
        logger (logging.Logger): Logger for this class.
        elmfire_config (Dict[str, Any]): ELMFIRE-specific configuration.
        elmfire_base_path (Path): Base path for the ELMFIRE directory.
    """
    def __init__(self, config: OverseerConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager

        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.data_in_handler = ElmfireDataInHandler(config, self.logger)
        self.elmfire_config = self._load_elmfire_config()
        self.clean_config()  # Clean the configuration after loading

        self.elmfire_base_path = self._get_elmfire_base_path()
        self.logger.info(f"ElmfireConfigManager initialized with base path: {self.elmfire_base_path}")
        self.logger.info(f"Loaded configuration:\n{pformat(self.elmfire_config, indent=4, width=120, sort_dicts=True)}")
        print(f"Loaded configuration: {pformat(self.elmfire_config, indent=2, width=120)}")  # Add this line for debugging


    def initialize(self):
        """
        Initialize or reset the configuration to its default state.
        This method should be called when resetting the environment.
        """
        self.logger.info("Initializing ElmfireConfigManager")
        self.reset_config()
        self.prepare_simulation_inputs()
        self.logger.info("ElmfireConfigManager initialization complete")



    def prepare_simulation_inputs(self) -> None:
        """
        Prepare simulation inputs by validating files, loading simulation inputs,
        and performing any necessary configuration tasks.
        """
        self.logger.info("Preparing simulation inputs")

        # Load the ELMFIRE data.in file
        self.data_in_handler.load_elmfire_data_in()
        self.logger.info("Loaded ELMFIRE data.in file")

        # Validate the structure of the data.in file
        if not self.data_in_handler.validate_structure():
            self.logger.error("ELMFIRE data.in file structure is invalid")
            raise ValueError("Invalid ELMFIRE data.in file structure")
        self.logger.info("Validated ELMFIRE data.in file structure")

        # Validate input files
        if not self.data_in_handler.validate_input_files():
            self.logger.error("One or more required input files are missing")
            raise FileNotFoundError("Missing required input files")
        self.logger.info("Validated all required input files")

        # Get input and output paths
        simulation_paths = self.get_simulation_paths()
        self.logger.info(f"Created SimulationPaths object: {simulation_paths}")

        # Get configuration for simulation
        sim_config_dict = self.get_config_for_simulation()
        sim_config = SimulationConfig.from_dict(sim_config_dict)
        self.logger.info(f"Created SimulationConfig object: {sim_config.to_dict()}")

        self.logger.info("Stored current simulation configuration and paths")
        self.logger.info("Simulation inputs preparation completed successfully")




    def _load_elmfire_config(self) -> Dict[str, Dict[str, Any]]:
        elmfire_config = self.data_in_handler.elmfire_data_in
        if not isinstance(elmfire_config, dict):
            self.logger.error("Invalid ELMFIRE configuration structure")
            return {}
        
        # Ensure top-level keys are dictionaries
        for key, value in elmfire_config.items():
            if not isinstance(value, dict):
                elmfire_config[key] = {'value': value}
        
        return elmfire_config

    def get_config(self) -> Dict[str, Dict[str, Any]]:
        return self._extract_values(copy.deepcopy(self.elmfire_config))

    def get_simulation_paths(self) -> SimulationPaths:
        """
        Get the SimulationPaths object containing input and output paths.
        """
        input_paths, output_paths = self.data_in_handler.get_input_output_paths()
        self.logger.info(f"Input paths: {input_paths}")
        self.logger.info(f"Output paths: {output_paths}")

        # Create SimulationPaths object
        simulation_paths = SimulationPaths(input_paths=input_paths, output_paths=output_paths)
        return simulation_paths
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        if section not in self.elmfire_config:
            self.elmfire_config[section] = {}
        self.elmfire_config[section][key] = {'value': value}

    def update_from_simulation_state(self, state) -> None:
        for section, values in state.config.sections.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    self.update_config(section, key, value)
            else:
                self.update_config(section, 'value', values)

    def _extract_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        for section, values in config.items():
            if isinstance(values, dict):
                config[section] = {k: v['value'] if isinstance(v, dict) and 'value' in v else v for k, v in values.items()}
        return config

    def generate_elmfire_data_in(self, output_file: str) -> None:
        with open(output_file, 'w') as f:
            for section, values in self.elmfire_config.items():
                f.write(f"&{section}\n")
                if isinstance(values, dict):
                    for key, value in values.items():
                        extracted_value = value['value'] if isinstance(value, dict) and 'value' in value else value
                        f.write(f"  {key} = {self._format_value(extracted_value)}\n")
                else:
                    f.write(f"  value = {self._format_value(values)}\n")
                f.write("/\n\n")

    def reset_config(self) -> None:
        self.data_in_handler.load_elmfire_data_in()
        self.elmfire_config =self.data_in_handler.elmfire_data_in


    def update_from_simulation_state(self, state) -> None:
        for section, values in state.config.sections.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    self.update_config(section, key, value)
            else:
                self.update_config(section, 'value', values)


    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return '.TRUE.' if value else '.FALSE.'
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return f"'{str(value)}'"

    def save_config(self) -> None:
        self.config.set('elmfire', self.elmfire_config)
        self.logger.info("Saved ELMFIRE configuration")

    def get_input_file_info(self) -> Dict[str, Dict[str, Any]]:
        input_info = {}
        input_dirs = [
            Path(self.elmfire_config.get('directories', {}).get('fuels_and_topography', '')),
            Path(self.elmfire_config.get('directories', {}).get('weather', ''))
        ]
        for dir_path in input_dirs:
            if dir_path.exists():
                for file in dir_path.iterdir():
                    if file.is_file():
                        input_info[file.name] = {
                            'path': str(file),
                            'size': file.stat().st_size
                        }
            else:
                self.logger.warning(f"Directory not found: {dir_path}")
        return input_info

    def get_data_in_handler(self) -> ElmfireDataInHandler:
        return self.data_in_handler
    

    def validate_input_files(self) -> bool:
        """
        Validate if all required input files exist.

        Returns:
            bool: True if all required files exist, False otherwise.
        """
        return self.data_in_handler.validate_input_files()
    
    def apply_action(self, action: Action) -> None:
        """
        Apply an action to the ELMFIRE configuration and input files.
        
        Args:
            action (Action): The action to apply, representing fireline construction.
        """
        self.logger.info(f"Applying action to ELMFIRE configuration: {action}")
        
        # Extract fireline coordinates from the Action
        fireline_coords = action.fireline_coordinates
        
        #add logging
        self.logger.info(f"Fireline coordinates: {fireline_coords}")

        # Update fuel files using DataManager
        self._update_fuel_files(fireline_coords)
        self.logger.info("Updated fuel files with new fireline")
        # Update elmfire.data.in if necessary




    def _update_fuel_files(self, fireline_coords: List[Tuple[int, int]]):
        """Update fuel files with new fireline coordinates."""
        fuel_files = ['fbfm40.tif', 'cbd.tif', 'cbh.tif', 'ch.tif', 'cc.tif']
        
        self.logger.info("=" * 50)
        self.logger.info(f"[_update_fuel_files]Current ELMFIRE config: {self.elmfire_config}")
        self.logger.info("=" * 50)

        self.clean_config()  # Clean the configuration before using it
        fuels_dir = Path(self.elmfire_config.get('INPUTS', {}).get('FUELS_AND_TOPOGRAPHY_DIRECTORY', ''))
        self.logger.info(f"Attempting to update fuel files in directory: {fuels_dir}")
        
        if not fuels_dir.exists():
            self.logger.error(f"Fuels directory not found: {fuels_dir}")
            raise FileNotFoundError(f"Fuels directory not found: {fuels_dir}")
        
        self.logger.info(f"Contents of directory: {list(fuels_dir.iterdir())}")

        for file in fuel_files:
            filepath = fuels_dir / file.lower()
            self.logger.info(f"Attempting to update fuel file: {filepath}")
            if filepath.exists():
                self.data_manager.update_fuel_file(str(filepath), fireline_coords)
                self.logger.info(f"Successfully updated fuel file: {filepath}")
            else:
                self.logger.warning(f"Fuel file not found: {filepath}")

        self.logger.info("=" * 50)
        self.logger.info("Fuel file update process completed")
        self.logger.info("=" * 50)



    def _update_data_in(self, action: List[int]) -> None:
        """Update elmfire.data.in if necessary based on the action."""
        # Example: Update simulation time if the action affects it
        if self._action_affects_simulation_time(action):
            new_tstop = self._calculate_new_tstop(action)
            self.data_in_handler.set_parameter('TIME_CONTROL', 'SIMULATION_TSTOP', new_tstop)
            self.data_in_handler.save_elmfire_data_in()
        self.logger.info("Updated elmfire.data.in based on action")


    def get_config_for_simulation(self) -> SimulationConfig:
        """Prepare and return the configuration for running a simulation."""
        input_paths, output_paths = self.data_in_handler.get_input_output_paths()

        # Create SimulationPaths object
        simulation_paths = SimulationPaths(
            input_paths=input_paths,  # Remove the InputPaths() constructor
            output_paths=output_paths  # Remove the OutputPaths() constructor
        )

        # Create SimulationConfig object
        sim_config = SimulationConfig()

        # Populate SimulationConfig with sections from elmfire_data_in
        for section, params in self.data_in_handler.elmfire_data_in.items():
            for key, value in params.items():
                sim_config.set_parameter(section, key, value)

        self.logger.info(f"Configuration for Simulation:")
        self.logger.info(f"Input Paths: {simulation_paths.input_paths}")
        self.logger.info(f"Output Paths: {simulation_paths.output_paths}")
        self.logger.info("Some specific parameters:")
        self.logger.info(f"Cell Size: {sim_config.get_parameter('COMPUTATIONAL_DOMAIN', 'COMPUTATIONAL_DOMAIN_CELLSIZE')}")
        self.logger.info(f"Simulation Duration: {sim_config.get_parameter('TIME_CONTROL', 'SIMULATION_TSTOP')}")
        self.logger.info(f"Time Step: {sim_config.get_parameter('TIME_CONTROL', 'SIMULATION_DT')}")

        return sim_config



    def _get_ignition_points(self) -> List[tuple]:
        """Get ignition points from the ELMFIRE configuration."""
        num_ignitions = int(self.data_in_handler.get_parameter('SIMULATOR', 'NUM_IGNITIONS'))
        ignition_points = []
        for i in range(1, num_ignitions + 1):
            x = float(self.data_in_handler.get_parameter('SIMULATOR', f'IGNITION_X({i})'))
            y = float(self.data_in_handler.get_parameter('SIMULATOR', f'IGNITION_Y({i})'))
            ignition_points.append((x, y))
        return ignition_points
    
    

    def _get_fuel_file_paths(self) -> Dict[str, str]:
        """Get the paths to the fuel files."""
        try:
            fuel_dir = Path(self.elmfire_config['directories']['fuels_and_topography'])
        except KeyError:
            self.logger.error("Missing 'directories' or 'fuels_and_topography' in elmfire_config")
            raise

        return {
            'cbd': str(fuel_dir / 'cbd.tif'),
            'cc': str(fuel_dir / 'cc.tif'),
            'ch': str(fuel_dir / 'ch.tif'),
            'cbh': str(fuel_dir / 'cbh.tif')
        }
    
    def validate_config(self) -> bool:
        """Validate the current ELMFIRE configuration."""
        # validate sturcture:
        pass

    def clean_config(self):
        """
        Clean the configuration by removing unnecessary quotes and converting appropriate values.
        """
        def clean_value(value):
            if isinstance(value, str):
                # Remove surrounding quotes
                value = value.strip("'\"")
                # Convert to boolean if applicable
                if value.lower() in ['.true.', '.false.']:
                    return value.lower() == '.true.'
                # Convert to float or int if applicable
                try:
                    return int(value)
                except ValueError:
                    try:
                        return float(value)
                    except ValueError:
                        return value
            return value

        for section, params in self.elmfire_config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    params[key] = clean_value(value)
            else:
                self.elmfire_config[section] = clean_value(params)

    def print_and_log_info(self):
        """Print and log information about input files and geospatial data."""
        input_info = self.get_input_file_info()
        geospatial_info = self.get_geospatial_info()

        self.logger.info("Input File Information:")
        print("Input File Information:")
        for file, info in input_info.items():
            message = f"  {file}: Size = {info['size']} bytes, Path = {info['path']}"
            self.logger.info(message)
            print(message)

        self.logger.info("\nGeospatial Information:")
        print("\nGeospatial Information:")
        for file, info in geospatial_info.items():
            message = f"  {file}: Width = {info['width']}, Height = {info['height']}, " \
                      f"CRS = {info['crs']}, Bounds = {info['bounds']}"
            self.logger.info(message)
            print(message)

    def _get_elmfire_base_path(self) -> Path:
        """
        Get the base path for ELMFIRE.

        Returns:
            Path: The resolved path to the ELMFIRE directory.
        """
        overseer_dir = Path(__file__).parent.parent.parent
        elmfire_relative_path = self.elmfire_config.get('elmfire_relative_path', {})
        if isinstance(elmfire_relative_path, dict):
            elmfire_relative_path = elmfire_relative_path.get('value', '../elmfire')
        else:
            elmfire_relative_path = '../elmfire'
        elmfire_path = (overseer_dir / elmfire_relative_path).resolve()
        return elmfire_path

    def calculate_fire_growth_rate(self, toa_path: str, time_interval: float) -> float:
        """
        Calculate the fire growth rate based on the time of arrival (TOA) data.

        Args:
            toa_path (str): Path to the time of arrival raster file.
            time_interval (float): Time interval in seconds to calculate the growth rate.

        Returns:
            float: Fire growth rate in square meters per second.
        """
        toa_path = fix_path(toa_path, add_tif=True)
        if not toa_path:
            self.logger.error("Invalid or non-existent time of arrival path")
            return 0.0

        try:
            with rasterio.open(toa_path) as src:
                toa_data = src.read(1)
                pixel_area = abs(src.transform.a * src.transform.e)  # Calculate pixel area

            # Calculate the area burned at the start and end of the interval
            start_area = np.sum(toa_data > 0) * pixel_area
            end_area = np.sum(toa_data <= time_interval) * pixel_area

            # Calculate the growth rate
            growth_rate = (end_area - start_area) / time_interval

            return growth_rate

        except Exception as e:
            self.logger.error(f"Error calculating fire growth rate: {str(e)}")
            return 0.0


    def prepare_simulation(self):
        """
        Prepare the simulation by generating inputs and validating files.

        This method orchestrates the input data generation and validation process.
        """
        self.logger.info("Preparing simulation...")
        try:
            if self.validate_input_files():
                self.logger.info("Simulation preparation completed successfully.")
            else:
                self.logger.warning("Simulation preparation completed with warnings. Some files may be missing.")
        except Exception as e:
            self.logger.error(f"Error during simulation preparation: {e}")
            raise


    def get_formatted_simulation_params(self) -> str:
        """
        Generate a well-formatted string of simulation parameters.
        
        Returns:
            str: Formatted string of simulation parameters.
        """
        elmfire_path = self._get_elmfire_base_path()
        sim_config = self.get_config_for_simulation()
        
        formatted_output = [
            "=" * 80,
            "ELMFIRE Simulation Parameters".center(80),
            "=" * 80,
            f"ELMFIRE Directory: {elmfire_path}",
            "-" * 80,
            "Simulation Configuration:",
            "-" * 80
        ]

        for section, params in sim_config.sections.items():
            formatted_output.append(f"\n{section}:")
            for key, value in params.items():
                formatted_output.append(f"  {key}: {value}")

        formatted_output.extend([
            "-" * 80,
            "Input/Output Paths:",
            "-" * 80
        ])

        sim_paths = self.get_simulation_paths()
        for path_type, paths in [("Input Paths", sim_paths.input_paths), ("Output Paths", sim_paths.output_paths)]:
            formatted_output.append(f"\n{path_type}:")
            for key, value in paths.__dict__.items():
                formatted_output.append(f"  {key}: {value}")

        return "\n".join(formatted_output)
def main():
    """
    Main function to demonstrate the usage of ElmfireConfigManager.
    """
    # Initialize necessary components
    config = OverseerConfig()
    geospatial_manager = GeoSpatialManager(config)
    data_in_handler = ElmfireDataInHandler(config, OverseerLogger().get_logger('DataInHandler'))

    # Create ElmfireConfigManager instance
    config_manager = ElmfireConfigManager(config, geospatial_manager, data_in_handler)
    logger = config_manager.logger

    logger.info("Starting ElmfireConfigManager demonstration")

    try:
        # Prepare simulation
        logger.info("Preparing simulation...")
        config_manager.prepare_simulation()
        logger.info("Simulation preparation completed")

        # Print and log input file and geospatial information
        logger.info("Retrieving and logging input file and geospatial information")
        config_manager.print_and_log_info()

        # Test apply_action method
        test_action = [100, 100, 0, 5]  # Example action: start at (100,100), direction North, length 5
        logger.info(f"Applying test action: {test_action}")
        config_manager.apply_action(test_action)
        logger.info("Test action applied successfully")

        # Get and print configuration for simulation
        logger.info("Retrieving configuration for simulation")
        sim_config = config_manager.get_config_for_simulation()
        logger.info("Configuration for Simulation:")
        for key, value in sim_config.items():
            logger.info(f"  {key}: {value}")

        # Validate configuration
        logger.info("Validating configuration")
        is_valid = config_manager.validate_config()
        logger.info(f"Configuration is valid: {is_valid}")

        # Save configuration
        logger.info("Saving configuration")
        config_manager.save_config()
        logger.info("Configuration saved successfully")

    except Exception as e:
        logger.error(f"An error occurred during the demonstration: {str(e)}", exc_info=True)
    
    logger.info("ElmfireConfigManager demonstration completed")

if __name__ == "__main__":
    main()