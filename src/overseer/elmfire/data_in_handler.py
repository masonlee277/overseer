"""
ElmfireDataInHandler

Responsibilities:
1. Load, parse, and manage the elmfire.data.in file content.
2. Provide methods to update specific parameters in the elmfire.data.in configuration.
3. Save changes back to the elmfire.data.in file.
4. Validate the structure and content of the elmfire.data.in file.

Does NOT:
1. Modify any files other than elmfire.data.in.
2. Perform any geospatial operations or calculations.
3. Interact directly with the ELMFIRE simulator.
4. Handle overall simulation flow or decision-making.

Interfaces with:
1. ElmfireConfigManager: Provides methods for the ConfigManager to update elmfire.data.in.
2. OverseerConfig: Retrieves configuration settings.
3. OverseerLogger: For logging operations and errors.

Design Considerations:
- Implements a dictionary-based representation of the elmfire.data.in file for easy manipulation.
- Uses type hints for better code readability and maintainability.
- Implements robust error handling and logging for all file operations.
- Designed to be thread-safe for potential future multi-threaded applications.

Performance Considerations:
- Minimizes file I/O by caching the elmfire.data.in content in memory.
- Uses efficient data structures for quick access and updates to parameters.

Future Enhancements:
- Could implement a more robust parsing mechanism for complex parameter types.
- May add version control or change tracking for elmfire.data.in modifications.
"""

# Replace relative imports with absolute imports
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import pprint
import re

from overseer.utils.logging import OverseerLogger
from overseer.config.config import OverseerConfig
from overseer.core.models import InputPaths, OutputPaths


class ElmfireDataInHandler:
    """
    Handles operations related to the elmfire.data.in file.

    This class is responsible for loading, parsing, modifying, and saving the
    elmfire.data.in file. It provides an interface for other components of the
    Overseer system to interact with the ELMFIRE configuration.

    Attributes:
        config (OverseerConfig): The configuration object for the Overseer system.
        logger (OverseerLogger): The logger object for logging operations and errors.
        elmfire_data_in (Dict[str, Dict[str, Any]]): A nested dictionary representing the content of elmfire.data.in.
        data_in_path (Path): The path to the elmfire.data.in file.
    """

    def __init__(self, config: OverseerConfig, logger: OverseerLogger):
        """
        Initialize the ElmfireDataInHandler.

        Args:
            config (OverseerConfig): The configuration object for the Overseer system.
            logger (OverseerLogger): The logger object for logging operations and errors.
        """
        self.config = config
        self.logger = logger
        self.elmfire_data_in: Dict[str, Dict[str, Any]] = {}
        self._load_config()
        self.load_elmfire_data_in()  # Add this line to load the data immediately
    
    def _load_config(self):
        yaml_config = self.config.get_config()
        self.logger.info("Loading configuration values:")
        self.logger.info(f"Full YAML config: {yaml_config}")

        self.use_relative_paths = yaml_config.get('use_relative_paths', False)
        
        if self.use_relative_paths:
            # Use paths relative to the Overseer project root
            overseer_root = Path(__file__).resolve().parent.parent.parent.parent
            self.base_path = overseer_root
        else:
            # Use paths relative to the ELMFIRE directory
            elmfire_relative_path = yaml_config.get('elmfire_relative_path', '../elmfire')
            self.base_path = Path(__file__).resolve().parent.parent.parent / elmfire_relative_path

        self.logger.info(f"Base path for file operations: {self.base_path}")

        # Set data_in_path
        data_in_relative_path = yaml_config.get('io', {}).get('data_in_path', 'elmfire.data.in')
        self.data_in_path = self.base_path / data_in_relative_path
        self.logger.info(f"Set data_in_path to: {self.data_in_path}")


    def update_from_yaml(self, yaml_config: Dict[str, Any]) -> None:
        """
        Update the elmfire.data.in content based on a YAML configuration.

        This method updates the elmfire_data_in attribute with values from a YAML
        configuration, typically provided by the OverseerConfig.

        Args:
            yaml_config (Dict[str, Any]): A dictionary containing configuration values from a YAML file.
        """
        self.logger.info("Updating elmfire.data.in content from YAML configuration")
        
        if 'directories' in yaml_config:
            dirs = yaml_config['directories']
            self.set_directory('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY', dirs.get('fuels_and_topography'))
            self.set_directory('INPUTS', 'WEATHER_DIRECTORY', dirs.get('weather'))
            self.set_directory('OUTPUTS', 'OUTPUTS_DIRECTORY', dirs.get('outputs'))
            self.set_directory('MISCELLANEOUS', 'SCRATCH', dirs.get('scratch'))

        if 'parameters' in yaml_config:
            params = yaml_config['parameters']
            self.set_parameter('TIME_CONTROL', 'SIMULATION_DT', params.get('simulation_dt'))
            self.set_parameter('TIME_CONTROL', 'SIMULATION_TSTOP', params.get('simulation_tstop'))
            # Add more parameters as needed

        self.logger.info("Updated elmfire.data.in content from YAML configuration")



    def clean_value(self, value: str) -> str:
        """
        Clean and format the value string.

        This method removes comments, trims whitespace, and ensures proper formatting.

        Args:
            value (str): The value to clean.

        Returns:
            str: The cleaned value.
        """
        # Remove comments
        value = re.sub(r'!.*$', '', value).strip()

        # Handle boolean values
        if value.upper() in ['.TRUE.', '.FALSE.']:
            return value.upper()

        # Handle numeric values
        try:
            float(value)
            return value
        except ValueError:
            pass

        # Handle string values (add single quotes if not present)
        if not (value.startswith("'") and value.endswith("'")):
            value = f"'{value}'"

        return value

    def set_parameter(self, section: str, key: str, value: Any) -> None:
        """
        Set a parameter in the elmfire.data.in configuration.

        Args:
            section (str): The section name in elmfire.data.in.
            key (str): The parameter name.
            value (Any): The value to set for the parameter.
        """
        if value is not None:
            cleaned_value = self.clean_value(str(value))
            self.elmfire_data_in.setdefault(section, {})[key] = cleaned_value
            self.logger.info(f"Set parameter in elmfire.data.in: [{section}] {key} = {cleaned_value}")
        else:
            self.logger.warning(f"Attempted to set parameter with None value: [{section}] {key}")

    def load_elmfire_data_in(self) -> None:
        """
        Load and parse the elmfire.data.in file.

        This method reads the elmfire.data.in file, parses its content, and stores
        it in the elmfire_data_in attribute as a nested dictionary.

        Raises:
            Exception: If there's an error reading or parsing the file.
        """
        self.logger.info(f"Loading elmfire.data.in from: {self.data_in_path}")
        try:
            with open(self.data_in_path, 'r') as file:
                content = file.read()

            sections = content.split('&')[1:]  # Split by '&' and remove the first empty element
            for section in sections:
                lines = section.strip().split('\n')
                section_name = lines[0].strip().upper()
                self.elmfire_data_in[section_name] = {}
                for line in lines[1:]:
                    if '=' in line and line.strip() != '/':
                        key, value = line.split('=', 1)
                        cleaned_value = self.clean_value(value)
                        self.elmfire_data_in[section_name][key.strip()] = cleaned_value

            self.logger.info("Successfully loaded elmfire.data.in content")
        except Exception as e:
            self.logger.error(f"Failed to load elmfire.data.in: {str(e)}")
            raise


    def get_input_paths(self) -> InputPaths:
        """
        Create an InputPaths object from the current elmfire.data.in configuration.
        """
        inputs = self.elmfire_data_in.get('INPUTS', {})
        return InputPaths(
            fuels_and_topography_directory=inputs.get('FUELS_AND_TOPOGRAPHY_DIRECTORY', '').strip("'"),
            asp_filename=inputs.get('ASP_FILENAME', '').strip("'"),
            cbd_filename=inputs.get('CBD_FILENAME', '').strip("'"),
            cbh_filename=inputs.get('CBH_FILENAME', '').strip("'"),
            cc_filename=inputs.get('CC_FILENAME', '').strip("'"),
            ch_filename=inputs.get('CH_FILENAME', '').strip("'"),
            dem_filename=inputs.get('DEM_FILENAME', '').strip("'"),
            fbfm_filename=inputs.get('FBFM_FILENAME', '').strip("'"),
            slp_filename=inputs.get('SLP_FILENAME', '').strip("'"),
            adj_filename=inputs.get('ADJ_FILENAME', '').strip("'"),
            phi_filename=inputs.get('PHI_FILENAME', '').strip("'"),
            weather_directory=inputs.get('WEATHER_DIRECTORY', '').strip("'"),
            ws_filename=inputs.get('WS_FILENAME', '').strip("'"),
            wd_filename=inputs.get('WD_FILENAME', '').strip("'"),
            m1_filename=inputs.get('M1_FILENAME', '').strip("'"),
            m10_filename=inputs.get('M10_FILENAME', '').strip("'"),
            m100_filename=inputs.get('M100_FILENAME', '').strip("'"),
            fire='',  # You'll need to set these based on your specific requirements
            vegetation='',
            elevation='',
            wind='',
            fuel_moisture=''
        )

    def get_input_output_paths(self) -> Tuple[InputPaths, OutputPaths]:
        inputs = self.elmfire_data_in.get('INPUTS', {})
        outputs = self.elmfire_data_in.get('OUTPUTS', {})

        input_paths = InputPaths(
            fuels_and_topography_directory=inputs.get('FUELS_AND_TOPOGRAPHY_DIRECTORY', '').strip("'"),
            asp_filename=inputs.get('ASP_FILENAME', '').strip("'"),
            cbd_filename=inputs.get('CBD_FILENAME', '').strip("'"),
            cbh_filename=inputs.get('CBH_FILENAME', '').strip("'"),
            cc_filename=inputs.get('CC_FILENAME', '').strip("'"),
            ch_filename=inputs.get('CH_FILENAME', '').strip("'"),
            dem_filename=inputs.get('DEM_FILENAME', '').strip("'"),
            fbfm_filename=inputs.get('FBFM_FILENAME', '').strip("'"),
            slp_filename=inputs.get('SLP_FILENAME', '').strip("'"),
            adj_filename=inputs.get('ADJ_FILENAME', '').strip("'"),
            phi_filename=inputs.get('PHI_FILENAME', '').strip("'"),
            weather_directory=inputs.get('WEATHER_DIRECTORY', '').strip("'"),
            ws_filename=inputs.get('WS_FILENAME', '').strip("'"),
            wd_filename=inputs.get('WD_FILENAME', '').strip("'"),
            m1_filename=inputs.get('M1_FILENAME', '').strip("'"),
            m10_filename=inputs.get('M10_FILENAME', '').strip("'"),
            m100_filename=inputs.get('M100_FILENAME', '').strip("'"),
            fire=inputs.get('FIRE_FILENAME', '').strip("'"),
            vegetation=inputs.get('VEGETATION_FILENAME', '').strip("'"),
            elevation=inputs.get('ELEVATION_FILENAME', '').strip("'"),
            wind=inputs.get('WIND_FILENAME', '').strip("'"),
            fuel_moisture=inputs.get('FUEL_MOISTURE_FILENAME', '').strip("'")
        )

        output_paths = self.get_output_paths()

        return input_paths, output_paths
    
    def get_output_paths(self) -> OutputPaths:
        """
        Create an OutputPaths object from the current elmfire.data.in configuration.
        """
        self.logger.info("Getting output paths from elmfire.data.in configuration")
        outputs = self.elmfire_data_in.get('OUTPUTS', {})
        if not outputs:
            self.logger.warning("OUTPUTS section not found in elmfire.data.in")
        
        output_dir = outputs.get('OUTPUTS_DIRECTORY', '').strip("'")
        if not output_dir:
            self.logger.warning("OUTPUTS_DIRECTORY not specified in elmfire.data.in")
        
        self.logger.debug(f"Output directory: {output_dir}")
        
        output_paths = OutputPaths(
            time_of_arrival=f"{output_dir}/time_of_arrival",
            fire_intensity=f"{output_dir}/fire_intensity",
            flame_length=f"{output_dir}/flame_length",
            spread_rate=f"{output_dir}/spread_rate"
        )
        
        for attr, path in output_paths.__dict__.items():
            if not path:
                self.logger.warning(f"{attr.upper()} path is empty")
            else:
                self.logger.debug(f"{attr.upper()} path: {path}")
        
        return output_paths

    

    def validate_structure(self) -> bool:
        """
        Validate the structure of the elmfire.data.in file.

        This method checks if all required sections are present in the elmfire.data.in file
        and if the values are properly formatted.

        Returns:
            bool: True if the structure is valid, False otherwise.
        """
        required_sections = ['INPUTS', 'OUTPUTS', 'COMPUTATIONAL_DOMAIN', 'TIME_CONTROL', 'SIMULATOR']
        optional_sections = ['SPOTTING', 'MISCELLANEOUS']
        
        for section in required_sections:
            if section not in self.elmfire_data_in:
                self.logger.error(f"Missing required section in elmfire.data.in: {section}")
                return False
            else:
                self.logger.debug(f"Found required section: {section}")

        for section in optional_sections:
            if section not in self.elmfire_data_in:
                self.logger.warning(f"Optional section '{section}' not found in elmfire.data.in")
            else:
                self.logger.debug(f"Found optional section: {section}")

        for section, params in self.elmfire_data_in.items():
            self.logger.debug(f"Validating section: {section}")
            for key, value in params.items():
                if not self.validate_value(value):
                    self.logger.error(f"Invalid value in elmfire.data.in: [{section}] {key} = {value}")
                    return False
                else:
                    self.logger.debug(f"Valid value: [{section}] {key} = {value}")

        self.logger.info("elmfire.data.in structure is valid")
        return True

    def validate_value(self, value: str) -> bool:
        """
        Validate a single value from the elmfire.data.in file.

        Args:
            value (str): The value to validate.

        Returns:
            bool: True if the value is valid, False otherwise.
        """
        # Check for boolean values
        if value.strip() in ['.TRUE.', '.FALSE.']:
            return True

        # Check for numeric values
        try:
            float(value)
            return True
        except ValueError:
            pass

        # Check for string values (must be enclosed in single quotes)
        if value.strip().startswith("'") and value.strip().endswith("'"):
            return True

        # Check for array values
        if value.strip().startswith("(") and value.strip().endswith(")"):
            return True

        return False
    
    def validate_input_files(self) -> bool:
        """
        Validate if all required input files specified in elmfire.data.in exist in the input directory.

        Returns:
            bool: True if all required files exist, False otherwise.
        """
        inputs = self.elmfire_data_in.get('INPUTS', {})
        fuels_dir = Path(self.base_path) / inputs.get('FUELS_AND_TOPOGRAPHY_DIRECTORY', '').strip("'")
        weather_dir = Path(self.base_path) / inputs.get('WEATHER_DIRECTORY', '').strip("'")

        required_files = [
            (fuels_dir / inputs.get('ASP_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('CBD_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('CBH_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('CC_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('CH_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('DEM_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('FBFM40_FILENAME', '').strip("'")),
            (fuels_dir / inputs.get('SLP_FILENAME', '').strip("'")),
            (weather_dir / inputs.get('WS_FILENAME', '').strip("'")),
            (weather_dir / inputs.get('WD_FILENAME', '').strip("'")),
            (weather_dir / inputs.get('M1_FILENAME', '').strip("'")),
            (weather_dir / inputs.get('M10_FILENAME', '').strip("'")),
            (weather_dir / inputs.get('M100_FILENAME', '').strip("'")),
        ]

        all_files_exist = True
        for file_path in required_files:
            if not file_path.exists():
                self.logger.error(f"Required input file not found: {file_path}")
                all_files_exist = False
            else:
                self.logger.debug(f"Found required input file: {file_path}")

        if all_files_exist:
            self.logger.info("All required input files are present.")
        else:
            self.logger.warning("Some required input files are missing.")

        return all_files_exist

    
    def save_elmfire_data_in(self) -> None:
        """
        Save the current elmfire.data.in configuration back to file.

        This method writes the content of the elmfire_data_in attribute back to
        the elmfire.data.in file, ensuring proper formatting.

        Raises:
            Exception: If there's an error writing to the file.
        """
        self.logger.info(f"Saving elmfire.data.in to: {self.data_in_path}")
        try:
            with open(self.data_in_path, 'w') as file:
                for section, params in self.elmfire_data_in.items():
                    file.write(f"&{section}\n")
                    for key, value in params.items():
                        file.write(f"  {key} = {value}\n")
                    file.write("/\n\n")
            self.logger.info("elmfire.data.in saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save elmfire.data.in: {str(e)}")
            raise


    def set_directory(self, section: str, key: str, value: Optional[str]) -> None:
        """
        Set a directory parameter in the elmfire.data.in configuration.

        Args:
            section (str): The section name in elmfire.data.in.
            key (str): The parameter name.
            value (Optional[str]): The directory path to set.
        """
        if value is not None:
            self.elmfire_data_in.setdefault(section, {})[key] = f"'{value}'"
            self.logger.info(f"Set directory in elmfire.data.in: [{section}] {key} = {value}")
        else:
            self.logger.warning(f"Attempted to set directory with None value: [{section}] {key}")


    def get_parameter(self, section: str, key: str) -> Optional[Any]:
        """
        Get a parameter value from the elmfire.data.in configuration.

        Args:
            section (str): The section name in elmfire.data.in.
            key (str): The parameter name.

        Returns:
            Optional[Any]: The value of the parameter if found, None otherwise.
        """
        value = self.elmfire_data_in.get(section, {}).get(key)
        if value is not None:
            self.logger.info(f"Retrieved parameter from elmfire.data.in: [{section}] {key} = {value}")
        else:
            self.logger.warning(f"Parameter not found in elmfire.data.in: [{section}] {key}")
        return value



    def print_input_file(self) -> None:
        """
        Print the contents of the elmfire.data.in file.
        """
        self.logger.info("Printing contents of elmfire.data.in file:")
        try:
            with open(self.data_in_path, 'r') as file:
                content = file.read()
                print(content)
                self.logger.info(content)
        except Exception as e:
            self.logger.error(f"Failed to read elmfire.data.in: {str(e)}")
            print(f"Error reading file: {str(e)}")

    def print_internal_structure(self) -> None:
        """
        Print the internal data structure representing elmfire.data.in.
        """
        self.logger.info("Printing internal data structure:")
        pp = pprint.PrettyPrinter(indent=2)
        pp.pprint(self.elmfire_data_in)
        self.logger.info(pprint.pformat(self.elmfire_data_in))


    def generate_default_in_file(self) -> None:
        """
        Generate a default elmfire.data.in file with basic structure and values.
        """
        self.elmfire_data_in = {
            'INPUTS': {
                'FUELS_AND_TOPOGRAPHY_DIRECTORY': "'./inputs'",
                'ASP_FILENAME': "'asp'",
                'CBD_FILENAME': "'cbd'",
                'CBH_FILENAME': "'cbh'",
                'CC_FILENAME': "'cc'",
                'CH_FILENAME': "'ch'",
                'DEM_FILENAME': "'dem'",
                'FBFM_FILENAME': "'fbfm40'",
                'SLP_FILENAME': "'slp'",
                'ADJ_FILENAME': "'adj'",
                'PHI_FILENAME': "'new_phi'",
                'DT_METEOROLOGY': '3600.0',
                'WEATHER_DIRECTORY': "'./inputs'",
                'WS_FILENAME': "'ws'",
                'WD_FILENAME': "'wd'",
                'M1_FILENAME': "'m1'",
                'M10_FILENAME': "'m10'",
                'M100_FILENAME': "'m100'",
                'LH_MOISTURE_CONTENT': '30.0',
                'LW_MOISTURE_CONTENT': '60.0'
            },
            'OUTPUTS': {
                'OUTPUTS_DIRECTORY': "'./outputs'",
                'DTDUMP': '3600.',
                'DUMP_FLIN': '.TRUE.',
                'DUMP_SPREAD_RATE': '.TRUE.',
                'DUMP_TIME_OF_ARRIVAL': '.TRUE.',
                'CONVERT_TO_GEOTIFF': '.TRUE.',
                'DUMP_SPOTTING_OUTPUTS': '.TRUE.'
            },
            'COMPUTATIONAL_DOMAIN': {
                'A_SRS': "'EPSG: 32610'",
                'COMPUTATIONAL_DOMAIN_CELLSIZE': '60',
                'COMPUTATIONAL_DOMAIN_XLLCORNER': '-6000.00',
                'COMPUTATIONAL_DOMAIN_YLLCORNER': '-6000.00'
            },
            'TIME_CONTROL': {
                'SIMULATION_DT': '30.0',
                'SIMULATION_TSTOP': '3600.0'  # Changed to 1 hour (3600 seconds)
            },
            'SIMULATOR': {
                'NUM_IGNITIONS': '2'
            },
            'SPOTTING': {
                'CRITICAL_SPOTTING_FIRELINE_INTENSITY(:)': '0.',
                'EMBER_GR': '1.0',
                'ENABLE_SPOTTING': '.TRUE.',
                'PIGN': '100.0',
                'PIGN_MAX': '100.0',
                'PIGN_MIN': '100.0',
                'TAU_EMBERGEN': '10.0',
                'USE_UMD_SPOTTING_MODEL': '.TRUE.',
                'P_EPS': '0.001',
                'USE_PHYSICAL_SPOTTING_DURATION': '.FALSE.',
                'USE_PHYSICAL_EMBER_NUMBER': '.FALSE.',
                'EMBER_SAMPLING_FACTOR': '50.0',
                'GLOBAL_SURFACE_FIRE_SPOTTING_PERCENT_MAX': '100.0',
                'GLOBAL_SURFACE_FIRE_SPOTTING_PERCENT_MIN': '100.0',
                'USE_SUPERSEDED_SPOTTING': '.FALSE.'
            },
            'MISCELLANEOUS': {
                'PATH_TO_GDAL': "'/usr/bin'",
                'SCRATCH': "'./scratch'"
            }
        }
        self.save_elmfire_data_in()
        self.logger.info("Generated default elmfire.data.in file")

def main():
    """
    Main function for testing the ElmfireDataInHandler.

    This function demonstrates the basic usage of the ElmfireDataInHandler class,
    including loading, modifying, saving, and reloading the elmfire.data.in file.
    """
    from overseer.config.config import OverseerConfig
    from overseer.utils.logging import OverseerLogger

    # Initialize config and logger
    config = OverseerConfig(r'src\overseer\config\elmfire_config.yaml')
    logger = OverseerLogger().get_logger('ElmfireDataInHandlerTest')

    # Create an instance of ElmfireDataInHandler
    handler = ElmfireDataInHandler(config, logger)

    # Test loading elmfire.data.in
    handler.load_elmfire_data_in()

    print("1. Original elmfire.data.in file contents:")
    handler.print_input_file()

    print("\n2. Internal data structure after loading:")
    handler.print_internal_structure()

    # Make some changes
    handler.set_parameter('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY', '/new/path/to/fuels')
    handler.set_parameter('OUTPUTS', 'DUMP_SPREAD_RATE', '.TRUE.')
    handler.set_parameter('TIME_CONTROL', 'SIMULATION_TSTOP', '43200.0')

    print("\n3. Internal data structure after first set of changes:")
    handler.print_internal_structure()

    # Save changes
    handler.save_elmfire_data_in()

    # Reload the file to ensure changes were saved
    handler.load_elmfire_data_in()

    print("\n4. Reloaded data structure to verify changes:")
    handler.print_internal_structure()

    # Make more changes
    handler.set_parameter('COMPUTATIONAL_DOMAIN', 'COMPUTATIONAL_DOMAIN_CELLSIZE', '60')
    handler.set_parameter('SIMULATOR', 'NUM_IGNITIONS', '2')
    
    # Update from YAML config
    yaml_config = config.get_config()
    handler.update_from_yaml(yaml_config)

    print("\n5. Internal data structure after second set of changes and YAML update:")
    handler.print_internal_structure()

    # Save changes again
    handler.save_elmfire_data_in()

    # Final reload and print
    handler.load_elmfire_data_in()
    
    print("\n6. Final elmfire.data.in file contents:")
    handler.print_input_file()

    print("\n7. Final internal data structure:")
    handler.print_internal_structure()

    # Validate structure
    print(f"\n8. elmfire.data.in structure is valid: {handler.validate_structure()}")

if __name__ == "__main__":
    main()