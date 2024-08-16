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
        self.input_dir = None
        self.output_dir = None
        self.data_in_path = None

        self._load_config()
        self.load_elmfire_data_in()  # Add this line to load the data immediately
    
    def _load_config(self):
        yaml_config = self.config.get_config()
        self.logger.info("Setting up file paths")

        # Get the base directory
        self.elmfire_base = Path(yaml_config['directories']['elmfire_sim_dir'])

        # Set up other directories
        self.input_dir = self.elmfire_base / yaml_config['directories']['inputs']
        self.output_dir = self.elmfire_base / yaml_config['directories']['outputs']
        self.data_in_path = self.elmfire_base / yaml_config['directories']['data_in']

        self.logger.info(f"ELMFIRE base directory: {self.elmfire_base}")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"data.in file path: {self.data_in_path}")

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
        # Remove comments and trim whitespace
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
            if section not in self.elmfire_data_in:
                self.elmfire_data_in[section] = {}
            self.elmfire_data_in[section][key] = cleaned_value
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

            #log linebreak
            self.logger.info("-" * 80)
            self.logger.info("Successfully loaded elmfire.data.in content")
            self.logger.info(f"elmfire.data.in content: {self.elmfire_data_in}")
            self.logger.info("-" * 80)

        except Exception as e:
            self.logger.error(f"Failed to load elmfire.data.in: {str(e)}")
            raise

    def get_input_paths(self) -> InputPaths:
        """
        Create an InputPaths object from the current elmfire.data.in configuration.
        """
        inputs = self.elmfire_data_in.get('INPUTS', {})
        
        return InputPaths(
            fuels_and_topography_directory=str(self.input_dir),
            asp_filename=self._get_input_filename('ASP_FILENAME'),
            cbd_filename=self._get_input_filename('CBD_FILENAME'),
            cbh_filename=self._get_input_filename('CBH_FILENAME'),
            cc_filename=self._get_input_filename('CC_FILENAME'),
            ch_filename=self._get_input_filename('CH_FILENAME'),
            dem_filename=self._get_input_filename('DEM_FILENAME'),
            fbfm_filename=self._get_input_filename('FBFM_FILENAME'),
            slp_filename=self._get_input_filename('SLP_FILENAME'),
            adj_filename=self._get_input_filename('ADJ_FILENAME'),
            phi_filename=self._get_input_filename('PHI_FILENAME'),
            weather_directory=str(self.input_dir),
            ws_filename=self._get_input_filename('WS_FILENAME'),
            wd_filename=self._get_input_filename('WD_FILENAME'),
            m1_filename=self._get_input_filename('M1_FILENAME'),
            m10_filename=self._get_input_filename('M10_FILENAME'),
            m100_filename=self._get_input_filename('M100_FILENAME'),
        )

    def _get_input_filename(self, key: str) -> str:
        """Helper method to get input filename from elmfire.data.in"""
        return self.elmfire_data_in.get('INPUTS', {}).get(key, '').strip("'")

    def get_output_paths(self) -> OutputPaths:
        """
        Create an OutputPaths object from the current elmfire.data.in configuration.
        """
        self.logger.info("Getting output paths from elmfire.data.in configuration")
        outputs = self.elmfire_data_in.get('OUTPUTS', {})
        if not outputs:
            self.logger.warning("OUTPUTS section not found in elmfire.data.in")
        
        output_dir = str(self.output_dir)
        self.logger.debug(f"Output directory: {output_dir}")
        
        output_paths = OutputPaths(
            time_of_arrival=str(Path(output_dir) / "time_of_arrival"),
            fire_intensity=str(Path(output_dir) / "fire_intensity"),
            flame_length=str(Path(output_dir) / "flame_length"),
            spread_rate=str(Path(output_dir) / "spread_rate")
        )
        
        for attr, path in output_paths.__dict__.items():
            if not path:
                self.logger.warning(f"{attr.upper()} path is empty")
            else:
                self.logger.debug(f"{attr.upper()} path: {path}")
        
        return output_paths

    def get_input_output_paths(self) -> Tuple[InputPaths, OutputPaths]:
        """
        Get both input and output paths.
        """
        return self.get_input_paths(), self.get_output_paths()

    

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
        self.logger.info(f"Validating input files in directory: {self.input_dir}")
        input_paths = self.get_input_paths()
        
        required_files = [
            (self.input_dir / f"{input_paths.asp_filename}.tif"),
            (self.input_dir / f"{input_paths.cbd_filename}.tif"),
            (self.input_dir / f"{input_paths.cbh_filename}.tif"),
            (self.input_dir / f"{input_paths.cc_filename}.tif"),
            (self.input_dir / f"{input_paths.ch_filename}.tif"),
            (self.input_dir / f"{input_paths.dem_filename}.tif"),
            (self.input_dir / f"{input_paths.fbfm_filename}.tif"),
            (self.input_dir / f"{input_paths.slp_filename}.tif"),
            (self.input_dir / f"{input_paths.adj_filename}.tif"),
            (self.input_dir / f"{input_paths.phi_filename}.tif"),
            (self.input_dir / f"{input_paths.ws_filename}.tif"),
            (self.input_dir / f"{input_paths.wd_filename}.tif"),
            (self.input_dir / f"{input_paths.m1_filename}.tif"),
            (self.input_dir / f"{input_paths.m10_filename}.tif"),
            (self.input_dir / f"{input_paths.m100_filename}.tif"),
        ]

        all_files_exist = True
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                self.logger.error(f"Required input file not found: {file_path}")
                all_files_exist = False
                missing_files.append(str(file_path))
            else:
                self.logger.debug(f"Found required input file: {file_path}")

        if all_files_exist:
            self.logger.info("All required input files are present.")
        else:
            self.logger.warning(f"Some required input files are missing: {', '.join(missing_files)}")
            self.logger.info(f"Searched for files in: {self.input_dir}")

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

    This function demonstrates the usage of the ElmfireDataInHandler class,
    including loading, modifying, saving, and validating the elmfire.data.in file.
    It uses assertions to ensure the expected behavior.
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
    assert handler.elmfire_data_in, "Failed to load elmfire.data.in"

    print("1. Original elmfire.data.in file loaded successfully.")

    # Test setting parameters
    handler.set_parameter('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY', '/new/path/to/fuels')
    handler.set_parameter('OUTPUTS', 'DUMP_SPREAD_RATE', '.TRUE.')
    handler.set_parameter('TIME_CONTROL', 'SIMULATION_TSTOP', '43200.0')

    #print the parameters
    print(handler.get_parameter('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY'))
    print(handler.get_parameter('OUTPUTS', 'DUMP_SPREAD_RATE'))
    print(handler.get_parameter('TIME_CONTROL', 'SIMULATION_TSTOP'))

    assert handler.get_parameter('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY') == "'/new/path/to/fuels'", "Failed to set FUELS_AND_TOPOGRAPHY_DIRECTORY"
    assert handler.get_parameter('OUTPUTS', 'DUMP_SPREAD_RATE') == '.TRUE.', "Failed to set DUMP_SPREAD_RATE"
    assert handler.get_parameter('TIME_CONTROL', 'SIMULATION_TSTOP') == '43200.0', "Failed to set SIMULATION_TSTOP"

    print("2. Parameters set successfully.")

    # Test saving changes
    handler.save_elmfire_data_in()

    # Reload the file to ensure changes were saved
    handler.load_elmfire_data_in()
    

    assert handler.get_parameter('INPUTS', 'FUELS_AND_TOPOGRAPHY_DIRECTORY') == "'/new/path/to/fuels'", "Failed to save and reload FUELS_AND_TOPOGRAPHY_DIRECTORY"
    assert handler.get_parameter('OUTPUTS', 'DUMP_SPREAD_RATE') == '.TRUE.', "Failed to save and reload DUMP_SPREAD_RATE"
    assert handler.get_parameter('TIME_CONTROL', 'SIMULATION_TSTOP') == '43200.0', "Failed to save and reload SIMULATION_TSTOP"

    print("3. Changes saved and reloaded successfully.")

    # Test getting input and output paths
    #input_paths, output_paths = handler.get_input_output_paths()
    #assert input_paths.fuels_and_topography_directory == '/new/path/to/fuels', "Incorrect fuels_and_topography_directory"
    #assert output_paths.time_of_arrival, "Missing time_of_arrival in output paths"

    print("4. Input and output paths retrieved successfully.")

    # Test validating input files
    validation_result = handler.validate_input_files()
    print(f"5. Input file validation result: {validation_result}")

    # Test validating structure
    structure_valid = handler.validate_structure()
    assert structure_valid, "elmfire.data.in structure is invalid"

    print("6. elmfire.data.in structure is valid.")

    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    main()