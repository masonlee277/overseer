# src/overseer/elmfire/validator.py

import os
import subprocess
from pathlib import Path
from ..config.config import OverseerConfig
from ..utils.logging import OverseerLogger

class ElmfireValidator:
    """
    Validates the ELMFIRE setup.
    """

    def __init__(self, config: OverseerConfig, logger: OverseerLogger):
        self.config = config
        self.logger = logger

    def validate_elmfire_setup(self) -> None:
        """
        Validate ELMFIRE setup including directory, environment variables, and executables.
        """
        self.logger.info("Starting ELMFIRE setup validation")
        self._validate_elmfire_directory()
        self._validate_environment_variables()
        self._validate_elmfire_executables()
        self.logger.info("ELMFIRE setup validation completed successfully")

    def _validate_elmfire_directory(self) -> None:
        """
        Validate the ELMFIRE base directory.
        """
        elmfire_base_dir = self.config.get('elmfire.base_directory')
        self.logger.info(f"Validating ELMFIRE base directory: {elmfire_base_dir}")
        if not os.path.isdir(elmfire_base_dir):
            error_msg = f"ELMFIRE base directory not found: {elmfire_base_dir}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        self.logger.info(f"ELMFIRE base directory validated: {elmfire_base_dir}")

    def _validate_environment_variables(self) -> None:
        """
        Validate required ELMFIRE environment variables.
        """
        self.logger.info("Validating ELMFIRE environment variables")
        required_vars = [
            'ELMFIRE_SCRATCH_BASE',
            'ELMFIRE_BASE_DIR',
            'ELMFIRE_INSTALL_DIR',
            'CLOUDFIRE_SERVER'
        ]
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                self.logger.info(f"Environment variable {var} is set to: {value}")
            else:
                self.logger.error(f"Required environment variable {var} is not set")
        
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        if missing_vars:
            error_msg = f"Missing required environment variables: {', '.join(missing_vars)}"
            self.logger.error(error_msg)
            raise EnvironmentError(error_msg)
        self.logger.info("All required ELMFIRE environment variables are set")

    def _validate_elmfire_executables(self) -> None:
        """
        Validate the presence of required ELMFIRE executables.
        """
        self.logger.info("Validating ELMFIRE executables")
        install_dir = os.environ.get('ELMFIRE_INSTALL_DIR')
        if not install_dir:
            error_msg = "ELMFIRE_INSTALL_DIR is not set"
            self.logger.error(error_msg)
            raise EnvironmentError(error_msg)

        required_executables = ['elmfire', 'elmfire_post']
        missing_executables = []

        for executable in required_executables:
            executable_path = Path(install_dir) / executable
            if executable_path.is_file():
                self.logger.info(f"Found executable: {executable_path}")
                try:
                    result = subprocess.run([str(executable_path), '--version'], capture_output=True, text=True)
                    self.logger.info(f"{executable} version: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    self.logger.warning(f"Failed to get version for {executable}: {e}")
            else:
                missing_executables.append(executable)
                self.logger.error(f"Executable not found: {executable_path}")

        if missing_executables:
            error_msg = f"Missing ELMFIRE executables: {', '.join(missing_executables)}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info("All required ELMFIRE executables found and validated")