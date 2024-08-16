# src/overseer/elmfire/environment_manager.py

import os
from ..config.config import OverseerConfig
from ..utils.logging import OverseerLogger

class EnvironmentManager:
    """
    Manages environment variables for ELMFIRE.
    """

    def __init__(self, config: OverseerConfig, logger: OverseerLogger):
        self.config = config
        self.logger = logger

    def export_environment_variables(self) -> None:
        """
        Export required ELMFIRE environment variables based on the configuration.
        """
        self.logger.info("Exporting ELMFIRE environment variables")
        env_vars = {
            'ELMFIRE_SCRATCH_BASE': self.config.get('elmfire.scratch_base'),
            'ELMFIRE_BASE_DIR': self.config.get('elmfire.base_directory'),
            'ELMFIRE_INSTALL_DIR': self.config.get('elmfire.install_directory'),
            'CLOUDFIRE_SERVER': self.config.get('elmfire.cloudfire_server')
        }

        for var, value in env_vars.items():
            if value:
                os.environ[var] = value
                self.logger.info(f"Exported environment variable: {var}={value}")
            else:
                self.logger.warning(f"Environment variable {var} not set in configuration")

        if 'ELMFIRE_INSTALL_DIR' in os.environ:
            new_path = f"{os.environ['ELMFIRE_INSTALL_DIR']}:{os.environ['PATH']}"
            os.environ['PATH'] = new_path
            self.logger.info(f"Updated PATH: {new_path}")
        else:
            self.logger.warning("ELMFIRE_INSTALL_DIR not set, PATH not updated")