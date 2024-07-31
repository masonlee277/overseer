from typing import Dict, Any, Union
import os
import yaml
from pathlib import Path

class OverseerConfig:
    """
    A comprehensive configuration manager for the Overseer project.

    This class handles loading, accessing, and validating configuration settings
    for all components of the Overseer system, including logging configuration.

    Attributes:
        config_path (Path): Path to the YAML configuration file.
        config (Dict[str, Any]): Loaded configuration dictionary.
    """

    _instance = None

    def __new__(cls, config_path: Union[str, Path] = None):
        if cls._instance is None:
            cls._instance = super(OverseerConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Union[str, Path] = None):
        if self._initialized:
            return
        self._initialized = True
        
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Set the default path relative to this file's location
            current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            self.config_path = current_dir / 'elmfire_config.yaml'
        
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration from the YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Successfully loaded configuration from {self.config_path}")
            return config
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise
        except IOError as e:
            print(f"Error reading configuration file: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self.config
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value

    def save(self):
        """Save the current configuration back to the YAML file."""
        with open(self.config_path, 'w') as file:
            yaml.dump(self.config, file)

    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get the logging configuration."""
        return self.get('logging_config', {})

    @property
    def log_dir(self) -> Path:
        """Get the log directory path."""
        return Path(self.logging_config.get('log_dir', 'logs'))

    @property
    def log_level(self) -> str:
        """Get the default log level."""
        return self.logging_config.get('level', 'INFO')

    @property
    def log_format(self) -> str:
        """Get the log format."""
        return self.logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to configuration items."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-like setting of configuration items."""
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return key in self.config