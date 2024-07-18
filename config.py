from typing import Dict, Any, List
import yaml
from pathlib import Path

class OverseerConfig:
    def __init__(self, config_path: Path):
        self.config_path: Path = config_path
        self.config: Dict[str, Any] = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    @property
    def data_directories(self) -> Dict[str, Path]:
        data_dirs = self.get('data_directories', {})
        return {k: Path(v) for k, v in data_dirs.items()}

    @property
    def elmfire_path(self) -> Path:
        return Path(self.get('elmfire_path'))

    @property
    def output_path(self) -> Path:
        return Path(self.get('output_path'))

    # Add more properties as needed for common configuration items