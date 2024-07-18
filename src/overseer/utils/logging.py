import os
from pathlib import Path
from typing import Dict, Optional, Any
import logging
from datetime import datetime
import shutil

class OverseerLogger:
    """
    A comprehensive logging class for the Overseer project.
    
    This class manages logging for multiple components, handles file paths,
    and formats logs properly. It uses the Singleton pattern to ensure
    a single logging instance across the application.
    """

    _instance = None

    def __new__(cls) -> 'OverseerLogger':
        if cls._instance is None:
            cls._instance = super(OverseerLogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir: Path = Path('logs')
        self.max_log_files: int = 5

    def initialize(self, log_dir: Optional[str] = None, max_log_files: int = 5) -> None:
        """
        Initialize the logging system.

        Args:
            log_dir (Optional[str]): Directory to store log files. Defaults to 'logs'.
            max_log_files (int): Maximum number of log files to keep per component. Defaults to 5.
        """
        self.log_dir = Path(log_dir) if log_dir else Path('logs')
        self.max_log_files = max_log_files
        self._create_log_directory()

    def _create_log_directory(self) -> None:
        """Create the log directory if it doesn't exist."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger for a specific component.

        Args:
            name (str): Name of the component requesting a logger.

        Returns:
            logging.Logger: Logger instance for the specified component.
        """
        if name not in self.loggers:
            self._setup_logger(name)
        return self.loggers[name]

    def _setup_logger(self, name: str) -> None:
        """
        Set up a new logger for a component.

        Args:
            name (str): Name of the component.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        self._rotate_logs(name)

        file_handler = logging.FileHandler(self.log_dir / f"{name}.log")
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        self.loggers[name] = logger

    def _rotate_logs(self, name: str) -> None:
        """
        Rotate log files for a component.

        Args:
            name (str): Name of the component.
        """
        log_file = self.log_dir / f"{name}.log"
        if log_file.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = self.log_dir / f"{name}_{timestamp}.log"
            shutil.move(str(log_file), str(new_name))

        self._cleanup_old_logs(name)

    def _cleanup_old_logs(self, name: str) -> None:
        """
        Remove old log files exceeding the maximum number allowed.

        Args:
            name (str): Name of the component.
        """
        log_files = sorted(self.log_dir.glob(f"{name}_*.log"), key=os.path.getmtime, reverse=True)
        for old_file in log_files[self.max_log_files - 1:]:
            old_file.unlink()

    def log(self, name: str, level: str, message: str, **kwargs: Any) -> None:
        """
        Log a message for a specific component.

        Args:
            name (str): Name of the component logging the message.
            level (str): Log level ('debug', 'info', 'warning', 'error', 'critical').
            message (str): The log message.
            **kwargs: Additional key-value pairs to be included in the log message.
        """
        logger = self.get_logger(name)
        log_method = getattr(logger, level.lower())

        if kwargs:
            message += " - " + " - ".join(f"{k}={v}" for k, v in kwargs.items())

        log_method(message)

    def start_new_run(self) -> None:
        """
        Start a new run by rotating all log files.
        """
        for name in self.loggers:
            self._rotate_logs(name)

    def get_recent_logs(self, name: str, lines: int = 100) -> str:
        """
        Retrieve the most recent log entries for a component.

        Args:
            name (str): Name of the component.
            lines (int): Number of recent log lines to retrieve. Defaults to 100.

        Returns:
            str: The most recent log entries.
        """
        log_file = self.log_dir / f"{name}.log"
        if not log_file.exists():
            return f"No log file found for {name}"

        with open(log_file, 'r') as file:
            return ''.join(file.readlines()[-lines:])

# Usage example:
# logger = OverseerLogger()
# logger.initialize(log_dir='my_logs', max_log_files=10)
# logger.log('SimulationManager', 'info', 'Starting simulation', sim_id=123)