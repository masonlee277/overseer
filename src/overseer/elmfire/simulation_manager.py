from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import numpy as np

from overseer.config.config import OverseerConfig
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.data.data_manager import DataManager
from overseer.core.models import SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics, Action
from overseer.utils.logging import OverseerLogger
from overseer.compute.compute_manager import ComputeManager

class SimulationManager:
    """
    SimulationManager is responsible for orchestrating the ELMFIRE simulation process.
    
    It delegates specific responsibilities to ConfigManager and DataManager:

    ConfigManager responsibilities:
    - Managing the ELMFIRE configuration (elmfire.data.in)
    - Providing methods to update and retrieve configuration parameters
    - Handling the data_in_handler for ELMFIRE-specific input/output operations

    DataManager responsibilities:
    - Managing simulation states and episodes
    - Handling data persistence and retrieval
    - Managing geospatial data and file operations

    SimulationManager responsibilities:
    - Initializing and coordinating the simulation process
    - Managing the simulation lifecycle (setup, run, teardown)
    - Interfacing with the ComputeManager for execution
    - Providing high-level methods for running simulations and retrieving results
    - Coordinating between ConfigManager and DataManager as needed

    This class serves as the main entry point for running ELMFIRE simulations and
    retrieving simulation results in the context of the reinforcement learning environment.
    """

    def __init__(self, config: OverseerConfig, config_manager: ElmfireConfigManager, data_manager: DataManager):
        self.config = config
        self.config_manager = config_manager
        self.data_manager = data_manager
        self.logger = OverseerLogger().get_logger(self.__class__.__name__)
        self.compute_manager = ComputeManager(self.config.get('compute_config', {}))
        self.logger.info("SimulationManger Initialized with simulation configuration")
        self.sim_config = None
        self.sim_paths = None
        self.sim_config = self.setup_simulation()

    def setup_simulation(self) -> SimulationConfig:
        """
        Set up the simulation by preparing the configuration and input files.
        
        Returns:
            SimulationConfig: The prepared simulation configuration.
        """
        if self.sim_config is None:
            self.logger.error("Simulation configuration is None")
        self.logger.info("Setting up simulation")
        self.config_manager.prepare_simulation_inputs()
        sim_config = self.config_manager.get_config_for_simulation()
        self.logger.info(f"Simulation configuration: {sim_config}")

        self.sim_paths = self.config_manager.get_simulation_paths()
        self.logger.info(f"Simulation paths: {self.sim_paths}")
        return sim_config

    def apply_action(self, action: Action) -> SimulationState:
        """Apply an action to the simulation and run it."""
        self.logger.info(f"Applying action: {action}")
        self.config_manager.apply_action(action)
        self.logger.info(f"Action applied, running simulation")
        results = self.run_simulation()
        self.logger.info("=" * 50)
        self.logger.info(f"[apply action] Simulation results")

        self.logger.info(f"Simulation results: {results}")
        self.logger.info("=" * 50)

        #check if the simulation is complete
        done = self.check_simulation_complete(results)
        if results is None: #log warnning
            self.logger.warning("[apply_action] Simulation results are None")
        return results, done
    

    def run_simulation(self, sim_config: SimulationConfig = None) -> SimulationState:
        """
        Run the ELMFIRE simulation using the provided configuration.
        
        Args:
            sim_config (SimulationConfig, optional): The simulation configuration to use.
                If None, uses self.sim_config. Defaults to None.
        
        Returns:
            SimulationState: The resulting state after running the simulation.
        """
        self.logger.info("Running ELMFIRE simulation")
        if sim_config is None:
            sim_config = self.sim_config
        # Here you would interface with the ComputeManager to actually run ELMFIRE
        # For now, we'll create a mock simulation state
        mock_state = self._create_mock_simulation_state(sim_config)
        

        self.data_manager.update_state(mock_state)
        return mock_state
    
    def check_simulation_complete(self, results: SimulationState) -> bool:
        """
        Check if the simulation is complete.
        
        Args:
            results (SimulationState): The results of the simulation.
        
        Returns:
            bool: True if the simulation is complete, False otherwise.
        """
        #return results.metrics.containment_percentage >= 100
        return False ##TODO: implement this
    
    def get_state(self) -> SimulationState:
        """
        Get the current simulation state.
        
        Returns:
            SimulationState: The current simulation state.
        """
        return self.data_manager.get_current_state()

    def get_simulation_results(self) -> SimulationState:
        """
        Retrieve the results of the most recent simulation.
        
        Returns:
            SimulationState: The current simulation state.
        """
        return self.data_manager.get_current_state()

    def create_initial_state(self) -> SimulationState:
        """
        Create an initial SimulationState without running a full simulation.
        Used for the initial state in the gym environment
        """
        initial_config = self.config_manager.get_config_for_simulation()
        initial_paths = self.config_manager.get_simulation_paths()
        
        return SimulationState(
            timestamp=datetime.now(),
            config=initial_config,
            paths=initial_paths,
            metrics=SimulationMetrics(
                burned_area=0.0,
                fire_perimeter_length=0.0,
                containment_percentage=0.0,
                execution_time=0.0,
                performance_metrics={},
                fire_intensity=np.zeros((100, 100))  # Placeholder size
            ),
            save_path=None,
            resources=None,
            weather=None
        )

    def update_simulation_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the simulation configuration with new parameters.
        
        Args:
            updates (Dict[str, Any]): A dictionary of configuration updates.
        """
        self.config_manager.update_config(updates)

    def _create_mock_simulation_state(self, sim_config: SimulationConfig) -> SimulationState:
        """
        Create a mock simulation state for testing purposes.
        
        Args:
            sim_config (SimulationConfig): The simulation configuration.
        
        Returns:
            SimulationState: A mock simulation state.
        """
        if sim_config is None:
            self.logger.error("SimulationConfig is None in _create_mock_simulation_state")
            raise ValueError("SimulationConfig cannot be None")

        return SimulationState(
            timestamp=datetime.now(),
            config=sim_config,
            paths=self.sim_paths,  # Use the SimulationPaths from config_manager
            metrics=SimulationMetrics(
                burned_area=1000.0,
                fire_perimeter_length=500.0,
                containment_percentage=30.0,
                execution_time=120.0,
                performance_metrics={'cpu_usage': 80.0, 'memory_usage': 4000.0},
                fire_intensity=None  # You might want to generate a mock numpy array here
            ),
            save_path=Path('mock/save/path'),
            resources={'firefighters': 20, 'trucks': 5},
            weather={'wind_speed': 10.0, 'wind_direction': 180.0}
        )

    def cleanup(self) -> None:
        """
        Perform any necessary cleanup after simulations.
        """
        self.logger.info("Cleaning up after simulation")
        self.data_manager.reset()

    def validate_simulation_setup(self) -> bool:
        """
        Validate the simulation setup, including configuration and input files.
        
        Returns:
            bool: True if the setup is valid, False otherwise.
        """
        return self.config_manager.validate_config() and self.data_manager.validate_input_files()

    def get_simulation_metrics(self) -> Dict[str, Any]:
        """
        Retrieve metrics from the most recent simulation.
        
        Returns:
            Dict[str, Any]: A dictionary of simulation metrics.
        """
        current_state = self.data_manager.get_current_state()
        return current_state.metrics.__dict__ if current_state else {}

    def save_simulation_state(self) -> None:
        """
        Save the current simulation state to disk.
        """
        self.data_manager.save_current_state()

    def load_simulation_state(self, state_id: str) -> Optional[SimulationState]:
        """
        Load a specific simulation state from disk.
        
        Args:
            state_id (str): The identifier for the state to load.
        
        Returns:
            Optional[SimulationState]: The loaded simulation state, or None if not found.
        """
        return self.data_manager.load_state(state_id)

    def get_input_file_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about the input files used in the simulation.
        
        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing information about each input file.
        """
        return self.config_manager.get_input_file_info()

    def print_simulation_summary(self) -> None:
        """
        Print a summary of the current simulation setup and results.
        """
        self.logger.info("Simulation Summary:")
        self.logger.info(f"Configuration: {self.config_manager.get_config_for_simulation()}")
        self.logger.info(f"Current State: {self.data_manager.get_current_state()}")
        self.logger.info(f"Metrics: {self.get_simulation_metrics()}")

    def reset_simulation(self) -> SimulationState:
        """
        Reset the simulation to its initial state.
        
        Returns:
            SimulationState: The initial simulation state after reset.
        """
        self.config_manager.reset_config()
        #TODO: fix this
        initial_state = self._create_mock_simulation_state(self.config_manager.get_config_for_simulation())
        self.data_manager.reset()
        self.data_manager.update_state(initial_state)
        return initial_state