from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import numpy as np
import traceback
from pprint import pformat

from overseer.config.config import OverseerConfig
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.data.data_manager import DataManager
from overseer.core.models import JobStatus, SimulationResult, SimulationState, SimulationConfig, SimulationPaths, SimulationMetrics, Action
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
        self.compute_manager = ComputeManager(config) ## local compute manager by defualt
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
    

    # def run_simulation(self, sim_config: SimulationConfig = None) -> SimulationState:
    #     """
    #     Run the ELMFIRE simulation using the provided configuration.
        
    #     Args:
    #         sim_config (SimulationConfig, optional): The simulation configuration to use.
    #             If None, uses self.sim_config. Defaults to None.
        
    #     Returns:
    #         SimulationState: The resulting state after running the simulation.
    #     """
    #     self.logger.info("Running ELMFIRE simulation")
    #     if sim_config is None:
    #         sim_config = self.sim_config
    #     # Here you would interface with the ComputeManager to actually run ELMFIRE
    #     # For now, we'll create a mock simulation state
    #     mock_state = self._create_mock_simulation_state(sim_config)
        

    #     self.data_manager.update_state(mock_state)
    #     return mock_state

    def run_simulation(self, action: Optional[Action] = None) -> Tuple[SimulationState, float]:
        """
        Run an ELMFIRE simulation, optionally applying an action before the run.

        This method orchestrates the entire simulation process:
        1. Applies the given action to the current state (if provided)
        2. Prepares the simulation configuration
        3. Submits the simulation job to the ComputeManager
        4. Waits for the simulation to complete
        5. Processes the simulation results
        6. Updates the simulation state in the DataManager
        7. Calculates the reward based on the new state
        8. Determines if the simulation/episode is complete

        Args:
            action (Optional[Action]): The action to apply before running the simulation.
                                       If None, the simulation runs with the current configuration.

        Returns:
            Tuple[SimulationState, float, bool]:
                - SimulationState: The new state after the simulation run
                - bool: Whether the simulation/episode is complete

        Raises:
            Exception: If there's an error during the simulation process
        """
        self.logger.info("Starting ELMFIRE simulation run")
        
        try:
            # Apply action if provided
            if action:
                self.logger.info(f"Applying action to current state: {action}")
                self.config_manager.apply_action(action)
            else:
                self.logger.info("No action provided, running simulation with current configuration")

            # Prepare simulation configuration and paths
            sim_config = self.config_manager.get_config_for_simulation()
            sim_paths = self.config_manager.get_simulation_paths()
            self.logger.debug(f"Prepared simulation configuration: {sim_config}")
            self.logger.debug(f"Prepared simulation paths: {sim_paths}")

            # Submit simulation to ComputeManager
            self.logger.info("Submitting simulation to ComputeManager")
            sim_result = self.compute_manager.submit_simulation()
            
            if sim_result.status != JobStatus.COMPLETED:
                self.logger.error(f"Simulation failed with status: {sim_result.status}")
                self.logger.error(f"Error message: {sim_result.error_message}")
                raise Exception(f"Simulation failed: {sim_result.error_message}")

            self.logger.info(f"Simulation completed successfully. Job ID: {sim_result.job_id}")
            self.logger.debug(f"Simulation results: {sim_result}")

            # Convert SimulationResult to SimulationState
            self.logger.info("Converting simulation result to SimulationState")
            new_state = self.data_manager.simresult_to_simstate(
                sim_result=sim_result,
                sim_config=sim_config,
                sim_paths=sim_paths,
                timestamp=datetime.now()
            )
            
            if new_state is None:
                self.logger.error("Failed to convert SimulationResult to SimulationState")
                raise Exception("Failed to create new SimulationState")

            self.logger.debug(f"New SimulationState created: {new_state}")

            # Update state in DataManager
            self.logger.info("Updating state in DataManager")
            self.data_manager.update_state(new_state)

            # Update PHI file for the next simulation
            self.update_phi_file(new_state)

            # Check if simulation is complete
            done = self.check_simulation_complete(new_state)
            self.logger.info(f"Simulation complete: {done}")

            return new_state, done

        except Exception as e:
            self.logger.error(f"Error during simulation run: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            raise


    def update_phi_file(self, state: SimulationState) -> None:
        """
        Update the PHI file based on the current simulation state.

        Args:
            state (SimulationState): The current simulation state.
        """
        self.logger.info("Updating PHI file for next simulation")
        try:
            # Construct the full PHI file path
            phi_filename = "phi.tif"
            phi_path = state.paths.input_paths.fuels_and_topography_directory / phi_filename
            toa_path = state.paths.output_paths.time_of_arrival
            flin_path = state.paths.output_paths.fire_intensity

            self.logger.debug(f"PHI file path: {phi_path}")
            self.logger.debug(f"TOA file path: {toa_path}")
            self.logger.debug(f"FLIN file path: {flin_path}")

            self.data_manager.geospatial_manager.update_phi_file(
                phi_path=str(phi_path),
                toa_path=str(toa_path),
                flin_path=str(flin_path)
            )
            self.logger.info("PHI file updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating PHI file: {str(e)}")
            self.logger.error("Full traceback:")
            self.logger.error(traceback.format_exc())
            raise


    def _create_simulation_state_from_result(self, result: SimulationResult, sim_config: SimulationConfig) -> SimulationState:
        # Implement this method to create a SimulationState from the SimulationResult
        # You'll need to parse the output files or use the result data to populate the SimulationState
        pass

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

    def print_simulation_params(self) -> None:
        """
        Print a well-formatted summary of the simulation parameters.
        """
        formatted_params = self.config_manager.get_formatted_simulation_params()
        print(formatted_params)
        self.logger.info("Printed simulation parameters summary")

    def get_simulation_params_summary(self) -> str:
        """
        Get a well-formatted summary of the simulation parameters as a string.

        Returns:
            str: Formatted string of simulation parameters.
        """
        return self.config_manager.get_formatted_simulation_params()
    
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
    

from overseer.config.config import OverseerConfig
from overseer.elmfire.config_manager import ElmfireConfigManager
from overseer.data.data_manager import DataManager
from overseer.compute.compute_manager import ComputeManager
from overseer.rl.spaces.action_space import ActionSpace
from overseer.core.models import Action

def main():
    # Initialize components
    config = OverseerConfig()
    data_manager = DataManager(config)
    config_manager = ElmfireConfigManager(config, data_manager)
    compute_manager = ComputeManager(config)

    # Create SimulationManager instance
    sim_manager = SimulationManager(config, config_manager, data_manager)

    # Create ActionSpace
    action_space = ActionSpace(config, data_manager)

    # Print simulation parameters
    print("Simulation Parameters Summary:")
    sim_manager.print_simulation_params()

    # Print initial simulation summary
    print("\nInitial Simulation Summary:")
    sim_manager.print_simulation_summary()

    # Perform 10 actions
    for i in range(10):
        print(f"\n--- Action {i+1} ---")
        
        # Generate a random action
        random_action = action_space.sample_action()
        print(f"Random Action Generated: {random_action}")

        # Run simulation with the random action
        try:
            new_state, done = sim_manager.run_simulation(random_action)
            print(f"Simulation Results:")
            print(f"New State: {new_state}")
            print(f"Done: {done}")

            # Print updated simulation summary
            print("\nUpdated Simulation Summary:")
            sim_manager.print_simulation_summary()

            if done:
                print("Simulation completed. Resetting...")
                sim_manager.reset_simulation()
        except Exception as e:
            print(f"Error during simulation: {str(e)}")
            break

    print("\nSimulation sequence completed.")

if __name__ == "__main__":
    main()