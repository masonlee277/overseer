class ParallelEnvLauncher:
    """
    ParallelEnvLauncher: A class for managing multiple ELMFIRE simulation environments in parallel.

    This class is responsible for launching and coordinating multiple instances of the ELMFIRE
    simulation environment, each running in its own process. It allows for parallel training
    of multiple agents in separate ELMFIRE environments.

    Project Structure:
    overseer/
    ├── src/
    │   ├── overseer/
    │   │   ├── rl/
    │   │   │   ├── envs/
    │   │   │   │   └── elmfire_gym_env.py  # Contains the ElmfireGymEnv class
    │   │   │   └── ...
    │   │   ├── elmfire/
    │   │   │   └── ...  # ELMFIRE-related modules
    │   │   └── ...
    │   ├── parallel_computing/
    │   │   ├── __init__.py
    │   │   ├── parallel_env_launcher.py  # This file
    │   │   ├── training_coordinator.py
    │   │   └── result_aggregator.py
    │   └── ...
    ├── data/
    │   ├── env_1/
    │   │   ├── input/
    │   │   └── output/
    │   ├── env_2/
    │   │   ├── input/
    │   │   └── output/
    │   └── ...
    ├── configs/
    │   ├── base_config.yaml
    │   ├── env_1_config.yaml
    │   ├── env_2_config.yaml
    │   └── ...
    └── ...

    The ParallelEnvLauncher works as follows:
    1. It initializes multiple ElmfireGymEnv instances, each in its own process.
    2. Each environment instance has its own configuration file and data directory.
    3. The launcher provides methods to interact with all environments simultaneously.
    4. It uses multiprocessing to achieve true parallelism, utilizing multiple CPU cores.

    Data Structure:
    - Each environment has its own subdirectory in the 'data/' folder.
    - Input data and configuration files are stored in 'data/env_X/input/'.
    - Simulation outputs are saved in 'data/env_X/output/'.
    - Configuration files in 'configs/' define parameters for each environment.

    Usage:
    - Create an instance of ParallelEnvLauncher with the desired number of environments.
    - Use the launcher to reset environments, take steps, and retrieve observations.
    - The TrainingCoordinator class (in training_coordinator.py) uses this launcher
      to manage the training process across all environments.
    - The ResultAggregator class (in result_aggregator.py) collects and analyzes
      results from all environments after training.

    This setup allows for efficient parallel training of multiple agents in separate
    ELMFIRE environments, maximizing CPU utilization and speeding up the overall
    training process.
    """

    def __init__(self, num_envs: int):
        """
        Initialize the ParallelEnvLauncher with a specified number of environments.

        Args:
            num_envs (int): The number of parallel environments to create.
        """
        # Implementation details...

    def launch_envs(self):
        """
        Launch all environments in separate processes.
        """
        # Implementation details...

    def step(self, actions):
        """
        Take a step in all environments simultaneously.

        Args:
            actions (list): A list of actions, one for each environment.

        Returns:
            list: A list of (observation, reward, done, info) tuples, one for each environment.
        """
        # Implementation details...

    def reset(self):
        """
        Reset all environments.

        Returns:
            list: A list of initial observations, one for each environment.
        """
        # Implementation details...

    def close(self):
        """
        Close all environments and terminate their processes.
        """
        # Implementation details...