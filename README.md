# 🔥 Overseer: Advanced ELMFIRE Simulation & RL Framework

![Overseer Logo](path/to/logo.png)

## 📚 Overview

Overseer is a cutting-edge Python framework that enhances the ELMFIRE (Eulerian Level Set Model of FIRE spread) simulator with advanced reinforcement learning capabilities. Designed to revolutionize wildfire suppression strategies, Overseer combines intelligent simulation, deep analysis, and powerful visualizations.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://overseer-docs.readthedocs.io/)

## 🌟 Key Features

- 🧠 Reinforcement Learning Integration
- 🌍 Advanced Geospatial Analysis
- 🔄 Dynamic ELMFIRE Simulation Management
- 📊 Real-time Data Visualization
- 🛠 Flexible Configuration System

## 🏗 Architecture

Overseer employs a modular, component-based architecture, leveraging key design patterns for maximum flexibility and extensibility.

### 🧩 Core Components

1. **Configuration Management** 🔧

   - Utilizes the Singleton pattern for global access to configuration settings
   - Manages ELMFIRE-specific configurations and system-wide settings

2. **Data Handling** 📊

   - Implements the Facade pattern for simplified data operations
   - Manages simulation states, geospatial data, and analysis results

3. **ELMFIRE Integration** 🔥

   - Uses the Strategy pattern for flexible simulation execution
   - Interfaces with ELMFIRE for running fire spread simulations

4. **Reinforcement Learning Environment** 🤖

   - Implements the Observer pattern for RL agent interactions
   - Provides a Gym-compatible environment for training and evaluation

5. **Geospatial Analysis** 🗺️
   - Employs the Decorator pattern for extensible geospatial operations
   - Handles complex geospatial data processing and visualization

## 🔄 Workflow and Logic Flow

1. **Initialization** 🚀

   - Load configuration settings
   - Set up logging and data management systems
   - Initialize ELMFIRE simulation parameters

2. **Simulation Loop** 🔁

   - Reset environment to initial state
   - For each step:
     - Get action from RL agent
     - Apply action to ELMFIRE simulation
     - Run ELMFIRE simulation step
     - Process simulation results
     - Calculate reward and next observation
     - Update RL agent

3. **Data Processing** 💾

   - Store simulation results and agent actions
   - Perform geospatial analysis on fire spread data
   - Generate performance metrics and visualizations

4. **Analysis and Visualization** 📈
   - Analyze agent performance and fire suppression effectiveness
   - Generate fire progression visualizations
   - Produce risk assessment maps and resource allocation reports

## 🛠️ Implementation Details

### Configuration Management

- Uses YAML for configuration files
- Implements `OverseerConfig` class for centralized config management
- Validates configuration settings on load

### Data Handling

- Utilizes `DataManager` for centralized data operations
- Implements efficient state history management with customizable limits
- Supports various data storage backends (file-based, database)

### ELMFIRE Integration

- `SimulationManager` class handles ELMFIRE execution
- Manages ELMFIRE input/output files
- Implements parallel simulation capabilities for performance

### Reinforcement Learning

- `ElmfireGymEnv` provides a Gym-compatible environment
- Implements custom action and observation spaces
- Supports various RL algorithms through modular design

### Geospatial Analysis

- `GeoSpatialManager` handles complex geospatial operations
- Utilizes libraries like GeoPandas and Rasterio for efficient processing
- Implements advanced analysis techniques (e.g., risk area identification)

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/masonlee277/overseer.git
   cd overseer
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the package and dependencies:
   ```bash
   pip install -e .
   ```

## 🚀 Quick Start

'''python
config = OverseerConfig('config/overseer_config.yaml')
data_manager = DataManager(config)
sim_manager = SimulationManager(config)
env = ElmfireGymEnv(config)
'''

This Quick Start guide demonstrates the core workflow of the Overseer framework:

1. **Configuration Management**:

   - `OverseerConfig` loads settings from a YAML file, providing a centralized configuration for all components.

2. **Logging**:

   - `OverseerLogger` sets up a consistent logging system across all components.

3. **Data Handling**:

   - `DataManager` manages simulation data, storing and retrieving episode information.
   - `ElmfireDataInHandler` handles the ELMFIRE input file (elmfire.data.in), allowing for dynamic updates to simulation parameters.

4. **Geospatial Analysis**:

   - `GeoSpatialManager` performs spatial calculations and visualizations, such as fire intensity calculation and risk area identification.

5. **Simulation Management**:

   - `SimulationManager` interfaces with the ELMFIRE simulator, managing the execution of fire spread simulations.

6. **Reinforcement Learning Environment**:

   - `ElmfireGymEnv` provides a Gym-compatible environment for RL agents to interact with the ELMFIRE simulation.

7. **Simulation Loop**:

   - The environment is reset, and steps are taken using actions (randomly sampled in this example, but would be provided by an RL agent in practice).
   - Observations, rewards, and other information are processed and stored at each step.

8. **Post-Simulation Analysis**:
   - After the simulation, the framework provides tools for analyzing the results, including visualization of fire progression and generation of risk maps.
   - Performance metrics are calculated to evaluate the effectiveness of the fire suppression strategies.

This Quick Start guide showcases the integration of various components in the Overseer framework, demonstrating how they work together to create a comprehensive system for ELMFIRE simulation with reinforcement learning capabilities.

## 🤖 Reinforcement Learning Environment

### 🌿 ELMFIRE Gym Environment

Overseer implements a custom OpenAI Gym-compatible environment (`ElmfireGymEnv`) that interfaces with the ELMFIRE simulator. This environment allows reinforcement learning agents to interact with wildfire simulations in a standardized way.

Key features of `ElmfireGymEnv`:

- **Observation Space**: Represents the current state of the wildfire, including fire intensity, spread patterns, and relevant environmental factors.
- **Action Space**: Defines possible firefighting actions, such as creating firebreaks or deploying resources.
- **Step Function**: Executes actions in the ELMFIRE simulator, advances the simulation, and returns the new state, reward, and other information.
- **Reset Function**: Initializes the environment to a new starting state for each episode.
- **Reward Function**: Calculates the reward based on fire suppression effectiveness and resource utilization.

### 📚 Offline Reinforcement Learning

Overseer supports offline reinforcement learning, allowing agents to learn from pre-collected datasets of wildfire simulations. This approach is crucial for several reasons:

1. **Safety**: Avoids potentially dangerous exploration in real-world wildfire scenarios.
2. **Data Efficiency**: Leverages existing historical data and expert knowledge.
3. **Scalability**: Enables learning from a wide range of scenarios without running new simulations.

### 🧠 DecisionDiffuser and CleanDiffuser Library

To implement offline RL, Overseer utilizes the DecisionDiffuser algorithm implemented with the CleanDiffuser library. This choice is justified by several factors:

1. **Handling Complex Action Spaces**: DecisionDiffuser can effectively model the complex, multi-dimensional action space required for wildfire suppression strategies.

2. **Conditional Generation**: The diffusion model allows for generating actions conditioned on the current wildfire state, capturing nuanced relationships between observations and optimal actions.

3. **Distribution Matching**: DecisionDiffuser can closely match the distribution of expert actions in the offline dataset, reducing the risk of extrapolation errors common in offline RL.

4. **Uncertainty Quantification**: Diffusion models provide a natural way to represent uncertainty in action selection, which is crucial in high-stakes wildfire management decisions.

5. **Sample Efficiency**: The CleanDiffuser library implements efficient training and sampling procedures, allowing for effective learning from limited offline data.

6. **Flexibility**: The modular nature of CleanDiffuser allows for easy experimentation with different neural network architectures and diffusion processes tailored to the ELMFIRE domain.

By integrating DecisionDiffuser and CleanDiffuser, Overseer provides a powerful framework for learning sophisticated wildfire suppression policies from offline data, combining the benefits of deep reinforcement learning with the safety and efficiency of offline learning approaches.
