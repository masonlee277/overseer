# 🔥 Overseer: Advanced ELMFIRE Simulation & RL Framework for Wildfire Suppression

## 📚 Overview

Overseer is a cutting-edge, open-source Python framework that enhances the ELMFIRE (Eulerian Level Set Model of FIRE spread) simulator with advanced reinforcement learning capabilities. Designed to revolutionize wildfire suppression strategies, Overseer serves as a comprehensive toolkit for:

1. Understanding and optimizing initial attack suppression plans
2. Developing safe and optimal wildfire control strategies
3. Providing an extensible platform for the wildfire research community

By integrating reinforcement learning with ELMFIRE simulations, Overseer aims to:

- Improve the effectiveness of wildfire suppression activities
- Operationalize wildfire models for real-world applications
- Foster innovation in RL-based approaches to wildfire management

Overseer provides a robust, modular foundation that allows researchers and practitioners to build upon and extend its capabilities, driving forward the state-of-the-art in wildfire suppression technology.

## 🏗 Architecture and Design Decisions

Overseer's architecture is built on a modular, component-based design that prioritizes flexibility, extensibility, and performance. Here's an overview of the major components and their interactions:

### 1. Configuration Management (`overseer/config/`)

- **OverseerConfig**: Centralized configuration handling using YAML files
- Manages both ELMFIRE-specific and system-wide settings
- Implements runtime configuration validation

### 2. Data Handling (`overseer/data/`)

- **DataManager**: Unified interface for all data operations
- **StateManager**: Efficient management of simulation state history
- Supports both file-based and database storage backends
- Implements data models for representing simulation states, actions, and results

### 3. ELMFIRE Integration (`overseer/elmfire/`)

- **SimulationManager**: Handles ELMFIRE execution and I/O management
- **ConfigManager**: Dynamically updates ELMFIRE input files (e.g., elmfire.data.in)
- Implements parallel simulation capabilities for performance optimization
- Runs simulations for 30-minute intervals per action, allowing for fine-grained control

### 4. Reinforcement Learning Environment (`overseer/rl/`)

- **ElmfireGymEnv**: OpenAI Gym-compatible environment for RL agent interaction
- Custom action space representing suppression activities (e.g., firebreak creation, resource deployment)
- Observation space encapsulating fire state, weather conditions, and terrain information
- Supports both online and offline RL algorithms, including DecisionDiffuser for offline learning

### 5. Geospatial Analysis (`overseer/analytics/`)

- **GeoSpatialManager**: Handles complex geospatial operations and analysis
- Utilizes GeoPandas and Rasterio for efficient spatial data processing
- Implements advanced analysis techniques (e.g., fire intensity calculation, risk area identification)

### 6. Visualization (`overseer/visualization/`)

- Real-time and post-simulation visualization capabilities
- Generates fire progression animations, risk assessment maps, and resource allocation reports

## 🔄 Workflow and Interaction with ELMFIRE

1. **Initialization**:

   - Load configuration settings from YAML files
   - Set up logging and data management systems
   - Initialize ELMFIRE simulation parameters

2. **Simulation Loop**:

   - Reset environment to initial state
   - For each 30-minute interval:
     - Get action from RL agent (e.g., firebreak locations, resource deployments)
     - Update ELMFIRE input files via ConfigManager
     - Run ELMFIRE simulation for 30 minutes using SimulationManager
     - Process simulation results with DataManager and GeoSpatialManager
     - Calculate reward based on fire suppression effectiveness and resource utilization
     - Generate next observation for the RL agent
     - Update RL agent (if using online learning)

3. **Data Processing and Analysis**:

   - Store simulation results and agent actions in DataManager
   - Perform geospatial analysis on fire spread data using GeoSpatialManager
   - Generate performance metrics (e.g., area burned, suppression cost, structures saved)
   - Update visualizations for real-time monitoring

4. **Post-Simulation Analysis**:
   - Generate comprehensive reports on suppression strategy effectiveness
   - Produce risk assessment maps and resource allocation recommendations
   - Analyze RL agent performance and learning progress

By chaining together multiple 30-minute simulations and allowing the RL agent to make decisions at each interval, Overseer creates a dynamic and responsive environment for developing and testing wildfire suppression strategies. This approach allows for:

- Fine-grained control over suppression activities
- Realistic representation of evolving fire conditions
- Evaluation of long-term suppression strategies
- Adaptation to changing weather and fire behavior

The modular design of Overseer enables researchers and practitioners to easily extend its capabilities, implement new RL algorithms, or integrate additional data sources and analysis techniques. This flexibility makes Overseer a powerful tool for advancing the field of wildfire management and operationalizing RL-based approaches to real-world wildfire challenges.

## 🔧 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/masonlee277/overseer.git
   cd overseer
   ```

2. Create a conda/mamba environment:

   ```bash
   conda create -n overseer python=3.9
   conda activate overseer
   conda install --file requirements.txt
   pip install -e .
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
