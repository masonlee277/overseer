# Overseer: ELMFIRE Simulation and Reinforcement Learning Framework

## Overview

Overseer is a Python framework designed to enhance and extend the capabilities of the ELMFIRE (Eulerian Level Set Model of FIRE spread) simulator. It integrates reinforcement learning techniques for optimizing wildfire suppression strategies and provides tools for running simulations, analyzing results, and visualizing outcomes.

## Project Structure

overseer/
├── src/
│ └── overseer/
│ ├── config/
│ │ ├── init.py
│ │ └── config.py
│ ├── data/
│ │ ├── init.py
│ │ └── data_manager.py
│ ├── elmfire/
│ │ ├── init.py
│ │ ├── simulation_manager.py
│ │ └── config_manager.py
│ ├── rl/
│ │ ├── agents/
│ │ ├── envs/
│ │ ├── rewards/
│ │ ├── spaces/
│ │ ├── utils/
│ │ └── init.py
│ ├── utils/
│ │ ├── init.py
│ │ └── logging.py
│ └── init.py
├── tests/
├── docs/
├── examples/
├── requirements.txt
├── setup.py
└── README.md

## Main Components

1. **Configuration (overseer.config)**

   - OverseerConfig: Manages global configuration for the entire system.

2. **Data Management (overseer.data)**

   - DataManager: Handles data operations, storage, and retrieval for simulations and RL training.

3. **ELMFIRE Integration (overseer.elmfire)**

   - SimulationManager: Manages ELMFIRE simulations, including running simulations and processing outputs.
   - ConfigurationManager: Handles ELMFIRE-specific configurations.

4. **Reinforcement Learning (overseer.rl)**

   - ElmfireGymEnv (in envs/): Implements a Gym-compatible environment for RL agents to interact with ELMFIRE simulations.
   - RewardManager (in rewards/): Calculates rewards for RL agents based on simulation outcomes.
   - ActionSpace and ObservationSpace (in spaces/): Define the action and observation spaces for the RL environment.
   - StateEncoder and ActionDecoder (in utils/): Handle conversion between ELMFIRE states/actions and RL-compatible formats.

5. **Utilities (overseer.utils)**
   - OverseerLogger: Provides logging capabilities across the entire system.

## Key Interactions

- The ElmfireGymEnv uses the SimulationManager to run ELMFIRE simulations based on RL agent actions.
- The DataManager is used by various components to save and load simulation states, RL metrics, and other data.
- The RewardManager calculates rewards based on simulation outcomes, which are then used by the ElmfireGymEnv.
- The ConfigurationManager is used by the SimulationManager to set up and modify ELMFIRE simulations.

## Usage

Here's a basic example of how to use Overseer:

```python
from overseer import OverseerConfig, ElmfireGymEnv, SimulationManager, DataManager

# Initialize configuration
config = OverseerConfig('path/to/config.yaml')

# Initialize managers
data_manager = DataManager(config)
sim_manager = SimulationManager(config)

# Create the RL environment
env = ElmfireGymEnv(config, sim_manager, data_manager)

# Use the environment with your RL algorithm
observation = env.reset()
for _ in range 1000:
    action = your_rl_agent.choose_action(observation)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
```
