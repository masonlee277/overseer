import os

# Define the directory structure and files
structure = {
    "src/overseer/rl/": [
        "__init__.py"
    ],
    "src/overseer/rl/agents/": [
        "__init__.py",
        "base_agent.py",
        "ppo_agent.py",
        "dqn_agent.py"
    ],
    "src/overseer/rl/envs/": [
        "__init__.py",
        "elmfire_gym_env.py"
    ],
    "src/overseer/rl/rewards/": [
        "__init__.py",
        "reward_manager.py",
        "reward_strategies.py",
        "reward_utils.py"
    ],
    "src/overseer/rl/spaces/": [
        "__init__.py",
        "action_space.py",
        "observation_space.py"
    ],
    "src/overseer/rl/utils/": [
        "__init__.py",
        "state_encoder.py",
        "action_decoder.py"
    ]
}

# Function to create directory structure and files
def create_structure(base_dir, structure):
    for directory, files in structure.items():
        # Create directories
        dir_path = os.path.join(base_dir, directory)
        os.makedirs(dir_path, exist_ok=True)
        # Create files
        for file in files:
            file_path = os.path.join(dir_path, file)
            open(file_path, 'a').close()  # 'a' mode will create the file if it doesn't exist

# Current directory
current_dir = os.getcwd()

# Create the structure
create_structure(current_dir, structure)

print("Directory structure and files created successfully.")
