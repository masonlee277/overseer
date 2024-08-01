import os
import sys
from setuptools import setup, find_packages

def read_requirements():
    print("Reading requirements.txt...")
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    requirements = [req for req in requirements if req and not req.startswith('#')]
    print(f"Found {len(requirements)} requirements.")
    return requirements

def load_config():
    config_path = os.path.join('src', 'overseer', 'config', 'elmfire_config.yaml')
    print(f"Configuration file path: {config_path}")
    print("Note: YAML configuration will be loaded during runtime, not during setup.")
    return {}

def set_environment_variables(config):
    print("Setting environment variables...")
    env = config.get('environment', {})
    variables = {
        'ELMFIRE_SCRATCH_BASE': env.get('elmfire_scratch_base', ''),
        'ELMFIRE_BASE_DIR': env.get('elmfire_base_dir', ''),
        'ELMFIRE_INSTALL_DIR': env.get('elmfire_install_dir', ''),
        'CLOUDFIRE_SERVER': env.get('cloudfire_server', '')
    }
    
    for key, value in variables.items():
        os.environ[key] = value
        print(f"Set {key} = {value}")
    
    # Append to PATH
    elmfire_paths = [
        os.environ.get('ELMFIRE_INSTALL_DIR', ''),
        os.path.join(os.environ.get('ELMFIRE_BASE_DIR', ''), 'cloudfire')
    ]
    os.environ['PATH'] = os.pathsep.join([os.environ['PATH']] + elmfire_paths)
    print(f"Updated PATH with: {elmfire_paths}")

print("Starting Overseer setup process...")

config = load_config()
set_environment_variables(config)

requirements = read_requirements()

print("Initiating setup...")
setup(
    name="overseer",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.10',
    install_requires=requirements,
)
print("Setup completed.")