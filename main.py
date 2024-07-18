import os
import sys

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

def create_file(path, content=""):
    with open(path, 'w') as f:
        f.write(content)
    print(f"Created file: {path}")

def setup_overseer_project(base_path):
    # Create main project structure
    create_directory(os.path.join(base_path, "src", "overseer"))
    create_directory(os.path.join(base_path, "tests"))
    create_directory(os.path.join(base_path, "docs"))
    create_directory(os.path.join(base_path, "examples"))

    # Create module directories
    modules = ["elmfire", "rl_env", "foundation", "ensemble", "analytics", "visualization"]
    for module in modules:
        module_path = os.path.join(base_path, "src", "overseer", module)
        create_directory(module_path)
        create_directory(os.path.join(module_path, "utils"))
        create_file(os.path.join(module_path, "__init__.py"))
        create_file(os.path.join(module_path, "utils", "__init__.py"))

    # Create core files
    create_file(os.path.join(base_path, "src", "overseer", "__init__.py"))
    create_file(os.path.join(base_path, "src", "overseer", "core.py"))

    # Create test directories
    create_directory(os.path.join(base_path, "tests", "unit"))
    create_directory(os.path.join(base_path, "tests", "integration"))

    # Create documentation files
    doc_files = ["conf.py", "index.rst", "installation.rst", "usage.rst", "api_reference.rst", "examples.rst"]
    for doc_file in doc_files:
        create_file(os.path.join(base_path, "docs", doc_file))

    # Create example files
    example_files = ["basic_simulation.py", "rl_agent_training.py", "ensemble_analysis.py", "custom_visualization.py"]
    for example_file in example_files:
        create_file(os.path.join(base_path, "examples", example_file))

    # Create root level files
    create_file(os.path.join(base_path, "setup.py"))
    create_file(os.path.join(base_path, "README.md"), "# Overseer Project\n\nAdd project description here.")
    create_file(os.path.join(base_path, "requirements.txt"))
    create_file(os.path.join(base_path, ".gitignore"))

    print("Overseer project structure created successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = os.getcwd()
    
    setup_overseer_project(base_path)