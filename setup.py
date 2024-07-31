from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return [req for req in requirements if req and not req.startswith('#')]

setup(
    name="overseer",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires='>=3.10',
    install_requires=read_requirements(),
)