"""
pip install -e .   →  adds `archon` to your PATH via the entry_points mechanism.
Works on Windows, Linux, and macOS.
"""

from setuptools import setup, find_packages

setup(
    name="archon-cli",
    version="1.0.0",
    description="Archon – Autonomous Coding Agent CLI",
    packages=find_packages(include=["archon", "archon.*"]),
    package_dir={"archon": "archon"},
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.32",
        "rich>=13.0",
        "python-dotenv>=1.0",
        "prompt_toolkit>=3.0",
    ],
    entry_points={
        "console_scripts": [
            "archon=archon.cli:main",
        ],
    },
)
