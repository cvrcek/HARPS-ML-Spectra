#!/bin/bash

# Create virtual environment
python3.10 -m venv replicated_env

# Activate the virtual environment
source replicated_env/bin/activate

# Install the package in editable mode with all dependencies
pip install -e .
