#!/bin/bash

# Exit on error
set -e

echo "Starting Streamlit upgrade process..."

# Check if conda exists
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found in PATH"
    echo "Make sure conda is installed and in your PATH"
    exit 1
fi

# Activate conda environment
echo "Activating conda environment..."
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate utta

# Check if activation succeeded
if [[ $CONDA_DEFAULT_ENV != "utta" ]]; then
    echo "Error: Failed to activate utta environment"
    exit 1
fi

# Upgrade Streamlit
echo "Upgrading Streamlit..."
pip install streamlit --upgrade

echo "Upgrade completed. You can now run the app with ./start_app.sh" 