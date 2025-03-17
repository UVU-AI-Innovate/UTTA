#!/bin/bash

echo "==============================================="
echo "Teaching Assistant Web Application Setup Script"
echo "==============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Activate conda environment or create it if it doesn't exist
if conda info --envs | grep -q "utta"; then
    echo "Activating existing conda environment 'utta'..."
    eval "$(conda shell.bash hook)"
    conda activate utta
else
    echo "Creating new conda environment 'utta'..."
    conda create -y -n utta python=3.10
    eval "$(conda shell.bash hook)"
    conda activate utta
fi

# Install core dependencies
echo "Installing core dependencies..."
pip install --no-cache-dir streamlit==1.32.0 python-dotenv textstat
pip install --no-cache-dir sentence-transformers==2.2.2 faiss-gpu
pip install --no-cache-dir spacy
python -m spacy download en_core_web_sm

# Critical dependency: Install specific version of OpenAI that works with DSPy
echo "Installing OpenAI version compatible with DSPy..."
pip install --no-cache-dir openai==0.28.0

# Install DSPy (after OpenAI)
echo "Installing DSPy..."
pip install --no-cache-dir dspy-ai==2.0.4

# Create required directories
echo "Creating data directories..."
mkdir -p data/index
mkdir -p data/cache
mkdir -p data/uploads

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Please ensure you have a valid .env file with your OpenAI API key."
    echo "You can create one by copying .env.example and updating the API key."
    exit 1
fi

# Run diagnostics
echo "Running diagnostics..."
python src/diagnostics.py

# Ask if user wants to run the web app
echo ""
echo "Do you want to run the web application now? (y/n)"
read run_app

if [[ $run_app == "y" || $run_app == "Y" ]]; then
    echo "Starting web application..."
    streamlit run src/web_app.py
else
    echo "You can run the web application later with: conda activate utta && streamlit run src/web_app.py"
fi

echo "Setup complete!" 