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

# Check if .env file exists, create it if not
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env 2>/dev/null || echo "# API Keys
OPENAI_API_KEY=your_api_key_here

# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000

# Storage Paths
CACHE_DIR=data/cache
INDEX_DIR=data/index
MODEL_DIR=data/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/app.log

# Web Interface
STREAMLIT_PORT=8501
STREAMLIT_THEME=light" > .env
    echo "Please edit .env file to add your OpenAI API key"
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