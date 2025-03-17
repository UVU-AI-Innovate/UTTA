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
    echo "Creating new conda environment from environment.yml..."
    conda env create -f environment.yml
    eval "$(conda shell.bash hook)"
    conda activate utta
    
    # Download spaCy model if needed
    echo "Ensuring spaCy model is available..."
    python -c "import spacy; spacy.load('en_core_web_sm')" || python -m spacy download en_core_web_sm
fi

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