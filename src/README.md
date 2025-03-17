# Source Code Directory

This directory contains the core source code for the Universal Teacher Training Assistant (UTTA) application.

## Directory Structure

- `diagnostics.py` - System diagnostic tool for identifying and fixing issues
- `web_app.py` - Streamlit web interface for interacting with the teaching assistant
- `cli.py` - Command-line interface for the teaching assistant
- `evaluation/` - Evaluation metrics and response quality assessment
- `fine_tuning/` - Model fine-tuning and training components
- `knowledge_base/` - Document indexing, retrieval, and knowledge management
- `llm/` - LLM interfaces and DSPy handlers
- `metrics/` - Automated metrics for evaluating teaching responses

## Key Components

### Web Interface (`web_app.py`)
The Streamlit-based web interface allows users to interact with the teaching assistant in a user-friendly environment. It includes:
- Student profile configuration
- Teaching context settings
- Document management
- Response evaluation with detailed metrics

### Diagnostics (`diagnostics.py`)
System diagnostics tool that automatically identifies and resolves common issues:
- Checks for correct dependencies and versions
- Verifies environment setup
- Tests component initialization
- Provides automatic fixes for common problems

### CLI (`cli.py`)
Command-line interface for headless operation and testing. 