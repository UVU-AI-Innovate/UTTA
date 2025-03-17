# UTTA (Utah Teacher Training Assistant)

> A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)

## Table of Contents
- [UTTA (Utah Teacher Training Assistant)](#utta-utah-teacher-training-assistant)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
    - [Option 1: Automated Setup (Recommended)](#option-1-automated-setup-recommended)
    - [Option 2: Manual Setup](#option-2-manual-setup)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
    - [Web Interface](#web-interface)
    - [CLI Interface](#cli-interface)
  - [Features](#features)
  - [Environment Configuration](#environment-configuration)
    - [Environment Details](#environment-details)
  - [Examples Directory](#examples-directory)
    - [Example Categories](#example-categories)
      - [DSPy Examples (`examples/dspy/`)](#dspy-examples-examplesdspy)
      - [OpenAI Examples (`examples/openai/`)](#openai-examples-examplesopenai)
      - [Hugging Face Examples (`examples/huggingface/`)](#hugging-face-examples-exampleshuggingface)
    - [Learning Focus](#learning-focus)
  - [Troubleshooting](#troubleshooting)
    - [Common Issues](#common-issues)
    - [Diagnostics](#diagnostics)
    - [Getting Help](#getting-help)
  - [License](#license)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)

## Prerequisites

- Python 3.10
- Conda package manager
- CUDA-capable GPU (for full functionality)
- OpenAI API key (already configured in `.env`)

## Installation

### Option 1: Automated Setup (Recommended)

The easiest way to get started is to use our setup script:

```bash
# Clone the repository
git clone https://github.com/majidmemari/UTTA.git
cd UTTA

# Make the script executable
chmod +x run_webapp.sh

# Run the setup script
./run_webapp.sh
```

The script will:
- Create and configure the conda environment
- Install all required dependencies
- Set up necessary directories
- Use existing `.env` configuration
- Run system diagnostics
- Optionally start the web application

### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/UVU-AI-Innovate/UTTA
cd UTTA
```

2. Create and activate the conda environment:
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate utta
```

3. Create required directories:
```bash
mkdir -p data/index data/cache data/uploads
```

4. Verify the installation:
```bash
# Run system diagnostics
python src/diagnostics.py
```

## Project Structure

```
UTTA/
├── data/               # Data storage
│   ├── cache/         # Cache for embeddings and API responses
│   ├── index/         # Document index storage
│   └── uploads/       # User-uploaded documents
├── src/               # Source code
│   ├── diagnostics.py # System diagnostics tool
│   ├── cli.py        # Command-line interface
│   ├── web_app.py    # Web application interface
│   ├── metrics/      # Response evaluation metrics
│   ├── fine_tuning/  # Model fine-tuning components
│   ├── knowledge_base/# Document indexing and retrieval
│   └── llm/          # LLM interface and DSPy handlers
├── examples/          # Example implementations (see Examples Directory section)
├── tests/            # Test directory
├── environment.yml   # Conda environment specification
├── run_webapp.sh    # Setup and launch script
├── .env             # Configuration file
└── LICENSE          # MIT License file
```

## Usage

### Web Interface

```bash
# Activate the environment (if not already active)
conda activate utta

# Start the web application
streamlit run src/web_app.py
```

Then open your browser and navigate to:
```
http://localhost:8501
```

### CLI Interface

```bash
# Activate the environment (if not already active)
conda activate utta

# Start the command-line interface
python src/cli.py
```

## Features

- Interactive educational conversations
- Adaptive learning paths
- Multi-modal content support
- Progress tracking
- Customizable teaching styles
- Knowledge base integration
- RAG (Retrieval Augmented Generation)
- Fine-tuning capabilities
- Evaluation metrics
- Web and CLI interfaces

## Environment Configuration

The `.env` file contains the following configuration:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key

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
STREAMLIT_THEME=light
```

### Environment Details

The project uses a conda environment with the following key components:

- **Python Version**: 3.10
- **Deep Learning**:
  - PyTorch 2.2.0
  - CUDA Toolkit 11.8
  - torchvision

- **Core Dependencies**:
  - streamlit==1.32.0
  - openai==0.28.0 (specific version required for DSPy)
  - dspy-ai==2.0.4
  - sentence-transformers==2.2.2
  - faiss-gpu>=1.7.4
  - spacy>=3.7.0

For a complete list of dependencies, see `environment.yml`.

## Examples Directory

The `examples/` directory contains educational examples and tutorials to help you understand and work with different aspects of the UTTA project:

### Example Categories

#### DSPy Examples (`examples/dspy/`)
- `optimize_educational_qa.py`: Demonstrates how to optimize educational Q&A using DSPy
- `custom_teleprompter.py`: Shows how to create custom teleprompters for educational content

#### OpenAI Examples (`examples/openai/`)
- `finetune_educational_qa.py`: Example of fine-tuning OpenAI models for educational Q&A
- `chat_completion_examples.py`: Various chat completion examples with different parameters

#### Hugging Face Examples (`examples/huggingface/`)
- `finetune_educational_qa.py`: Example of fine-tuning Hugging Face models
- `model_inference.py`: Demonstrates model inference with different Hugging Face models

### Learning Focus

Each example includes detailed comments and documentation to help you understand:
- How to set up the environment
- Required dependencies
- Usage examples
- Best practices
- Common pitfalls to avoid

See `examples/README.md` for more details on the different approaches to fine-tuning and their use cases.

## Troubleshooting

### Common Issues

1. **Environment Setup**
   - Ensure conda is installed and initialized
   - Check Python version compatibility
   - Verify all dependencies are installed

2. **API Key Issues**
   - Verify OPENAI_API_KEY is set in `.env`
   - Check API key validity
   - Ensure proper permissions

3. **Directory Structure**
   - Verify all required directories exist
   - Check directory permissions
   - Ensure proper path configuration

4. **OpenAI API errors**
   - Check your API key in `.env`
   - Verify connectivity

5. **CUDA errors**
   - Ensure CUDA toolkit is installed
   - Check GPU compatibility

6. **Import errors**
   - Verify conda environment is active
   - Use `fix_openai_dep.py` for OpenAI compatibility issues

### Diagnostics

Run the diagnostics tool to identify issues:
```bash
python src/diagnostics.py
```

### Getting Help

- Check the documentation
- Review example code
- Open an issue for bugs
- Contact maintainers for support

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Copyright (c) 2025 Utah Teacher Training Assistant (UTTA)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and diagnostics
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- DSPy team for the foundational LLM framework
- Contributors and testers

