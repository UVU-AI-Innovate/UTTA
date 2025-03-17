# UTTA (Utah Teacher Training Assistant)

> A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)

## Prerequisites

- Python 3.10
- Conda package manager
- CUDA-capable GPU (for full functionality)
- OpenAI API key

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
- Use existing .env configuration
- Run system diagnostics
- Optionally start the web application

### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/majidmemari/UTTA.git
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

4. Environment Configuration:
   - The `.env` file is already configured with your OpenAI API key
   - Current configuration includes:
     - OpenAI API key
     - Model settings (gpt-3.5-turbo)
     - Temperature and token settings
     - Storage paths
     - Logging configuration
     - Web interface settings

5. Verify the installation:
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
├── examples/          # Example implementations
├── tests/            # Test directory
├── environment.yml   # Conda environment specification
├── requirements.txt  # Pip requirements (alternative to conda)
├── run_webapp.sh    # Setup and launch script
└── .env             # Configuration file
```

## Running the Application

### Web Interface
```bash
# Activate the environment (if not already active)
conda activate utta

# Start the web application
streamlit run src/web_app.py
```

### CLI Interface
```bash
# Activate the environment (if not already active)
conda activate utta

# Start the command-line interface
python src/cli.py
```

## Environment Details

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

## Troubleshooting

If you encounter issues:

1. Run the diagnostics tool:
```bash
python src/diagnostics.py
```

2. Common issues:
   - **OpenAI API errors**: Check your API key in .env
   - **CUDA errors**: Ensure CUDA toolkit is installed
   - **Import errors**: Verify conda environment is active
   - **Missing directories**: Run `mkdir -p data/index data/cache data/uploads`

3. For dependency issues:
```bash
# Fix OpenAI compatibility
python fix_openai_dep.py
```

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

