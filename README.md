# UTTA (Utah Teacher Training Assistant)

> A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)

## Features

- **Enhanced LLM Interface**: Optimized for pedagogical interactions using DSPy
- **Knowledge Base Integration**: Efficient document indexing and retrieval
- **Fine-Tuning Capabilities**: Specialized model training for educational contexts
- **Automated Evaluation**: Comprehensive metrics for teaching responses
- **Web Interface**: Interactive testing and development environment
- **CLI Interface**: Command-line interface for testing and development
- **Automated Debugging**: Comprehensive diagnostic tools for quick issue resolution
- **Robust Fallbacks**: Graceful degradation when dependencies are unavailable

## Project Structure

```
UTTA/
├── data/               # Training data and model storage
│   ├── cache/         # Cache for embeddings and API responses
│   ├── index/         # Document index storage
│   ├── models/        # Trained and fine-tuned models
│   ├── test/          # Test datasets
│   └── uploads/       # User-uploaded documents
├── src/               # Source code
│   ├── diagnostics.py # System diagnostics tool
│   ├── cli.py         # Command-line interface
│   ├── web_app.py     # Web application interface
│   ├── metrics/       # Response evaluation metrics
│   ├── fine_tuning/   # Model fine-tuning components
│   ├── knowledge_base/# Document indexing and retrieval
│   └── llm/          # LLM interface and DSPy handlers
├── tests/             # Test directory
│   ├── unit/         # Unit tests
│   └── integration/  # Integration tests
├── fix_openai_dep.py  # OpenAI dependency fix script  
├── run_webapp.sh      # Setup and launch script
└── environment.yml    # Conda environment file
```

## Installation

### Option 1: Quick Start (Recommended)

The easiest way to get started is to use our setup script:

```bash
# Make the script executable
chmod +x run_webapp.sh

# Run the setup script (creates environment, installs dependencies, and starts the app)
./run_webapp.sh
```

### Option 2: Conda Environment (Manual Setup)

1. Clone the repository:
```bash
git clone https://github.com/UVU-AI-Innovate/UTTA.git
cd UTTA
```

2. Create and activate the conda environment:
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate utta
```

3. Verify the installation:
```bash
# Check if all major components are installed correctly
python -c "import dspy, torch, spacy; print(f'Python {torch.__version__}, DSPy {dspy.__version__}, SpaCy {spacy.__version__}')"
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
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

- **Development Tools**:
  - pytest>=7.0.0
  - black>=23.0.0
  - flake8>=6.0.0

For a complete list of dependencies, see `environment.yml`.

## Running the Application

### Web Interface
```bash
# Start the web application
streamlit run src/web_app.py
```

### CLI Interface
```bash
# Start the command-line interface
python src/cli.py
```

The CLI interface provides:
- Interactive chat with the teaching assistant
- Configuration of settings (grade level, subject, learning style)
- Document management
- Response evaluation with metrics

## Troubleshooting

If you encounter issues, use our diagnostic tools:

```bash
# Run system diagnostics (checks environment, dependencies, components)
python src/diagnostics.py

# Run diagnostics with automatic fixes
python src/diagnostics.py --fix

# Fix OpenAI compatibility issues specifically
python fix_openai_dep.py
```

Common issues and solutions:

1. **LLM initialization fails**: Run `fix_openai_dep.py` to install the correct OpenAI version
2. **Missing components**: Check diagnostic output and install missing dependencies
3. **API key issues**: Ensure your OpenAI API key is in the `.env` file

## Core Dependencies

- **LLM and NLP**:
  - dspy-ai==2.0.4
  - openai==0.28.0 (specific version required)
  - spacy>=3.7.0
  - streamlit==1.32.0

- **Knowledge Base**:
  - sentence-transformers==2.2.2
  - faiss-gpu

- **Data Processing**:
  - pandas
  - numpy
  - torch

For a complete list of dependencies, see `environment.yml`.

## Components

### LLM Interface
- Enhanced DSPy integration for pedagogical interactions
- Context-aware response generation
- Performance optimization with caching
- Graceful degradation when components are unavailable

### Knowledge Base
- Document chunking and indexing
- Semantic search capabilities
- Knowledge graph for pedagogical relationships

### Fine-Tuning
- Custom training data preparation
- DSPy-based optimization
- Model evaluation metrics

### Automated Evaluation
- Response clarity assessment
- Engagement metrics
- Age-appropriate content verification
- Pedagogical elements analysis

### Interfaces
- **Web Interface**: Streamlit-based UI with visualization and metrics
- **CLI Interface**: Command-line tool for testing and development

### Diagnostic Tools
- System dependency checks
- Component initialization verification
- Automatic issue detection and repair
- Environment validation

## Testing

Run the test suite:
```bash
# Navigate to the project root
cd /path/to/UTTA

# Run all tests
pytest tests/

# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Run specific test files
pytest tests/unit/test_components.py
pytest tests/unit/test_web_app.py
pytest tests/integration/test_integration.py

# Run quick test script
python tests/quick_test.py
```

## Development

1. Code Style:
```bash
# Format code
black .

# Check style
flake8
```

2. Running the Web Interface:
```bash
streamlit run src/web_app.py
```

3. Running the CLI Interface:
```bash
python src/cli.py
```

4. Running Diagnostics:
```bash
python src/diagnostics.py
```

## Recent Improvements

- **Initial Release (v0.1.0)**:
  - Basic framework implementation
  - DSPy integration for pedagogical interactions
  - Knowledge base management system
  - Web and CLI interfaces
  - Comprehensive documentation
  - Example implementations
- **CLI Interface**: Added a command-line interface for testing and development
- **Robust Error Handling**: Improved error handling throughout the application
- **Dependency Management**: Better handling of missing or incompatible dependencies
- **Module Structure**: Reorganized code into proper Python packages
- **Documentation**: Added comprehensive README files for each module
- **Metrics Module**: Enhanced evaluation system with detailed metrics
- **Testing Infrastructure**: Improved test organization and coverage

## Version History

- **0.1.0** (Initial Release - March 2024)
  - Basic framework implementation
  - Core functionality and interfaces
  - Documentation and examples
  - Testing infrastructure

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and diagnostics
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Copyright (c) 2025 Utah Teacher Training Assistant (UTTA)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DSPy team for the foundational LLM framework
- Contributors and testers

