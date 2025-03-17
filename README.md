# UTTA (Utah Teacher Training Assistant)

A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

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

## Quick Start

The easiest way to get started is to use our setup script:

```bash
# Make the script executable
chmod +x run_webapp.sh

# Run the setup script (creates environment, installs dependencies, and starts the app)
./run_webapp.sh
```

## Manual Setup

1. Create and activate a conda environment:
```bash
conda create -n utta python=3.10
conda activate utta
```

2. Install dependencies in the correct order:
```bash
# Install base dependencies
pip install python-dotenv streamlit==1.32.0 textstat sentence-transformers==2.2.2 faiss-gpu spacy

# Install specific OpenAI version (critical for DSPy compatibility)
pip install openai==0.28.0

# Install DSPy after OpenAI
pip install dspy-ai==2.0.4

# Download spaCy model
python -m spacy download en_core_web_sm
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

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

- **CLI Interface**: Added a command-line interface for testing and development
- **Robust Error Handling**: Improved error handling throughout the application
- **Dependency Management**: Better handling of missing or incompatible dependencies
- **Module Structure**: Reorganized code into proper Python packages
- **Documentation**: Added comprehensive README files for each module
- **Metrics Module**: Enhanced evaluation system with detailed metrics
- **Testing Infrastructure**: Improved test organization and coverage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and diagnostics
5. Submit a pull request

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

- DSPy team for the foundational LLM framework
- Contributors and testers

