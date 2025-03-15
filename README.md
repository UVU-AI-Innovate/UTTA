# UTTA (Universal Teacher Training Assistant)

A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

## Features

- **Enhanced LLM Interface**: Optimized for pedagogical interactions using DSPy
- **Knowledge Base Integration**: Efficient document indexing and retrieval
- **Fine-Tuning Capabilities**: Specialized model training for educational contexts
- **Automated Evaluation**: Comprehensive metrics for teaching responses
- **Web Interface**: Interactive testing and development environment
- **Automated Debugging**: Comprehensive diagnostic tools for quick issue resolution

## Project Structure

```
UTTA/
├── data/               # Training data and model storage
├── src/               # Source code
│   ├── diagnostics.py # System diagnostics tool
│   ├── evaluation/    # Response evaluation metrics
│   ├── fine_tuning/   # Model fine-tuning components
│   ├── knowledge_base/# Document indexing and retrieval
│   └── llm/          # LLM interface and DSPy handlers
├── fix_openai_dep.py  # OpenAI dependency fix script  
├── run_webapp.sh      # Setup and launch script
├── tests/             # Unit tests
└── test_*.py         # Integration and component tests
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

### Diagnostic Tools
- System dependency checks
- Component initialization verification
- Automatic issue detection and repair
- Environment validation

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test files
pytest test_components.py
pytest test_web_app.py
pytest test_integration.py
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

3. Running Diagnostics:
```bash
python src/diagnostics.py
```

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