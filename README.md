# UTTA (Universal Teacher Training Assistant)

A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

## Features

- **Enhanced LLM Interface**: Optimized for pedagogical interactions using DSPy
- **Knowledge Base Integration**: Efficient document indexing and retrieval
- **Fine-Tuning Capabilities**: Specialized model training for educational contexts
- **Automated Evaluation**: Comprehensive metrics for teaching responses
- **Web Interface**: Interactive testing and development environment

## Project Structure

```
UTTA/
├── data/               # Training data and model storage
├── src/               # Source code
│   ├── evaluation/    # Response evaluation metrics
│   ├── fine_tuning/   # Model fine-tuning components
│   ├── knowledge_base/# Document indexing and retrieval
│   └── llm/          # LLM interface and DSPy handlers
├── tests/             # Unit tests
└── test_*.py         # Integration and component tests
```

## Setup

1. Create and activate a conda environment:
```bash
conda create -n utta python=3.10
conda activate utta
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Core Dependencies

- **LLM and NLP**:
  - dspy-ai>=2.6.12
  - langchain>=0.1.0
  - transformers>=4.49.0
  - spacy>=3.8.4

- **Knowledge Base**:
  - sentence-transformers>=3.4.1
  - llama-index>=0.12.22
  - faiss-cpu>=1.7.4

- **Data Processing**:
  - pandas>=2.2.3
  - numpy>=1.26.4
  - torch>=2.1.0

For a complete list of dependencies, see `requirements.txt`.

## Components

### LLM Interface
- Enhanced DSPy integration for pedagogical interactions
- Context-aware response generation
- Performance optimization with caching

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

- DSPy team for the foundational LLM framework
- Contributors and testers