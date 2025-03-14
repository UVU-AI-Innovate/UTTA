# LLM Chatbot Framework

## Overview

This repository provides a comprehensive framework for developing advanced LLM-based chatbots with sophisticated knowledge retrieval capabilities. It demonstrates how to build domain-specific AI assistants that leverage both general LLM capabilities and specialized knowledge bases.

The repository includes a complete case study of an **Enhanced Teacher Training System** - a practical implementation of the framework for educational purposes. This system provides realistic classroom management scenarios for elementary education teachers, showcasing how domain-specific knowledge can be integrated with LLMs to create powerful, context-aware applications.

## Key Features

- **Document Processing Pipeline**: Process various document formats (PDF, EPUB, TXT, JSON) to extract domain knowledge
- **Vector Knowledge Base**: Store and retrieve information using semantic search in a SQLite vector database
- **Multi-Provider LLM Support**: Integrate with OpenAI, Anthropic, and local LLM models
- **DSPy Integration**: Leverage Stanford's DSPy framework for self-improving prompts and declarative programming
- **LlamaIndex Integration**: Utilize modern knowledge retrieval systems with configurable parameters
- **Performance Tracking**: Monitor which knowledge is most effective in different situations
- **Web Application Integration**: Ready-to-use utilities for Flask and FastAPI
- **Production Testing Tools**: Comprehensive testing and benchmarking tools
- **Fine-Tuning Capabilities**: Three approaches to fine-tuning (DSPy, OpenAI, HuggingFace)

## Directory Structure

```
├── cache/                     # Cache for fine-tuning and optimized modules
│   └── optimized_modules/     # Saved DSPy optimized modules
├── data/                      # Data storage
│   └── sample_documents/      # Sample educational documents
├── docs/                      # Documentation
│   ├── fine-tuning/           # Fine-tuning documentation
│   └── Home.md                # Documentation home page
├── examples/                  # Example applications
│   ├── simple_chatbot.py      # Simple chatbot example
│   └── teacher_training/      # Teacher training system
├── knowledge_base/            # Knowledge base storage
│   ├── documents/             # Raw documents
│   ├── embeddings/            # Saved embeddings
│   └── vectors/               # Vector database files
├── logs/                      # Log files
├── src/                       # Core framework code
│   ├── core/                  # Core components
│   │   ├── document_processor.py # Document processing utilities
│   │   └── vector_database.py # Vector database implementation
│   ├── llm/                   # LLM integration
│   │   ├── dspy/              # DSPy integration
│   │   ├── fine_tuning/       # Fine-tuning implementations
│   │   ├── handlers/          # LLM handlers
│   │   └── prompts/           # Prompt templates
│   ├── retrieval/             # Retrieval components
│   │   ├── basic/             # Basic retrieval methods
│   │   └── llama_index/       # LlamaIndex integration
│   ├── utils/                 # Utility functions
│   └── web/                   # Web application components
│       └── static/            # Static web assets
├── tests/                     # Unit and integration tests
│   ├── core/                  # Tests for core components
│   ├── examples/              # Tests for examples
│   ├── llm/                   # Tests for LLM components
│   └── retrieval/             # Tests for retrieval components
├── tools/                     # Utility tools
│   ├── fine_tuning_cli.py     # Fine-tuning CLI tool
│   ├── knowledge_analyzer.py  # Knowledge base analysis tool
│   ├── performance_testing/   # Performance testing tools
│   └── setup/                 # Setup scripts
├── .gitignore                 # Git ignore file
├── requirements.txt           # Dependencies
├── setup.py                   # Package setup file
└── README.md                  # This file
```

## Installation

### Using Conda (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-chatbot-framework.git
   cd llm-chatbot-framework
   ```

2. Create and activate the conda environment:
   ```
   conda create -n utta python=3.10
   conda activate utta
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```
   pip install -e .
   ```

5. Fix any environment issues (recommended):
   ```
   cd tools/setup
   bash fix_environment.sh
   ```

### Using venv (Alternative)

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/llm-chatbot-framework.git
   cd llm-chatbot-framework
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```
   pip install -e .
   ```

5. Set up environment variables:
   Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

### Simple Chatbot Example

To run the simple chatbot example:

```bash
conda activate utta  # If using conda
python examples/simple_chatbot.py
```

This will:
1. Initialize a simple vector database with sample knowledge
2. Start an interactive chat session where you can ask questions about classroom management, instruction, or assessment

### Teacher Training System Example

The Teacher Training System is an example application that demonstrates how to use the framework to create a system that helps teachers practice classroom scenarios.

To run the Teacher Training System:

```bash
conda activate utta  # If using conda
python -m examples.teacher_training.run
```

This will:
1. Process educational documents in `data/sample_documents/`
2. Build a knowledge base
3. Start an interactive session where you can practice responding to classroom scenarios
4. Receive feedback on your teaching strategies and see simulated student responses

## Fine-Tuning Capabilities

The framework includes comprehensive fine-tuning capabilities that allow you to adapt language models to your specific educational domains and tasks. Three main approaches are supported:

### 1. DSPy Optimization (Soft Fine-Tuning)

DSPy optimization provides a "soft fine-tuning" approach that optimizes prompts rather than model weights. This is faster, cheaper, and often very effective for specialized tasks.

For detailed documentation, see [DSPy Optimization](docs/fine-tuning/DSPy-Optimization.md).

### 2. OpenAI Fine-Tuning

For applications requiring deeper customization, the framework supports OpenAI's fine-tuning API to create custom versions of models like GPT-3.5.

For detailed documentation, see [OpenAI Fine-Tuning](docs/fine-tuning/OpenAI-Fine-Tuning.md).

### 3. HuggingFace Model Fine-Tuning

For fully local deployment, the framework supports fine-tuning Hugging Face models using techniques like LoRA (Low-Rank Adaptation) for efficient training.

For detailed documentation, see [HuggingFace Fine-Tuning](docs/fine-tuning/HuggingFace-Fine-Tuning.md).

### Fine-Tuning CLI Tool

The framework includes a command-line tool to manage fine-tuning jobs:

```bash
# Get help
python tools/fine_tuning_cli.py --help

# DSPy optimization
python tools/fine_tuning_cli.py dspy optimize --examples examples.json --module teacher_responder

# OpenAI fine-tuning
python tools/fine_tuning_cli.py openai start --file training_data.jsonl --model gpt-3.5-turbo

# HuggingFace fine-tuning
python tools/fine_tuning_cli.py huggingface fine-tune --file training_data.jsonl --model microsoft/phi-2 --use-lora
```

For detailed documentation, see [Fine-Tuning CLI](docs/fine-tuning/Fine-Tuning-CLI.md).

## Documentation

Comprehensive documentation is available in the `docs` directory:

- [Home](docs/Home.md) - Documentation home page
- [Fine-Tuning Overview](docs/fine-tuning/Fine-Tuning-Overview.md) - Overview of fine-tuning approaches
- [Best Practices](docs/fine-tuning/Fine-Tuning-Best-Practices.md) - Best practices for fine-tuning
- [Troubleshooting](docs/fine-tuning/Fine-Tuning-Troubleshooting.md) - Solutions to common issues

## Troubleshooting

If you encounter any issues:

1. Make sure you have activated the correct environment:
   ```
   conda activate utta
   ```

2. Run the environment fix script:
   ```
   cd tools/setup
   bash fix_environment.sh
   ```

3. Check the log files for detailed error messages in the `logs` directory.

4. Common issues:
   - Invalid distribution warning: This can be ignored, or fix with `find ~/.local/lib/python3.10/site-packages -name "*-orch*" -type d | xargs rm -rf`
   - Missing dependencies: Make sure all dependencies are installed with `pip install -r requirements.txt`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](./LICENSE). 