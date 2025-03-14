# UTTA (Utah Teacher Training Assistant)

UTTA is a comprehensive framework for developing and evaluating Large Language Model (LLM) based teaching assistants. This project focuses on creating effective, AI-powered educational support systems using state-of-the-art language models and evaluation methodologies.

## 🌟 Features

- **Multiple LLM Integration**
  - Support for various LLM frameworks (DSPy, OpenAI, HuggingFace)
  - Easy integration with popular models (LLaMA 2, Mistral, etc.)
  - Flexible model switching and comparison capabilities

- **Educational Focus**
  - Specialized for teaching assistant applications
  - Support for various educational scenarios
  - Customizable educational content delivery

- **Advanced Evaluation Framework**
  - Comprehensive metrics for response quality
  - Automated evaluation pipelines
  - Human-in-the-loop evaluation support

- **Terminal-Based Interface**
  - CLI tools for all core functionalities
  - Interactive chatbot terminal interface
  - Fine-tuning management through command line

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Conda package manager
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/UVU-AI-Innovate/UTTA.git
cd UTTA
```

2. Create and activate the conda environment:
```bash
conda create -n utta python=3.10
conda activate utta
```

3. Install the package and dependencies:
```bash
pip install -e .
```

## 📚 Documentation

Detailed documentation is available in our [GitHub Wiki](https://github.com/UVU-AI-Innovate/UTTA/wiki):

- [Getting Started](https://github.com/UVU-AI-Innovate/UTTA/wiki/Getting-Started)
- [Environment Setup](https://github.com/UVU-AI-Innovate/UTTA/wiki/Environment-Setup)
- [Dataset Preparation](https://github.com/UVU-AI-Innovate/UTTA/wiki/Dataset-Preparation)
- [Technical Architecture](https://github.com/UVU-AI-Innovate/UTTA/wiki/Technical-Architecture)
- [DSPy Tutorial](https://github.com/UVU-AI-Innovate/UTTA/wiki/DSPy-Tutorial)
- [OpenAI Tutorial](https://github.com/UVU-AI-Innovate/UTTA/wiki/OpenAI-Tutorial)
- [HuggingFace Tutorial](https://github.com/UVU-AI-Innovate/UTTA/wiki/HuggingFace-Tutorial)

## 🏗️ Project Structure

```
UTTA/
├── utta/                    # Main package
│   ├── core/                # Core functionality (Vector DB, Document Processor)
│   ├── chatbot/             # Chatbot functionality and LLM handlers
│   ├── fine_tuning/         # Fine-tuning implementations
│   ├── evaluation/          # Evaluation tools and metrics
│   └── utils/               # Utility functions and helpers
├── examples/                # Usage examples
│   ├── simple_chatbot.py    # Basic chatbot example
│   └── fine_tuning/         # Fine-tuning examples
├── tools/                   # CLI tools
│   └── fine_tuning_cli.py   # Fine-tuning CLI tool
├── tests/                   # Test files
│   └── test_chatbot.py      # Chatbot tests
├── data/                    # Data files
│   ├── evaluation/          # Evaluation results
│   └── benchmarks/          # Benchmark results
├── chatbot.py               # Convenience script to run the chatbot
├── finetune.py              # Convenience script to run the fine-tuning CLI
├── run_tests.py             # Script to run all tests
├── setup.py                 # Package installation setup
└── requirements.txt         # Project dependencies
```

## 🔧 Usage

### Simple Chatbot

Run the simple terminal-based chatbot using the convenience script:

```bash
# From the UTTA directory
./chatbot.py

# Or using the module path
python examples/simple_chatbot.py
```

### Fine-Tuning CLI

UTTA provides a comprehensive CLI tool for managing fine-tuning, accessible through a convenience script:

```bash
# From the UTTA directory
./finetune.py --help

# Or using the module path
python tools/fine_tuning_cli.py --help

# Or using the installed entry point
utta-finetune --help

# DSPy optimization
./finetune.py dspy optimize --examples examples.json --module teacher_responder

# OpenAI fine-tuning
./finetune.py openai start --file training_data.jsonl --model gpt-3.5-turbo

# HuggingFace fine-tuning
./finetune.py huggingface fine-tune --file training_data.jsonl --model microsoft/phi-2 --use-lora
```

### Running Tests

UTTA includes a comprehensive test suite:

```bash
# Run all tests
./run_tests.py

# Run individual test files
python tests/test_chatbot.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape UTTA
- Special thanks to the DSPy, OpenAI, and HuggingFace communities
- Gratitude to the educational institutions that have provided valuable feedback

## 📞 Contact

For questions and support, please:
- Open an issue in the GitHub repository
- Contact the maintainers directly
- Join our community discussions

---

*UTTA - Empowering Education with AI*