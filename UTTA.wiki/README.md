# UTTA (Utah Teacher Training Assistant)

UTTA is a comprehensive framework for developing and evaluating Large Language Model (LLM) based teaching assistants. This project focuses on creating effective, AI-powered educational support systems using state-of-the-art language models and evaluation methodologies.

## 🌟 Features

- **Multiple LLM Integration**
  - Support for various LLM frameworks (DSPy, LangChain, LlamaIndex)
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

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📚 Documentation

Detailed documentation is available in our [GitHub Wiki](https://github.com/UVU-AI-Innovate/UTTA/wiki):

- [Getting Started](https://github.com/UVU-AI-Innovate/UTTA/wiki/Getting-Started)
- [Environment Setup](https://github.com/UVU-AI-Innovate/UTTA/wiki/Environment-Setup)
- [Dataset Preparation](https://github.com/UVU-AI-Innovate/UTTA/wiki/Dataset-Preparation)
- [DSPy Tutorial](https://github.com/UVU-AI-Innovate/UTTA/wiki/DSPy-Tutorial)
- [DSPy Optimization](https://github.com/UVU-AI-Innovate/UTTA/wiki/DSPy-Optimization)
- [OpenAI Tutorial](https://github.com/UVU-AI-Innovate/UTTA/wiki/OpenAI-Tutorial)
- [HuggingFace Tutorial](https://github.com/UVU-AI-Innovate/UTTA/wiki/HuggingFace-Tutorial)
- [LLM Evaluation](https://github.com/UVU-AI-Innovate/UTTA/wiki/LLM-Evaluation)

## 🏗️ Project Structure

```
UTTA/
├── llm-fine-tuning/        # Fine-tuning implementations
├── llm-chatbot-framework/  # Core chatbot framework
├── evaluation_results/     # Evaluation outputs
├── benchmark_results/      # Benchmark data
└── docs/                   # Additional documentation
```

## 🔧 Usage

UTTA supports multiple LLM frameworks, each with its own implementation approach:

### DSPy Implementation
```python
# Example DSPy implementation
from utta.dspy import TeachingAssistant

assistant = TeachingAssistant()
response = assistant.answer_question("Explain photosynthesis")
```

### LangChain Implementation
```python
# Example LangChain implementation
from utta.langchain import UTTAChain

chain = UTTAChain()
response = chain.run("Explain the concept of gravity")
```

### LlamaIndex Implementation
```python
# Example LlamaIndex implementation
from utta.llamaindex import UTTAIndex

index = UTTAIndex()
response = index.query("What is the theory of relativity?")
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
- Special thanks to the DSPy, LangChain, and LlamaIndex communities
- Gratitude to the educational institutions that have provided valuable feedback

## 📞 Contact

For questions and support, please:
- Open an issue in the GitHub repository
- Contact the maintainers directly
- Join our community discussions

---

*UTTA - Empowering Education with AI*