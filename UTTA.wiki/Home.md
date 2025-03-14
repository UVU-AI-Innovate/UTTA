# Welcome to the UTTA Wiki

**UTTA** (Universal Teaching and Training Assistant) is an open-source framework designed to train and optimize AI models for educational applications. This wiki serves as the comprehensive documentation for the UTTA project.

## 🚀 Quick Start

* [Getting Started](Getting-Started) - Setup instructions and basic usage
* [Environment Setup](Environment-Setup) - How to set up your development environment
* [Code of Conduct](Code-of-Conduct) - Community guidelines for participation

## 📚 Documentation

### Framework Implementations

* [DSPy Tutorial](DSPy-Tutorial) - Learn to use DSPy for prompt optimization
* [OpenAI Tutorial](OpenAI-Tutorial) - Guide for using OpenAI models
* [HuggingFace Tutorial](HuggingFace-Tutorial) - Guide for using HuggingFace models

### Data Preparation

* [Dataset Preparation](Dataset-Preparation) - Learn how to prepare and format datasets for training

## 🏗️ Project Architecture

UTTA consists of two main components that work together to create AI-powered educational assistants:

1. **LLM Chatbot Framework** (`llm-chatbot-framework/`)
   * Core implementation for building teaching assistants with:
     * LLM integration (OpenAI, HuggingFace, DSPy)
     * Knowledge retrieval capabilities
     * Web interface for users
     * Evaluation tools to measure performance

2. **LLM Fine-Tuning Module** (`llm-fine-tuning/`)
   * Tools for customizing LLMs to educational tasks:
     * Multiple fine-tuning approaches (OpenAI, HuggingFace, DSPy)
     * Dataset preprocessing utilities
     * Performance comparison tools
     * Integration with the chatbot framework

### How Components Interact

```
┌────────────────────────┐      ┌────────────────────┐
│  LLM Chatbot Framework │◄─────┤  LLM Fine-Tuning   │
│                        │      │                    │
│  - User Interface      │      │  - Model Training  │
│  - Context Management  │      │  - Optimization    │
│  - Response Generation │      │  - Evaluation      │
└────────────┬───────────┘      └────────────────────┘
             │
             ▼
┌─────────────────────────┐
│      End Users          │
│  (Students & Educators) │
└─────────────────────────┘
```

The fine-tuned models are integrated into the chatbot framework to provide specialized educational assistance.

## 🔑 Key Features

* **Framework Agnostic** - Works with various LLM frameworks (OpenAI, HuggingFace, DSPy)
* **Optimization Tools** - Includes tools for prompt optimization and fine-tuning
* **Evaluation Suite** - Comprehensive tools for evaluating model performance
* **Educational Focus** - Specialized for educational use cases and teaching applications

## 🧩 Project Components in Detail

### Core Framework (`llm-chatbot-framework/`)

* **`src/core/`** - Core functionality and base classes
* **`src/llm/`** - LLM integrations for different providers
* **`src/retrieval/`** - Knowledge retrieval components
* **`src/web/`** - Web interface implementation
* **`src/evaluation/`** - Evaluation tools and metrics
* **`src/utils/`** - Utility functions and helpers
* **`knowledge_base/`** - Educational content storage
* **`examples/`** - Usage examples and demonstrations

### Fine-Tuning Module (`llm-fine-tuning/`)

* **`src/dspy/`** - DSPy-based fine-tuning implementation
* **`src/huggingface/`** - HuggingFace-based fine-tuning
* **`src/openai/`** - OpenAI-based fine-tuning
* **`examples/`** - Example fine-tuning workflows
* **`finetune.py`** - Main entry point for fine-tuning

## 🤝 Contributing

We welcome contributions from the community! Please check out:

* [Contributing Guide](Contributing) - How to contribute to the project

## 🆘 Support & Community

* GitHub Issues: For bug reports and feature requests
* GitHub Discussions: For general questions and discussions

## 📄 License

UTTA is released under the [MIT License](https://github.com/UVU-AI-Innovate/UTTA/blob/main/LICENSE).

---

Last updated: 2024-03-13 