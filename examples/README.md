# LLM Fine-Tuning Examples (v0.1.0)

> Initial release of educational examples for LLM fine-tuning approaches

This directory contains examples of different approaches to fine-tuning large language models (LLMs) for educational question answering. These examples are designed to help educators and developers understand different LLM adaptation methods.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)

## 🚀 Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run any example:
   ```bash
   python dspy_example.py      # For prompt optimization
   python openai_finetune.py   # For cloud fine-tuning
   python huggingface_lora.py  # For local fine-tuning
   ```

## 🎯 Overview

This project demonstrates three approaches to improving LLM performance:

| Approach | Description | Best For |
|----------|------------|----------|
| **DSPy** | Prompt optimization without model changes | Quick prototyping |
| **OpenAI** | Cloud-based fine-tuning | Production deployment |
| **HuggingFace** | Local fine-tuning with LoRA | Full customization |

## 📊 Comparison

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (5 examples) | Medium (10+ examples) | Large (20+ examples) |
| Cost | API calls only | API + training | One-time compute |
| Setup difficulty | Simple | Simple | Complex |
| Hardware required | None | None | GPU (16GB+ VRAM) |
| Time to results | Immediate | Hours | Hours (with GPU) |

## 📁 Project Structure

```
examples/
├── dspy_example.py         # Prompt optimization example
├── openai_finetune.py     # OpenAI fine-tuning example
├── huggingface_lora.py    # HuggingFace LoRA example
├── small_edu_qa.jsonl     # Base dataset (5 examples)
└── openai_edu_qa_training.jsonl  # Extended dataset
```

## 💻 Requirements

### Basic Requirements (All Examples)
- Python 3.8+
- pip or conda

### Example-Specific Requirements

**DSPy Example**
- OpenAI API key
- dspy library
- Internet connection

**OpenAI Example**
- OpenAI API key
- openai library
- OpenAI account with billing

**HuggingFace Example**
- GPU with 16GB+ VRAM
- CUDA support
- transformers, peft libraries

## 🔍 Example Usage

### DSPy Example
```python
# Simple prompt optimization
python dspy_example.py
```

### OpenAI Example
```python
# Cloud-based fine-tuning
python openai_finetune.py
```

### HuggingFace Example
```python
# Local LoRA fine-tuning
python huggingface_lora.py
```

## 🎓 Educational Purpose

Each example demonstrates a different approach to LLM adaptation:

1. **DSPy**: Prompt optimization for quick results
2. **OpenAI**: Cloud fine-tuning for scalability
3. **HuggingFace**: Local fine-tuning for full control

## 🤝 Contributing

This is an alpha release - contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📝 License

Copyright (c) 2025 Utah Teacher Training Assistant (UTTA)

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 📋 Todo

- [ ] Add more example datasets
- [ ] Include evaluation metrics
- [ ] Add automated testing
- [ ] Improve documentation
- [ ] Add more fine-tuning approaches

## 🔄 Version History

- 0.1.0 (Initial Release)
  - Basic examples for three approaches
  - Educational QA dataset
  - Documentation and comparisons