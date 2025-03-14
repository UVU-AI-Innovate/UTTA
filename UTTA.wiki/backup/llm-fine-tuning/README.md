# LLM Fine-Tuning Toolkit

A streamlined toolkit for fine-tuning language models using three different approaches. This project emphasizes simplicity, educational value, and practical applications.

## Overview

This toolkit demonstrates three approaches to LLM fine-tuning, from least to most resource-intensive:

1. **DSPy Optimization** - Zero-shot prompt optimization without weight changes
2. **OpenAI Fine-Tuning** - Cloud-based fine-tuning through OpenAI's API
3. **HuggingFace Fine-Tuning** - Local fine-tuning using LoRA for efficiency

Each approach is implemented as a modular Python package with consistent interfaces.

## 📋 Comparison of Approaches

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| **What changes** | Prompts | Model weights (cloud) | Model weights (local) |
| **Training data needed** | Small (10s) | Medium (100s-1000s) | Medium/Large (1000s+) |
| **Cost** | API calls only | API + training | One-time compute |
| **Setup difficulty** | Simple | Simple | Complex |
| **Control** | Limited | Medium | Full |
| **Hardware required** | None | None | GPU recommended |
| **Deployment** | API calls | API calls | Self-host |
| **Data privacy** | Data shared with API | Data shared with API | Data stays local |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/llm-fine-tuning.git
cd llm-fine-tuning

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### Running Examples

```bash
# Set OpenAI API key (required for DSPy and OpenAI examples)
export OPENAI_API_KEY=your-api-key-here

# Try each approach
python examples/dspy/optimize_educational_qa.py       # Fast, no training
python examples/openai/finetune_educational_qa.py     # Medium, API-based
python examples/huggingface/finetune_educational_qa.py  # Slow, local training

# Or compare all approaches
python compare_approaches.py
```

## 📁 Project Structure

```
llm-fine-tuning/
├── src/                          # Core implementation
│   ├── dspy/                     # DSPy prompt optimization
│   │   ├── models.py             # Educational QA model
│   │   └── optimizer.py          # DSPy optimization logic
│   ├── openai/                   # OpenAI fine-tuning
│   │   └── finetuner.py          # OpenAI API integration
│   ├── huggingface/              # HuggingFace fine-tuning
│   │   └── finetuner.py          # Local LoRA fine-tuning
│   └── utils.py                  # Shared utilities
├── examples/                     # Example implementations
│   ├── data/                     # Sample datasets
│   │   └── educational_qa_sample.jsonl  # Education QA examples
│   ├── dspy/                     # DSPy example
│   ├── openai/                   # OpenAI example
│   └── huggingface/              # HuggingFace example
├── compare_approaches.py         # Script to compare all methods
├── check_structure.py            # Verify project structure
├── setup.py                      # Setup script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 💡 When to Use Each Approach

- **DSPy Optimization**: When you need quick improvements with minimal data and don't want to change model weights
- **OpenAI Fine-Tuning**: When you need reliable, hosted solutions with moderate customization
- **HuggingFace Fine-Tuning**: When you need full control, data privacy, or specialized customization

## 📚 Educational Question-Answering

All examples in this project focus on educational question-answering, demonstrating:

- How to format data for each approach
- The appropriate level of training data
- The tradeoffs in complexity vs customization

## 📝 License

MIT

## 🙏 Acknowledgements

This project uses:
- [DSPy](https://github.com/stanfordnlp/dspy) - Stanford NLP's framework for prompt programming
- [OpenAI API](https://github.com/openai/openai-python) - For cloud-based LLM access and fine-tuning
- [Transformers](https://github.com/huggingface/transformers) - HuggingFace's model framework
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning methods 