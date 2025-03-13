# Getting Started with UTTA

Welcome to UTTA (Utah Teacher Training Assistant)! This guide will help you get started with using and contributing to the project.

## Prerequisites

Before you begin, ensure you have:
- Python 3.10 or higher installed
- Conda package manager installed
- Git installed
- Basic understanding of LLMs and Python programming

## Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/UVU-AI-Innovate/UTTA.git
   cd UTTA
   ```

2. **Set Up Environment**
   ```bash
   conda create -n utta python=3.10
   conda activate utta
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```python
   # Test your installation
   python -c "from utta.core import TeachingAssistant; print('Installation successful!')"
   ```

## Quick Start Example

Here's a simple example to get you started:

```python
from utta.core import TeachingAssistant

# Initialize the teaching assistant
assistant = TeachingAssistant()

# Ask a question
response = assistant.answer_question(
    "Explain the concept of photosynthesis in simple terms"
)

print(response)
```

## Next Steps

1. Set up your development environment properly by following the [Environment Setup](Environment-Setup) guide
2. Learn how to prepare your educational datasets using the [Dataset Preparation](Dataset-Preparation) guide
3. Choose a framework to start with:
   - [DSPy Tutorial](DSPy-Tutorial) for optimized prompting
   - [OpenAI Tutorial](OpenAI-Tutorial) for using OpenAI models
   - [HuggingFace Tutorial](HuggingFace-Tutorial) for open-source models

## Common Issues

1. **Environment Issues**
   - Make sure you're using Python 3.10
   - Ensure all dependencies are installed
   - Check that your conda environment is activated

2. **Model Access**
   - Set up necessary API keys
   - Verify model permissions
   - Check network connectivity

3. **Resource Usage**
   - Monitor GPU memory usage
   - Check API rate limits
   - Manage batch sizes appropriately

## Getting Help

If you run into issues:
1. Check the documentation
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Join our community discussions

## Contributing

We welcome contributions! See our [Contributing Guidelines](https://github.com/UVU-AI-Innovate/UTTA/blob/main/CONTRIBUTING.md) for details on:
- Code style
- Pull request process
- Development workflow
- Testing requirements 