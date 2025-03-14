# Getting Started with UTTA

Welcome to UTTA (Universal Teaching and Training Assistant)! This guide will help you get started with using and contributing to the project.

## Prerequisites

Before you begin, ensure you have the following:

* Python 3.10 or higher
* Conda (recommended for environment management)
* Git
* Basic understanding of LLMs (Large Language Models) and Python

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/UVU-AI-Innovate/UTTA.git
   cd UTTA
   ```

2. **Set Up the Environment**
   ```bash
   conda create -n utta python=3.10
   conda activate utta
   pip install -r requirements.txt
   ```

3. **Configure API Keys** (if using external models)
   
   For OpenAI models:
   ```bash
   # Add to your environment or create a .env file
   export OPENAI_API_KEY="your-key-here"
   ```

   For HuggingFace models:
   ```bash
   # Add to your environment or create a .env file
   export HUGGINGFACE_API_KEY="your-key-here"
   ```

4. **Verify Installation**
   ```bash
   python -c "from utta import TeachingAssistant; print('UTTA is installed correctly')"
   ```

## Quick Start Example

Here's a simple example to get you started:

```python
from utta import TeachingAssistant

# Initialize the teaching assistant
assistant = TeachingAssistant(model="openai/gpt-3.5-turbo")

# Ask a question
response = assistant.ask("What is the difference between supervised and unsupervised learning?")
print(response)
```

## Next Steps

- Check out the [Environment Setup](Environment-Setup) guide for more detailed configuration
- Learn about using [DSPy for optimization](DSPy-Tutorial)
- Explore our guides for [OpenAI](OpenAI-Tutorial) and [HuggingFace](HuggingFace-Tutorial) integration
- Learn how to [prepare datasets](Dataset-Preparation) for training and evaluation

## Common Issues

- **Environment Issues**: Make sure you've activated the conda environment with `conda activate utta`
- **Model Access**: Verify that your API keys are correctly set up
- **Resource Usage**: Be aware of token usage and rate limits when using external APIs

## Getting Help

If you encounter any issues:

1. Check the existing documentation in the wiki
2. Look for similar issues in the [GitHub Issues](https://github.com/UVU-AI-Innovate/UTTA/issues)
3. Create a new issue if you can't find a solution

## Contributing

We welcome contributions! Please see our [Contributing Guide](Contributing) for more information. 