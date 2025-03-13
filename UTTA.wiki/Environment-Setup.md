# Environment Setup Guide

This guide provides detailed instructions for setting up your development environment for the UTTA project.

## System Requirements

* **Operating System**: Linux, macOS, or Windows
* **Python**: Version 3.10 or higher
* **RAM**: At least 8GB (16GB+ recommended for larger models)
* **Disk Space**: At least 2GB for installation and dependencies
* **GPU**: Optional but recommended for training and running larger models

## Python Environment Setup

We strongly recommend using Conda for managing your Python environment:

### Using Conda (Recommended)

1. **Install Conda**:
   * Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual)

2. **Create a new environment**:
   ```bash
   conda create -n utta python=3.10
   conda activate utta
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Using venv (Alternative)

1. **Create a virtual environment**:
   ```bash
   python -m venv utta-env
   ```

2. **Activate the environment**:
   * On Windows: `utta-env\Scripts\activate`
   * On macOS/Linux: `source utta-env/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## API Keys Configuration

### OpenAI API

1. Create an account at [OpenAI](https://platform.openai.com/)
2. Generate an API key from your account dashboard
3. Add the key to your environment:
   ```bash
   export OPENAI_API_KEY="your-openai-key"
   ```
   For Windows: `set OPENAI_API_KEY=your-openai-key`

### HuggingFace API

1. Create an account at [HuggingFace](https://huggingface.co/)
2. Generate an API token from your account settings
3. Add the token to your environment:
   ```bash
   export HUGGINGFACE_API_KEY="your-huggingface-key"
   ```
   For Windows: `set HUGGINGFACE_API_KEY=your-huggingface-key`

## Using .env Files (Recommended)

For easier management of API keys, create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-huggingface-key
# Add other environment variables as needed
```

Then load it using:

```python
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env
```

## GPU Support

If you have a compatible NVIDIA GPU, you can enable GPU acceleration:

1. **Install CUDA Toolkit**:
   * Download from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)

2. **Install PyTorch with CUDA support**:
   ```bash
   conda install pytorch cudatoolkit=11.7 -c pytorch
   ```

3. **Verify GPU detection**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Number of GPUs: {torch.cuda.device_count()}")
   ```

## Troubleshooting

### Common Issues

* **ImportError**: Make sure you've activated your environment and installed all dependencies
* **API Key Errors**: Verify your API keys are correctly set and accessible
* **CUDA Errors**: Ensure compatible versions of PyTorch and CUDA

### Environment Verification

Run this script to verify your environment:

```python
import sys
import torch
import os

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenAI API Key configured: {'OPENAI_API_KEY' in os.environ}")
print(f"HuggingFace API Key configured: {'HUGGINGFACE_API_KEY' in os.environ}")
```

For additional help, please check the [UTTA GitHub repository](https://github.com/UVU-AI-Innovate/UTTA/issues) or create a new issue. 