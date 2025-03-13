# Environment Setup Guide

This guide provides detailed instructions for setting up your development environment for UTTA.

## System Requirements

### Hardware Requirements
- **CPU**: Modern multi-core processor
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 20GB+ free space
- **GPU**: Optional but recommended for HuggingFace models
  - NVIDIA GPU with 8GB+ VRAM
  - CUDA support

### Software Requirements
- **Operating System**: Linux, macOS, or Windows
- **Python**: Version 3.10 or higher
- **Conda**: Latest version
- **Git**: Latest version

## Step-by-Step Setup

### 1. Python Installation
```bash
# Check Python version
python --version

# If needed, install Python 3.10 using your package manager
# For Ubuntu:
sudo apt update
sudo apt install python3.10
```

### 2. Conda Installation
```bash
# Download Miniconda (recommended)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Or use system package manager
# For Ubuntu:
sudo apt install conda
```

### 3. UTTA Environment Setup
```bash
# Clone the repository
git clone https://github.com/UVU-AI-Innovate/UTTA.git
cd UTTA

# Create and activate conda environment
conda create -n utta python=3.10
conda activate utta

# Install dependencies
pip install -r requirements.txt
```

### 4. API Keys Setup

#### OpenAI API
1. Create an account at [OpenAI](https://platform.openai.com/)
2. Generate an API key
3. Set the environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

#### HuggingFace Hub
1. Create an account at [HuggingFace](https://huggingface.co/)
2. Generate an access token
3. Login using the CLI:
   ```bash
   huggingface-cli login
   ```

### 5. GPU Setup (Optional)

For HuggingFace models, CUDA support is recommended:

1. **Install CUDA Toolkit**
   ```bash
   # Check NVIDIA driver
   nvidia-smi
   
   # Install CUDA (example for Ubuntu)
   sudo apt install nvidia-cuda-toolkit
   ```

2. **Install PyTorch with CUDA**
   ```bash
   # Install PyTorch with CUDA support
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

## Verification

Test your installation:

```python
# Test basic functionality
python -c "from utta.core import TeachingAssistant; print('Core module working!')"

# Test GPU support (if applicable)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Common Issues and Solutions

### 1. Dependency Conflicts
```bash
# If you encounter conflicts, try:
conda create -n utta python=3.10 --no-deps
conda activate utta
pip install -r requirements.txt --no-deps
```

### 2. CUDA Issues
- Ensure NVIDIA drivers are up to date
- Check CUDA version compatibility
- Verify GPU is recognized:
  ```bash
  nvidia-smi
  ```

### 3. API Authentication
- Double-check API keys
- Verify environment variables
- Test API access:
  ```python
  import openai
  openai.Model.list()  # Should return available models
  ```

## Development Tools

### Recommended IDE Setup
- VSCode with Python extension
- PyCharm Professional/Community
- Jupyter Lab/Notebook

### Useful Extensions
- Python
- Jupyter
- Git
- Docker
- YAML

## Next Steps

1. Follow the [Getting Started Guide](Getting-Started)
2. Prepare your datasets using [Dataset Preparation](Dataset-Preparation)
3. Choose a framework tutorial:
   - [DSPy Tutorial](DSPy-Tutorial)
   - [OpenAI Tutorial](OpenAI-Tutorial)
   - [HuggingFace Tutorial](HuggingFace-Tutorial) 