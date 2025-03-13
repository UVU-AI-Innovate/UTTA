# Environment Setup Tutorial

This tutorial will guide you through setting up your development environment for the LLM Fine-Tuning Toolkit. We'll cover all the necessary steps to get your system ready for any of the fine-tuning methods.

## Prerequisites

Before starting, ensure you have:
- Python 3.8+ installed
- Conda or Miniconda installed
- Git installed
- (Optional) CUDA-capable GPU for HuggingFace method

## Step 1: Create a Virtual Environment

```bash
# Create a new conda environment
conda create -n utta python=3.10
conda activate utta

# Verify Python installation
python --version  # Should show Python 3.10.x
```

## Step 2: Install Base Dependencies

```bash
# Clone the repository
git clone https://github.com/your-org/llm-finetuning-toolkit
cd llm-finetuning-toolkit

# Install base requirements
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import openai; print(f'OpenAI: {openai.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Step 3: Method-Specific Setup

### DSPy Setup
```bash
# Install DSPy requirements
pip install dspy-ai
pip install jupyter notebook  # for tutorials

# Verify installation
python -c "import dspy; print(f'DSPy: {dspy.__version__}')"
```

### OpenAI Setup
```bash
# Install OpenAI requirements
pip install openai tiktoken

# Set up API key
export OPENAI_API_KEY="your-api-key-here"

# Verify setup
python -c """
import openai
try:
    openai.OpenAI().models.list()
    print('OpenAI setup successful!')
except Exception as e:
    print(f'Error: {e}')
"""
```

### HuggingFace Setup
```bash
# Install HuggingFace requirements
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft evaluate

# Verify CUDA setup
python -c """
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"""
```

## Step 4: IDE Setup (Optional)

### VSCode Setup
1. Install VSCode
2. Install Python extension
3. Configure settings:
   ```json
   {
       "python.defaultInterpreterPath": "path/to/conda/envs/utta/python",
       "python.formatting.provider": "black",
       "python.linting.enabled": true,
       "python.linting.pylintEnabled": true
   }
   ```

### Jupyter Setup
```bash
# Install Jupyter
pip install jupyter notebook

# Install helpful extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Start Jupyter
jupyter notebook
```

## Step 5: Testing Your Setup

Create `test_setup.py`:
```python
import sys
import torch
import openai
import transformers
import datasets
import peft
import dspy

def check_versions():
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Datasets: {datasets.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"DSPy: {dspy.__version__}")

def check_gpu():
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("No GPU available")

def check_openai():
    try:
        client = openai.OpenAI()
        models = client.models.list()
        print("OpenAI API connection successful")
    except Exception as e:
        print(f"OpenAI API error: {e}")

if __name__ == "__main__":
    print("=== Environment Check ===")
    check_versions()
    print("\n=== GPU Check ===")
    check_gpu()
    print("\n=== OpenAI Check ===")
    check_openai()
```

Run the test:
```bash
python test_setup.py
```

## Troubleshooting

### CUDA Issues
```bash
# If CUDA not found, try:
conda install cudatoolkit
# or
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
```

### OpenAI Issues
```bash
# Check API key
echo $OPENAI_API_KEY

# Set API key if needed
export OPENAI_API_KEY="your-key-here"
```

### Package Conflicts
```bash
# Create fresh environment
conda deactivate
conda remove -n utta --all
conda create -n utta python=3.10
conda activate utta

# Install packages one by one
pip install torch
pip install transformers
# etc.
```

## System Requirements

### Minimum Requirements
- CPU: Any modern processor
- RAM: 8GB
- Storage: 5GB
- Python: 3.8+

### Recommended Requirements
- CPU: 4+ cores
- RAM: 16GB
- GPU: 8GB+ VRAM (for HuggingFace)
- Storage: 20GB
- Python: 3.10+

## Next Steps

1. Verify your setup with `test_setup.py`
2. Choose your fine-tuning method:
   - [DSPy Tutorial](DSPy-Tutorial.md)
   - [OpenAI Tutorial](OpenAI-Tutorial.md)
   - [HuggingFace Tutorial](HuggingFace-Tutorial.md)
3. Prepare your development environment:
   - Set up version control
   - Configure your IDE
   - Organize your project structure

## Additional Resources

- [Conda Documentation](https://docs.conda.io/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [VSCode Python Setup](https://code.visualstudio.com/docs/python/python-tutorial) 