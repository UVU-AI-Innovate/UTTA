# UTTA (Utah Teacher Training Assistant)

> A sophisticated chatbot framework designed for teacher training, leveraging advanced LLM capabilities with DSPy integration, knowledge base management, and automated evaluation metrics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-alpha-orange)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31012/)

## Documentation

📚 **Project Website**: [https://uvu-ai-innovate.github.io/UTTA/](https://uvu-ai-innovate.github.io/UTTA/)

Visit our GitHub Pages site for an overview of the project, fine-tuning approaches, and implementation details.

## Prerequisites

- Python 3.10
- Conda package manager
- CUDA-capable GPU (for full functionality)
- OpenAI API key (already configured in `.env`)

## Installation

### Option 1: Automated Setup (Recommended)

The easiest way to get started is to use our setup script:

```bash
# Clone the repository
git clone https://github.com/majidmemari/UTTA.git
cd UTTA

# Make the script executable
chmod +x run_webapp.sh

# Run the setup script
./run_webapp.sh
```

The script will:
- Create and configure the conda environment
- Install all required dependencies
- Set up necessary directories
- Use existing `.env` configuration
- Run system diagnostics
- Optionally start the web application

### Option 2: Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/UVU-AI-Innovate/UTTA
cd UTTA
```

2. Create and activate the conda environment:
```bash
# Create environment from yml file
conda env create -f environment.yml

# Activate the environment
conda activate utta
```

3. Create required directories:
```bash
mkdir -p data/index data/cache data/uploads
```

4. Verify the installation:
```bash
# Run system diagnostics
python src/diagnostics.py
```

## Project Structure

```
UTTA/
├── data/               # Data storage
│   ├── cache/         # Cache for embeddings and API responses
│   ├── index/         # Document index storage
│   └── uploads/       # User-uploaded documents
├── src/               # Source code
│   ├── diagnostics.py # System diagnostics tool
│   ├── cli.py        # Command-line interface
│   ├── web_app.py    # Web application interface
│   ├── metrics/      # Response evaluation metrics
│   ├── fine_tuning/  # Model fine-tuning components
│   ├── knowledge_base/# Document indexing and retrieval
│   └── llm/          # LLM interface and DSPy handlers
├── examples/          # Example implementations (see Examples Directory section)
├── tests/            # Test directory
├── environment.yml   # Conda environment specification
├── run_webapp.sh    # Setup and launch script
├── .env             # Configuration file
└── LICENSE          # MIT License file
```

## Usage

### Web Interface

```bash
# Activate the environment (if not already active)
conda activate utta

# Start the web application
streamlit run src/web_app.py
```

Then open your browser and navigate to:
```
http://localhost:8501
```

### CLI Interface

```bash
# Activate the environment (if not already active)
conda activate utta

# Start the command-line interface
python src/cli.py
```

## Features

- Interactive educational conversations
- Adaptive learning paths
- Multi-modal content support
- Progress tracking
- Customizable teaching styles
- Knowledge base integration
- RAG (Retrieval Augmented Generation)
- Fine-tuning capabilities
- Evaluation metrics
- Web and CLI interfaces

## Environment Configuration

The `.env` file contains the following configuration:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key

# Model Configuration
DEFAULT_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7
MAX_TOKENS=1000

# Storage Paths
CACHE_DIR=data/cache
INDEX_DIR=data/index
MODEL_DIR=data/models

# Logging
LOG_LEVEL=INFO
LOG_FILE=data/app.log

# Web Interface
STREAMLIT_PORT=8501
STREAMLIT_THEME=light
```

### Environment Details

The project uses a conda environment with the following key components:

- **Python Version**: 3.10
- **Deep Learning**:
  - PyTorch 2.2.0
  - CUDA Toolkit 11.8
  - torchvision

- **Core Dependencies**:
  - streamlit==1.32.0
  - openai==0.28.0 (specific version required for DSPy)
  - dspy-ai==2.0.4
  - sentence-transformers==2.2.2
  - faiss-gpu>=1.7.4
  - spacy>=3.7.0

For a complete list of dependencies, see `environment.yml`.

## Examples Directory

The `examples/` directory contains educational examples and tutorials to help you understand and work with different aspects of the UTTA project:

### Example Categories

#### DSPy Examples (`examples/dspy/`)
- `optimize_educational_qa.py`: Demonstrates how to optimize educational Q&A using DSPy
- `custom_teleprompter.py`: Shows how to create custom teleprompters for educational content

#### OpenAI Examples (`examples/openai/`)
- `finetune_educational_qa.py`: Example of fine-tuning OpenAI models for educational Q&A
- `chat_completion_examples.py`: Various chat completion examples with different parameters

#### Hugging Face Examples (`examples/huggingface/`)
- `finetune_educational_qa.py`: Example of fine-tuning Hugging Face models
- `model_inference.py`: Demonstrates model inference with different Hugging Face models

### Learning Focus

Each example includes detailed comments and documentation to help you understand:
- How to set up the environment
- Required dependencies
- Usage examples
- Best practices
- Common pitfalls to avoid

See `examples/README.md` for more details on the different approaches to fine-tuning and their use cases.

## Troubleshooting

### Common Issues

1. **Environment Setup**
   - Ensure conda is installed and initialized
   - Check Python version compatibility
   - Verify all dependencies are installed

2. **API Key Issues**
   - Verify OPENAI_API_KEY is set in `.env`
   - Check API key validity
   - Ensure proper permissions

3. **Directory Structure**
   - Verify all required directories exist
   - Check directory permissions
   - Ensure proper path configuration

4. **OpenAI API errors**
   - Check your API key in `.env`
   - Verify connectivity

5. **CUDA errors**
   - Ensure CUDA toolkit is installed
   - Check GPU compatibility

6. **Import errors**
   - Verify conda environment is active
   - Use `fix_openai_dep.py` for OpenAI compatibility issues

### Diagnostics

Run the diagnostics tool to identify issues:
```bash
python src/diagnostics.py
```

### Getting Help

- Check the documentation
- Review example code
- Open an issue for bugs
- Contact maintainers for support

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Copyright (c) 2025 Utah Teacher Training Assistant (UTTA)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and diagnostics
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- DSPy team for the foundational LLM framework
- Contributors and testers

# 🧠 LLM Fine-Tuning Examples for Teacher-Student Dialogues

This directory contains examples of three different approaches to fine-tuning large language models (LLMs) for realistic teacher-student dialogue interactions.

## 📚 Quick Start Guide

For the best learning experience, follow these steps:

1. **Start with the overview**: Run the simple example to understand the concepts
   ```bash
   python simple_example.py
   ```
   This script works without any special dependencies and gives you a clear understanding of all three approaches.

2. **Explore the detailed examples**: The main example files provide in-depth implementations
   ```bash
   # View the code to understand the implementation details
   cat dspy_example.py
   cat openai_finetune.py
   cat huggingface_lora.py
   ```

3. **Review the assignment**: Check `assignment.md` for detailed instructions on how to implement these approaches in your own projects.

## 🔍 Three Fine-Tuning Approaches

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (10-50 dialogues) | Medium (50-100+ dialogues) | Large (100-1000+ dialogues) |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU (16GB+ VRAM) |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |
| Time to results | Immediate | Hours | Hours (with GPU) |

## 📝 Environment Setup

To run the complete examples (beyond the simple overview), you'll need:

### DSPy Example
```bash
pip install dspy-ai==0.11.0  # The example requires a specific version with Metric support
```

### OpenAI Example
```bash
pip install openai==0.28.0  # The example is compatible with this version
```

### HuggingFace Example
```bash
pip install transformers==4.30.0 peft==0.4.0 bitsandbytes==0.40.0
# A GPU with 8GB+ VRAM is recommended
```

## 📁 Example Files

### Core Example Files
- `simple_example.py` - Works with minimal dependencies, provides overview of all approaches
- `dspy_example.py` - DSPy prompt optimization for teacher-student dialogue
- `openai_finetune.py` - OpenAI fine-tuning for educational dialogues
- `huggingface_lora.py` - HuggingFace LoRA fine-tuning with local models

### Data Files
- `teacher_student_dialogues.jsonl` - Sample dialogue data for DSPy
- `small_edu_qa.jsonl` - Sample Q&A pairs for simple examples
- `openai_edu_qa_training.jsonl` - Sample training data for OpenAI fine-tuning

### Utility Scripts
- `run_all_examples.sh` - Script to run examples with helpful guidance

## 🔍 Detailed Approach Explanations

### DSPy: Prompt Optimization

DSPy optimizes the prompts used with LLMs without changing the model weights:
- Requires minimal training data (10-50 examples)
- Gives immediate results
- Works with any LLM through APIs
- Uses Chain-of-Thought prompting for improved pedagogical responses

Key implementation features:
```python
# Define the teacher response signature 
class TeacherResponse(dspy.Signature):
    student_question = dspy.InputField()
    teacher_response = dspy.OutputField()

# Use Chain-of-Thought for pedagogical reasoning
generator = dspy.ChainOfThought(TeacherResponse)
```

### OpenAI: Cloud Fine-Tuning

OpenAI fine-tuning updates model weights through their cloud service:
- Requires more training data than DSPy (50-100+ examples)
- Creates a specialized model variant
- Handles multi-turn conversations well
- Needs no local GPU infrastructure

Key implementation features:
```python
# Format data for fine-tuning
formatted_data = [
    {"messages": [
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": "How does photosynthesis work?"},
        {"role": "assistant", "content": "Let me explain photosynthesis..."}
    ]}
]

# Create fine-tuning job
job = openai.FineTuningJob.create(
    training_file=file_id,
    model="gpt-3.5-turbo"
)
```

### HuggingFace: Local LoRA Fine-Tuning

HuggingFace's LoRA approach:
- Updates model weights locally with Low-Rank Adaptation
- Keeps all data private on your hardware
- Requires GPU for efficient training
- Gives complete control over the training process

Key implementation features:
```python
# Configure LoRA parameters
lora_config = LoraConfig(
    r=8,              # Rank of adapter matrices
    lora_alpha=16,    # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

# Apply LoRA to base model
model = get_peft_model(base_model, lora_config)
```

## 🔧 Which Approach Is Right For You?

- **DSPy**: Best for quick experiments, small datasets, immediate results
- **OpenAI**: Best for production deployment, balanced control/convenience
- **HuggingFace**: Best for complete control, data privacy, long-term usage

