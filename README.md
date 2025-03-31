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

## Cost Analysis for Fine-Tuning Approaches

Understanding the costs associated with each fine-tuning method is crucial for planning and budgeting educational AI projects.

### DSPy Cost Analysis

DSPy leverages existing Large Language Models (LLMs) via API calls. Costs depend on the chosen LLM (e.g., OpenAI models), the number of calls during optimization and inference, and token usage.

**Key Cost Components:**

1.  **Optimization Phase:** Using DSPy optimizers (`Teleprompter`) involves multiple API calls per example in your dataset to find the best prompts. This is a development-time cost.
2.  **Inference Phase:** Each execution of the optimized DSPy program makes one or more API calls as defined by your modules.

**Estimating Costs with OpenAI Models (Example Pricing - Check current rates):**

*Pricing below is illustrative and subject to change.* 

| Model              | Input Token Price (per 1K tokens) | Output Token Price (per 1K tokens) |
|--------------------|-----------------------------------|------------------------------------|
| GPT-3.5-Turbo      | ~$0.0005 - $0.0015                | ~$0.0015 - $0.0045                 |
| GPT-4              | ~$0.03                            | ~$0.06                             |
| GPT-4-Turbo        | ~$0.01                            | ~$0.03                             |

**Table 1: Estimated DSPy Cost During Optimization (Per Example in Dataset)**

*Assumes 10 trial calls per example, avg. 1K input / 0.5K output tokens per trial.*

| Base LLM         | Est. Cost per Example |
|------------------|-----------------------|
| GPT-3.5-Turbo    | ~$0.01 - $0.04        |
| GPT-4-Turbo      | ~$0.25                |
| GPT-4            | ~$0.60                |

**Table 2: Estimated DSPy Cost During Inference (Per Program Execution)**

*Assumes 2 chained API calls, avg. 1K input / 0.5K output tokens per call.*

| Base LLM         | Estimated Cost per Execution |
|------------------|------------------------------|
| GPT-3.5-Turbo    | ~$0.0025 - $0.0075           |
| GPT-4-Turbo      | ~$0.05                       |
| GPT-4            | ~$0.12                       |

**DSPy Dataset Size Influence:** Larger datasets increase *total optimization cost* proportionally.

**(Note: Detailed cost analyses for OpenAI Fine-tuning and LoRA should also be added here if desired, following a similar structure.)**

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

# Educational AI Implementation Guides

This repository provides comprehensive guides for implementing educational AI systems using three different approaches to fine-tuning Large Language Models (LLMs). Each method has its unique advantages and is suited for different educational use cases.

## Method Comparison

| Feature | DSPy | OpenAI Fine-tuning | LoRA |
|---------|------|-------------------|------|
| **Approach** | Prompt optimization without model changes | Cloud-based full model fine-tuning | Local parameter-efficient fine-tuning |
| **Data Required** | 10-50 examples | 50-100+ examples | 100+ examples |
| **Cost Structure** | Pay per API call | Training + API calls | One-time hardware cost |
| **Control** | Limited to prompt engineering | API-level control | Complete model control |
| **Privacy** | Data sent to API | Data sent to API | Data stays local |
| **Setup Complexity** | Low | Medium | High |
| **Best For** | Quick prototyping, limited data | Production chatbots | Data privacy, full control |

## Detailed Method Overview

### 1. DSPy Prompt Optimization
- **How it Works**: Optimizes prompts without modifying model weights
- **Key Benefits**:
  - Minimal data requirements (10-50 examples)
  - Quick iteration and prototyping
  - No model training needed
- **Cost Efficiency**:
  - Development: $0.50-$10 per session
  - Production: $0.001-$0.002 per interaction
  - Monthly (100 students): $30-$300
- **Example Format**:
```json
{
  "dialogue": [
    {"role": "teacher", "content": "What happens when ice melts?"},
    {"role": "student", "content": "It turns into water! We did an experiment in class."},
    {"role": "teacher", "content": "Why do you think it melted?"},
    {"role": "student", "content": "Because the sun is hot!"}
  ]
}
```

### 2. OpenAI Fine-tuning
- **How it Works**: Updates model weights through cloud API
- **Key Benefits**:
  - Consistent behavior
  - Scalable cloud infrastructure
  - Professional support
- **Cost Breakdown**:
  - Training: $0.008 per 1K tokens
  - Usage: $0.003-$0.006 per 1K tokens
  - Monthly (100 students): $33-$165
- **Example Format**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a second-grade student."},
    {"role": "user", "content": "What is the water cycle?"},
    {"role": "assistant", "content": "Water goes up to the sky and makes clouds."}
  ]
}
```

### 3. LoRA (Low-Rank Adaptation)
- **How it Works**: Adapts specific model parameters locally
- **Key Benefits**:
  - Complete control over model
  - Data privacy
  - One-time hardware cost
- **Cost Structure**:
  - Hardware: $1,500-$5,000 (one-time)
  - Cloud Alternative: $14-$122 monthly
  - Training: $1-$5 per run
- **Example Format**:
```json
{
  "question": "How do plants grow?",
  "answer": "Plants grow from seeds! First you put a seed in dirt and water it."
}
```

## Implementation Requirements

### DSPy Setup
- Python 3.8+
- OpenAI API key
- Basic prompt engineering knowledge
- Minimal computational resources

### OpenAI Setup
- OpenAI API key
- Training dataset (50+ examples)
- Python environment
- Basic API knowledge

### LoRA Setup
- GPU with 8GB+ VRAM
- Python 3.8+
- PyTorch with CUDA
- Machine learning expertise

## Choosing the Right Method

1. **Choose DSPy if you**:
   - Have limited training data
   - Need quick prototypes
   - Want minimal setup
   - Prefer pay-per-use pricing

2. **Choose OpenAI if you**:
   - Need production-ready solutions
   - Want managed infrastructure
   - Have moderate data (50+ examples)
   - Prefer cloud-based solutions

3. **Choose LoRA if you**:
   - Need complete data privacy
   - Want full model control
   - Have significant data (100+ examples)
   - Can manage local infrastructure

## Getting Started

Each method has its own detailed guide:

- [DSPy Guide](docs/dspy_guide.md) - For prompt optimization without model changes
- [OpenAI Guide](docs/openai_guide.md) - For cloud-based fine-tuning
- [LoRA Guide](docs/lora_guide.md) - For local model adaptation

## Cost Considerations

Each guide includes detailed cost analysis:
- Development and training costs
- Production usage estimates
- Scaling considerations
- Cost optimization strategies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Contact

For questions or suggestions, please open an issue in the repository.

# Fine-tuning LLMs for Educational Applications

A comprehensive guide to implementing and comparing three different approaches for fine-tuning Large Language Models (LLMs) for educational applications, specifically focused on creating a second-grade student chatbot.

## Table of Contents
- [Overview](#overview)
- [Method Comparison](#method-comparison)
- [Dataset Requirements](#dataset-requirements)
- [Cost Analysis](#cost-analysis)
- [Implementation Guide](#implementation-guide)
- [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
- [Resources](#resources)

## Overview

Each fine-tuning method offers unique advantages for educational AI applications:

### 1. DSPy (Prompt Optimization)
- **Approach**: Optimizes prompts without modifying model weights
- **Use Case**: Quick prototyping, limited data scenarios
- **Key Benefit**: Minimal data requirements, rapid iteration
- **Limitation**: Less consistent than full fine-tuning

### 2. OpenAI Fine-tuning
- **Approach**: Updates model weights through cloud API
- **Use Case**: Production-ready educational chatbots
- **Key Benefit**: Consistent behavior, managed infrastructure
- **Limitation**: Requires more training data

### 3. LoRA (Low-Rank Adaptation)
- **Approach**: Efficient parameter updates on local hardware
- **Use Case**: Complete control, data privacy requirements
- **Key Benefit**: Full control over model and data
- **Limitation**: Requires technical expertise

## Method Comparison

| Feature | DSPy | OpenAI | LoRA |
|---------|------|--------|------|
| **Data Required** | 10-50 examples | 50-100+ examples | 100+ examples |
| **Setup Complexity** | Low | Medium | High |
| **Infrastructure** | Cloud API | Cloud API | Local GPU |
| **Cost Structure** | Pay per call | Training + API | Hardware + Power |
| **Control Level** | Limited | API-level | Complete |
| **Privacy** | API-dependent | API-dependent | Full control |
| **Update Speed** | Immediate | Hours | Minutes |
| **Scaling Cost** | Linear | Sub-linear | Fixed |

## Dataset Requirements

### 1. DSPy Dataset
```json
{
  "dialogue": [
    {
      "role": "teacher",
      "content": "What happens when ice melts?",
      "context": "Second-grade science lesson about states of matter"
    },
    {
      "role": "student",
      "content": "It turns into water! We did an experiment in class.",
      "knowledge_level": "basic",
      "subject": "science"
    },
    {
      "role": "teacher",
      "content": "Why do you think it melted?",
      "teaching_strategy": "inquiry-based"
    },
    {
      "role": "student",
      "content": "Because the sun is hot!",
      "reasoning_level": "concrete"
    }
  ],
  "metadata": {
    "grade_level": "2nd",
    "subject": "science",
    "topic": "states of matter",
    "vocabulary_level": "basic"
  }
}
```

### 2. OpenAI Dataset
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a second-grade student learning about the water cycle. You understand basic concepts and speak in simple terms. You're curious and enthusiastic about learning."
    },
    {
      "role": "user",
      "content": "What is the water cycle?"
    },
    {
      "role": "assistant",
      "content": "Water goes up to the sky and makes clouds! Then when there's lots of water in the clouds, it rains back down. I learned about this in class - it's like a big circle that water follows.",
      "metadata": {
        "confidence": 0.9,
        "knowledge_demonstrated": ["evaporation", "precipitation"],
        "vocabulary_level": "grade_2"
      }
    }
  ],
  "parameters": {
    "temperature": 0.7,
    "max_tokens": 150,
    "frequency_penalty": 0.5
  }
}
```

### 3. LoRA Dataset
```json
{
  "instruction": "Respond as a second-grade student learning about plant growth.",
  "input": {
    "question": "How do plants grow?",
    "context": "Science class discussion about plant life cycles",
    "grade_level": 2
  },
  "output": {
    "answer": "Plants grow from seeds! First you put a seed in dirt and water it. The seed gets bigger and bigger, and then a tiny plant comes out. It needs sun and water to grow tall. We did this experiment in class with bean seeds!",
    "metadata": {
      "concepts_covered": ["seeds", "growth", "plant needs"],
      "vocabulary_level": "grade_appropriate",
      "scientific_accuracy": 0.9
    }
  }
}
```

## Cost Analysis

### 1. DSPy Costs
- **Development**:
  - Prompt optimization: $0.002 per 1K tokens
  - Testing sessions: $0.50-$10 each
  - Monthly (100 students): $30-$300

### 2. OpenAI Costs
- **Training**:
  - Base cost: $0.008 per 1K tokens
  - Typical dataset (1000 examples): ~$4.00
  - Production deployment: $0.003-$0.006 per 1K tokens
  - Monthly (100 students): $33-$165

### 3. LoRA Costs
- **Hardware**:
  - GPU (8GB+ VRAM): $500-$2000
  - Training time: 2-8 hours per model
  - Power consumption: $1-$5 per training run
  - Cloud alternatives: $14-$122 monthly

## Implementation Guide

### 1. DSPy Implementation
```python
import dspy
from dspy.teleprompt import BootstrapFewShot

class StudentResponse(dspy.Signature):
    """Generate age-appropriate student responses"""
    context = dspy.InputField()
    question = dspy.InputField()
    response = dspy.OutputField()

class EducationalBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.student = StudentResponse()
        
    def forward(self, context, question):
        return self.student(context=context, question=question)

# Usage
bot = EducationalBot()
response = bot("We're learning about plants", "How do seeds grow?")
```

### 2. OpenAI Implementation
```python
import openai

def prepare_training_data(dialogues):
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a second-grade student."},
                {"role": "user", "content": d["question"]},
                {"role": "assistant", "content": d["answer"]}
            ]
        }
        for d in dialogues
    ]

# Create fine-tuning job
response = openai.FineTuningJob.create(
    training_file="file_id",
    model="gpt-3.5-turbo",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": 4
    }
)
```

### 3. LoRA Implementation
```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def setup_lora_model(base_model_id):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    
    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)
    return peft_model
```

## Evaluation Metrics

### 1. Response Quality
- Age-appropriate vocabulary
- Concept accuracy
- Natural conversation flow
- Learning progression

### 2. Technical Metrics
- Response latency
- Token efficiency
- Cost per interaction
- Error rates

### 3. Educational Impact
- Knowledge retention
- Engagement levels
- Learning outcomes
- Teacher feedback

## Examples

### 1. Science Dialogue
```json
{
  "topic": "States of Matter",
  "grade": 2,
  "dialogue": [
    {"role": "teacher", "content": "What happens to chocolate when it gets hot?"},
    {"role": "student", "content": "It melts and gets all gooey! Like when I leave my chocolate bar in the sun."},
    {"role": "teacher", "content": "What do you think happens when it cools down?"},
    {"role": "student", "content": "It gets hard again! We can make chocolate shapes that way."}
  ]
}
```

### 2. Math Dialogue
```json
{
  "topic": "Basic Addition",
  "grade": 2,
  "dialogue": [
    {"role": "teacher", "content": "If you have 3 apples and get 2 more, how many do you have?"},
    {"role": "student", "content": "Let me count! 3... then 4, 5. I have 5 apples now!"},
    {"role": "teacher", "content": "How did you figure that out?"},
    {"role": "student", "content": "I counted up from 3, just like we learned in class!"}
  ]
}
```

## Resources

### Documentation
- [DSPy Documentation](https://dspy.ai/)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [HuggingFace LoRA Guide](https://huggingface.co/docs/peft/conceptual_guides/lora)

### Tools and Libraries
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [OpenAI Python Library](https://github.com/openai/openai-python)
- [PEFT Library for LoRA](https://github.com/huggingface/peft)

### Additional Resources
- [Educational AI Best Practices](https://example.com/edu-ai-best-practices)
- [Cost Optimization Strategies](https://example.com/ai-cost-optimization)
- [Dataset Preparation Guide](https://example.com/dataset-guide)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Contact

For questions or suggestions, please open an issue in the repository.

