# Fine-Tuning Methods in UTTA

This page provides a comprehensive overview of the various fine-tuning approaches supported by UTTA for customizing language models to educational tasks.

## Introduction

UTTA supports multiple fine-tuning methods, each with different characteristics, requirements, and use cases. This document explains each method and helps you choose the right approach for your specific educational needs.

## Supported Fine-Tuning Methods

UTTA currently supports three primary fine-tuning approaches:

1. **DSPy Optimization** - Prompt optimization without changing model weights
2. **OpenAI Fine-Tuning** - Traditional fine-tuning of OpenAI models
3. **HuggingFace Fine-Tuning** - Fine-tuning open-source models with various techniques

## DSPy Optimization

DSPy optimization is a technique that improves model performance without changing the underlying model weights. It's especially useful when full model fine-tuning is cost-prohibitive or unnecessary.

### How It Works

DSPy optimization uses techniques like prompt engineering and few-shot learning to optimize the interaction with language models. The framework analyzes examples and automatically discovers better prompting strategies.

### Optimization Methods

UTTA supports the following DSPy optimization methods:

- **BootstrapFewShot** - Uses a bootstrapping approach to discover effective few-shot examples
- **BootstrapFewShotWithRandomSearch** - Combines bootstrapping with random search for prompt optimization

### When to Use DSPy

DSPy optimization is ideal when:
- You have limited computational resources
- You need quick improvements without a full fine-tuning process
- You're working with a small number of examples
- You want to maintain the general capabilities of the model while specializing for specific tasks

### Example Usage

```python
from utta.fine_tuning.dspy import DSPyOptimizer, EducationalQAModel

# Create model and optimizer
model = EducationalQAModel()
optimizer = DSPyOptimizer(model)

# Optimize using examples
optimizer.optimize(
    train_data=[{"question": "What is photosynthesis?", "answer": "Photosynthesis is..."}],
    num_iterations=3,
    metric="answer_quality"
)

# Use the optimized model
result = optimizer.optimized_module("What are plant cells?")
```

## OpenAI Fine-Tuning

OpenAI fine-tuning allows you to customize OpenAI's models (like GPT-3.5) for specific educational tasks through API-based fine-tuning.

### How It Works

OpenAI fine-tuning updates the model weights based on your provided examples. This creates a specialized version of the model that's optimized for your specific use case while retaining its general capabilities.

### Features

- Full model weight adjustment
- API-based (no need for local GPU)
- Managed by OpenAI's infrastructure
- Supports various hyperparameter configurations

### When to Use OpenAI Fine-Tuning

OpenAI fine-tuning is ideal when:
- You need deeper customization than prompt engineering can provide
- You have a significant number of examples (typically 100+ for best results)
- You want to create a specialized model variant for production use
- You prefer API-based fine-tuning without managing infrastructure

### Example Usage

```python
from utta.fine_tuning.openai import OpenAIFineTuner

# Initialize the tuner
tuner = OpenAIFineTuner(
    base_model="gpt-3.5-turbo",
    suffix="education-specialist",
    n_epochs=3
)

# Prepare and upload data
tuner.prepare_data("examples/raw_data.json", "prepared_data.jsonl")

# Start fine-tuning
job_id = tuner.start_finetuning("prepared_data.jsonl")
```

## HuggingFace Fine-Tuning

HuggingFace fine-tuning gives you full control over training open-source models from the HuggingFace ecosystem, with support for various parameter-efficient fine-tuning techniques.

### How It Works

HuggingFace fine-tuning in UTTA provides a simplified interface to train models locally or on your own infrastructure. It supports various advanced techniques that make fine-tuning more efficient and accessible.

### Supported Techniques

UTTA supports several parameter-efficient fine-tuning methods through HuggingFace:

- **LoRA (Low-Rank Adaptation)** - Efficient fine-tuning by adding low-rank matrices to key layers
- **Quantization** - Supports 8-bit and 4-bit quantization for reduced memory usage
- **Full Fine-Tuning** - Traditional fine-tuning of all model parameters

### When to Use HuggingFace Fine-Tuning

HuggingFace fine-tuning is ideal when:
- You need full control over the fine-tuning process
- You want to use open-source models
- You have access to appropriate computing resources (GPUs)
- You need advanced techniques like LoRA or quantization
- You want to avoid API costs for inference

### Example Usage

```python
from utta.fine_tuning.huggingface import HuggingFaceFineTuner

# Initialize the tuner
tuner = HuggingFaceFineTuner(
    model_name="microsoft/phi-2",
    output_dir="models/fine_tuned"
)

# Prepare data
dataset = tuner.prepare_data(conversations)

# Fine-tune with LoRA
tuner.fine_tune(
    dataset=dataset,
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    use_lora=True,
    lora_r=16,
    lora_alpha=32
)
```

## Choosing the Right Method

When deciding which fine-tuning method to use, consider these factors:

| Factor | DSPy Optimization | OpenAI Fine-Tuning | HuggingFace Fine-Tuning |
|--------|-------------------|-------------------|-------------------------|
| **Computational Requirements** | Low | None (API-based) | High (requires GPU) |
| **Training Data Needed** | Small (10+ examples) | Medium (50+ examples) | Medium-Large (100+ examples) |
| **Cost** | API usage only | API usage + fine-tuning costs | Compute costs only |
| **Customization Level** | Moderate | High | Very High |
| **Inference Speed** | API-dependent | API-dependent | Local (can be fast) |
| **Recommended Use Case** | Quick prototyping, Simple specialization | Production deployment, Specialized task | Full control, Open-source needs |

## For More Information

For detailed tutorials on each method, see:
- [DSPy Tutorial](DSPy-Tutorial)
- [OpenAI Tutorial](OpenAI-Tutorial)
- [HuggingFace Tutorial](HuggingFace-Tutorial)

For CLI usage of these methods, see:
- [Fine-Tuning CLI](Fine-Tuning-CLI) 