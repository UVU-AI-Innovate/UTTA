# Fine-Tuning CLI Tool

The LLM Chatbot Framework includes a comprehensive command-line interface (CLI) tool for managing fine-tuning jobs across all three fine-tuning approaches.

## Overview

The fine-tuning CLI tool (`tools/fine_tuning_cli.py`) provides a unified interface for all fine-tuning methods, offering:

- **Simplified Workflow**: Standard commands across all fine-tuning methods
- **Automation Support**: Easy integration with scripts and automation tools
- **Management Capabilities**: Tools for tracking, evaluating, and managing fine-tuned models
- **Accessibility**: Makes fine-tuning accessible to users without deep ML expertise

## Basic Usage

```bash
# View available commands
python tools/fine_tuning_cli.py --help

# View help for a specific command
python tools/fine_tuning_cli.py dspy --help
python tools/fine_tuning_cli.py openai --help
python tools/fine_tuning_cli.py huggingface --help
```

## DSPy Commands

The CLI provides commands to work with DSPy optimization:

```bash
# Optimize a DSPy module
python tools/fine_tuning_cli.py dspy optimize --module-name teacher_responder --input-file examples.json

# Save an optimized module
python tools/fine_tuning_cli.py dspy save --module-name teacher_responder --output-dir cache/optimized

# Load an optimized module
python tools/fine_tuning_cli.py dspy load --module-name teacher_responder --input-dir cache/optimized
```

### Command Options

#### `dspy optimize`
- `--module-name`: Name of the module to optimize (required)
- `--input-file`: Path to the input file with training examples (required)
- `--method`: Optimization method to use (default: bootstrap_few_shot)
- `--metric`: Evaluation metric (default: exact_match)
- `--num-examples`: Number of examples to use (default: all examples)

#### `dspy save`
- `--module-name`: Name of the module to save (required)
- `--output-dir`: Directory to save the optimized module (required)

#### `dspy load`
- `--module-name`: Name of the module to load (required)
- `--input-dir`: Directory containing the saved module (required)

## OpenAI Commands

Commands for working with OpenAI fine-tuning:

```bash
# Start a fine-tuning job
python tools/fine_tuning_cli.py openai start --training-file data.jsonl --model gpt-3.5-turbo

# List jobs
python tools/fine_tuning_cli.py openai list-jobs

# Check job status
python tools/fine_tuning_cli.py openai status --job-id ft-job-123456

# List fine-tuned models
python tools/fine_tuning_cli.py openai list-models

# Cancel a job
python tools/fine_tuning_cli.py openai cancel --job-id ft-job-123456
```

### Command Options

#### `openai start`
- `--training-file`: Path to the JSONL training file (required)
- `--model`: Base model to fine-tune (default: gpt-3.5-turbo)
- `--suffix`: Custom suffix for the fine-tuned model
- `--validation-file`: Optional validation file path
- `--batch-size`: Number of examples in each batch (default: auto)
- `--learning-rate-multiplier`: Learning rate multiplier (default: auto)
- `--epochs`: Number of epochs (default: auto)

#### `openai status`
- `--job-id`: ID of the fine-tuning job to check (required)

#### `openai cancel`
- `--job-id`: ID of the fine-tuning job to cancel (required)

## HuggingFace Commands

Commands for working with HuggingFace fine-tuning:

```bash
# Fine-tune a model
python tools/fine_tuning_cli.py huggingface fine-tune --model-name meta-llama/Llama-2-7b-chat-hf --training-file data.jsonl

# List fine-tuned models
python tools/fine_tuning_cli.py huggingface list-models

# Delete a model
python tools/fine_tuning_cli.py huggingface delete-model --model-dir cache/fine_tuned_models/model-name

# Export a model for integration
python tools/fine_tuning_cli.py huggingface export-model --model-dir cache/fine_tuned_models/model-name
```

### Command Options

#### `huggingface fine-tune`
- `--model-name`: Name of the base model to fine-tune (required)
- `--training-file`: Path to the training file (required)
- `--output-dir`: Directory to save the fine-tuned model (default: cache/fine_tuned_models/model-name)
- `--epochs`: Number of training epochs (default: 3)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--batch-size`: Training batch size (default: 4)
- `--load-in-4bit`: Whether to load the model in 4-bit precision (default: False)
- `--load-in-8bit`: Whether to load the model in 8-bit precision (default: False)
- `--lora-r`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha (default: 32)
- `--lora-dropout`: LoRA dropout (default: 0.05)

#### `huggingface list-models`
- `--models-dir`: Directory containing fine-tuned models (default: cache/fine_tuned_models)

#### `huggingface delete-model`
- `--model-dir`: Directory of the model to delete (required)

#### `huggingface export-model`
- `--model-dir`: Directory of the model to export (required)
- `--output-dir`: Directory to save the exported model (default: same as model-dir with _exported suffix)
- `--format`: Export format (default: gguf)

## Data Preparation

The CLI also includes utilities for preparing training data:

```bash
# Convert data format
python tools/fine_tuning_cli.py prepare convert --input input.json --output output.jsonl --format openai

# Validate data format
python tools/fine_tuning_cli.py prepare validate --input data.jsonl --format openai
```

### Command Options

#### `prepare convert`
- `--input`: Input file path (required)
- `--output`: Output file path (required)
- `--format`: Output format (required, one of: openai, dspy, huggingface)

#### `prepare validate`
- `--input`: Input file to validate (required)
- `--format`: Format to validate against (required, one of: openai, dspy, huggingface)

## Examples

### DSPy Optimization Example

```bash
# Create training data
python tools/fine_tuning_cli.py prepare convert --input classroom_examples.json --output classroom_dspy.json --format dspy

# Optimize a module
python tools/fine_tuning_cli.py dspy optimize --module-name teacher_responder --input-file classroom_dspy.json --method bootstrap_few_shot

# Save the optimized module
python tools/fine_tuning_cli.py dspy save --module-name teacher_responder --output-dir cache/optimized_modules
```

### OpenAI Fine-Tuning Example

```bash
# Convert training data to OpenAI format
python tools/fine_tuning_cli.py prepare convert --input education_examples.json --output education.jsonl --format openai

# Validate the data format
python tools/fine_tuning_cli.py prepare validate --input education.jsonl --format openai

# Start fine-tuning
python tools/fine_tuning_cli.py openai start --training-file education.jsonl --model gpt-3.5-turbo --suffix edu-assistant

# Check status
python tools/fine_tuning_cli.py openai status --job-id ft-job-123456
```

### HuggingFace Fine-Tuning Example

```bash
# Convert training data to HuggingFace format
python tools/fine_tuning_cli.py prepare convert --input education_examples.json --output education_hf.jsonl --format huggingface

# Fine-tune a model with 4-bit quantization
python tools/fine_tuning_cli.py huggingface fine-tune --model-name meta-llama/Llama-2-7b-chat-hf --training-file education_hf.jsonl --output-dir cache/llama-2-edu --load-in-4bit

# Export the model
python tools/fine_tuning_cli.py huggingface export-model --model-dir cache/llama-2-edu
```

## Related Pages

- [Fine-Tuning Overview](Fine-Tuning-Overview)
- [DSPy Optimization](DSPy-Optimization)
- [OpenAI Fine-Tuning](OpenAI-Fine-Tuning)
- [HuggingFace Fine-Tuning](HuggingFace-Fine-Tuning)
- [Fine-Tuning Best Practices](Fine-Tuning-Best-Practices)
- [Troubleshooting](Fine-Tuning-Troubleshooting) 