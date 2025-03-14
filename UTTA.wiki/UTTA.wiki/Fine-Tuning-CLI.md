# Fine-Tuning CLI Tool

This page documents how to use the command-line interface for fine-tuning language models in UTTA.

## Overview

The Fine-Tuning CLI tool provides a unified interface for managing fine-tuning jobs across different LLM frameworks. It supports:

- DSPy prompt optimization
- OpenAI fine-tuning
- HuggingFace model fine-tuning

## Requirements

- Python 3.10+
- UTTA environment set up (see [Environment Setup](Environment-Setup))
- API keys for LLM providers (OpenAI, HuggingFace, etc.)
- Properly formatted training data

## Basic Usage

The Fine-Tuning CLI is located in the `tools` directory of the llm-chatbot-framework:

```bash
# Navigate to the UTTA directory
cd path/to/UTTA

# Activate the conda environment
conda activate utta

# Get help on available commands
python llm-chatbot-framework/tools/fine_tuning_cli.py --help
```

## Command Structure

The CLI follows this general command structure:

```
python tools/fine_tuning_cli.py [framework] [command] [options]
```

Where:
- `framework` is one of: `dspy`, `openai`, `huggingface`
- `command` depends on the framework (e.g., `optimize`, `start`, `fine-tune`)
- `options` are specific to each command

## DSPy Optimization

DSPy optimization helps improve prompts without fine-tuning the model itself:

```bash
# Optimize using examples
python tools/fine_tuning_cli.py dspy optimize --examples examples.json --module teacher_responder

# List optimization results
python tools/fine_tuning_cli.py dspy list-runs
```

## OpenAI Fine-Tuning

Manage OpenAI fine-tuning jobs:

```bash
# Start a fine-tuning job
python tools/fine_tuning_cli.py openai start --file training_data.jsonl --model gpt-3.5-turbo

# Check job status
python tools/fine_tuning_cli.py openai status --job-id ft-job-123456

# List active jobs
python tools/fine_tuning_cli.py openai list-jobs

# List fine-tuned models
python tools/fine_tuning_cli.py openai list-models
```

## HuggingFace Fine-Tuning

Fine-tune open-source models from HuggingFace:

```bash
# Fine-tune a model
python tools/fine_tuning_cli.py huggingface fine-tune --file training_data.jsonl --model microsoft/phi-2 --use-lora

# List available models
python tools/fine_tuning_cli.py huggingface list-models

# Export a fine-tuned model
python tools/fine_tuning_cli.py huggingface export-model --model-dir path/to/model
```

## Data Preparation

The CLI also offers utilities for preparing and validating training data:

```bash
# Validate data format
python tools/fine_tuning_cli.py prepare validate --input data.jsonl --format openai

# Convert between formats
python tools/fine_tuning_cli.py prepare convert --input data.json --output data.jsonl --format openai
```

## Common Options

Several options are available across different commands:

- `--verbose`: Enable verbose logging
- `--api-key`: Specify API key (overrides environment variable)
- `--output-dir`: Specify directory for outputs
- `--config-file`: Use a configuration file instead of command-line options

## Configuration File

For complex settings, you can use a configuration file:

```bash
python tools/fine_tuning_cli.py openai start --config-file configs/fine_tuning_config.json
```

Example configuration file:

```json
{
  "file": "training_data.jsonl",
  "model": "gpt-3.5-turbo",
  "suffix": "english-teacher",
  "hyperparameters": {
    "n_epochs": 3,
    "batch_size": 4
  }
}
```

## Troubleshooting

If you encounter issues:

1. Check the log file: `fine_tuning_cli.log`
2. Use the `--verbose` flag for more detailed output
3. Verify your API keys are correctly set
4. Validate your training data format 