# UTTA Tools

This directory contains command-line tools for working with the UTTA framework.

## Fine-Tuning CLI

[`fine_tuning_cli.py`](fine_tuning_cli.py) provides a comprehensive command-line interface for managing fine-tuning jobs:

- Start, monitor, and manage fine-tuning jobs
- Support for OpenAI, HuggingFace, and DSPy optimization
- Data preparation utilities
- Model management

### Usage

```bash
# Get help
python tools/fine_tuning_cli.py --help

# OpenAI fine-tuning
python tools/fine_tuning_cli.py openai start --file training_data.jsonl --model gpt-3.5-turbo

# HuggingFace fine-tuning
python tools/fine_tuning_cli.py huggingface fine-tune --file training_data.jsonl --model microsoft/phi-2 --use-lora

# DSPy optimization
python tools/fine_tuning_cli.py dspy optimize --examples examples.json --module teacher_responder
```

## Installation as CLI Tool

After installing the UTTA package, you can use the fine-tuning CLI as a command-line tool:

```bash
# Install the package
pip install -e .

# Use the CLI tool
utta-finetune --help
```

This will make the fine-tuning CLI available as `utta-finetune` in your shell. 