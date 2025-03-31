# LoRA Fine-tuning Guide

This guide explains how to use Low-Rank Adaptation (LoRA) for fine-tuning language models in the UTTA project.

## Overview

LoRA is an efficient fine-tuning method that significantly reduces the number of trainable parameters while maintaining model performance. It's particularly useful for adapting large language models to specific tasks with limited computational resources.

## Getting Started

1. Install required packages:
```bash
pip install transformers peft datasets
```

2. Import necessary modules:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
```

## Basic Usage

Here's a simple example of how to apply LoRA to a pre-trained model:

```python
# Load base model and tokenizer
model = AutoModelForCausalLM.from_pretrained("base_model_name")
tokenizer = AutoTokenizer.from_pretrained("base_model_name")

# Configure LoRA
lora_config = LoraConfig(
    r=8,  # rank of update matrices
    lora_alpha=32,  # scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Create PEFT model
model = get_peft_model(model, lora_config)
```

## Best Practices

1. Choose appropriate rank for your task
2. Select relevant target modules
3. Monitor training metrics
4. Use proper evaluation methods

## Resources

- [Example Implementation](../huggingface_lora.py)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685) 