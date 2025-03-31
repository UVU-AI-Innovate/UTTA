# OpenAI Integration Guide

This guide covers how to integrate and use OpenAI's models in the UTTA project.

## Overview

OpenAI provides powerful language models through their API, which can be used for various educational AI applications. This guide covers both basic API usage and fine-tuning approaches.

## Getting Started

1. Install the OpenAI package:
```bash
pip install openai
```

2. Set up your API key:
```python
import openai
openai.api_key = 'your-api-key'  # Better to use environment variable
```

## Basic Usage

Here's how to make basic API calls:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful teaching assistant."},
        {"role": "user", "content": "Explain photosynthesis simply."}
    ]
)
```

## Fine-tuning

For custom applications, you can fine-tune models:

```python
# Prepare training file
training_file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo"
)
```

## Best Practices

1. Use environment variables for API keys
2. Implement proper error handling
3. Monitor API usage and costs
4. Cache responses when appropriate

## Resources

- [Example Implementation](../openai_finetune.py)
- [OpenAI Documentation](https://platform.openai.com/docs)
- [Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) 