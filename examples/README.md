# LLM Fine-Tuning Examples

This directory contains examples of different approaches to fine-tuning large language models (LLMs) for educational question answering.

## Comparison of Fine-Tuning Approaches

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small | Medium | Large |
| Cost | API calls only | API + training | One-time compute |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU recommended |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |

## Example Files

- **DSPy Optimization**: [dspy_example.py](dspy_example.py)
- **OpenAI Fine-Tuning**: [openai_finetune.py](openai_finetune.py)
- **HuggingFace LoRA**: [huggingface_lora.py](huggingface_lora.py)

## Datasets

These examples use the same basic educational QA dataset for consistency and comparison:
- Base dataset: 5 educational QA pairs (used in all examples)
- Extended dataset: 10-20 educational QA pairs (used in OpenAI and HuggingFace examples)
- The OpenAI formatted dataset is saved as [openai_edu_qa_training.jsonl](openai_edu_qa_training.jsonl)
- DSPy uses [small_edu_qa.jsonl](small_edu_qa.jsonl) (created on first run)

## Running the Examples

All examples are designed to be educational and include safety measures to prevent accidental execution that might incur costs or require significant computing resources.

### DSPy Example

```bash
python dspy_example.py
```

The DSPy example demonstrates:
- Using Chain-of-Thought prompting rather than model fine-tuning
- Working with very small datasets (as few as 5 examples)
- No model weight updates (only better prompting)

### OpenAI Example

```bash
python openai_finetune.py
```

The OpenAI example demonstrates:
- Cloud-based fine-tuning through OpenAI's API
- More examples required than DSPy (10+ examples recommended)
- Actual model weight updates (currently simulation only)

### HuggingFace LoRA Example

```bash
python huggingface_lora.py
```

The HuggingFace example demonstrates:
- Local fine-tuning with LoRA (Low-Rank Adaptation)
- Full control over the training process
- Data stays on your local machine
- Requires GPU for actual training (currently simulation only)

## Step-by-Step Structure

Each example follows the same educational structure for easy comparison:

1. **Environment & Data Preparation**: Setup and dataset preparation
2. **Data Format Preparation**: Converting data to the required format
3. **Model Configuration**: Setting up the model architecture
4. **Fine-Tuning Implementation**: The actual training process
5. **Inference and Demonstration**: Using the model to answer questions
6. **Comparison**: Direct comparison with other methods

## Requirements

Different requirements depending on which example you're running:

- For DSPy: OpenAI API key, dspy library
- For OpenAI: OpenAI API key, openai library
- For HuggingFace: GPU with CUDA, transformers, peft, bitsandbytes

See the main requirements.txt file for specific versions.

## Educational Purpose

These examples are designed for educational comparison and should not be run in production without further development. They demonstrate the key differences between:

1. **Prompt optimization** (DSPy): No model changes, just better instructions
2. **Cloud fine-tuning** (OpenAI): Updating model weights through a service
3. **Local fine-tuning** (HuggingFace): Full control with your own infrastructure