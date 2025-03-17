# LLM Fine-Tuning Examples

This directory contains examples of different approaches to fine-tuning large language models (LLMs) for educational question answering. These examples are provided under the terms of the LICENSE file included in the repository.

## Overview

This educational project demonstrates three different approaches to improving LLM performance:
- **DSPy**: Prompt optimization without model changes
- **OpenAI**: Cloud-based fine-tuning
- **HuggingFace**: Local fine-tuning with LoRA

## Comparison of Fine-Tuning Approaches

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (5 examples) | Medium (10+ examples) | Large (20+ examples) |
| Cost | API calls only | API + training | One-time compute |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU (16GB+ VRAM) |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |
| Time to results | Immediate | Hours | Hours (with GPU) |
| Answer quality | Good | Better | Best |

## Example Files & Datasets

### Files
- **DSPy Optimization**: [dspy_example.py](dspy_example.py)
- **OpenAI Fine-Tuning**: [openai_finetune.py](openai_finetune.py)
- **HuggingFace LoRA**: [huggingface_lora.py](huggingface_lora.py)

### Datasets
These examples use consistent datasets for comparison:
- Base dataset: 5 educational QA pairs (used in all examples)
- Extended dataset: 10-20 educational QA pairs (used in OpenAI and HuggingFace examples)
- The OpenAI formatted dataset is saved as [openai_edu_qa_training.jsonl](openai_edu_qa_training.jsonl)
- DSPy uses [small_edu_qa.jsonl](small_edu_qa.jsonl) (created on first run)

## Running the Examples

All examples include safety measures to prevent accidental execution that might incur costs or require significant computing resources.

### DSPy Example

```bash
python dspy_example.py
```

The DSPy example demonstrates:
- Using Chain-of-Thought prompting rather than model fine-tuning
- Working with very small datasets (as few as 5 examples)
- No model weight updates (only better prompting)

Expected Results:
- Direct, concise answers with logical reasoning
- Immediate results through API calls
- Good performance with minimal setup
- Example output:
  ```
  Q: What causes the seasons on Earth?
  A: The changing angle of sunlight due to the Earth's tilted axis 
     as it orbits the Sun causes the seasons on Earth.
  ```

### OpenAI Example

```bash
python openai_finetune.py
```

The OpenAI example demonstrates:
- Cloud-based fine-tuning through OpenAI's API
- More examples required than DSPy (10+ examples recommended)
- Actual model weight updates (currently simulation only)

Expected Results:
- More detailed and domain-specific answers
- Better handling of complex questions
- Consistent response style
- Example output:
  ```
  Q: What causes the seasons on Earth?
  A: The seasons on Earth are caused by the tilt of the Earth's axis 
     as it orbits around the Sun. This tilt changes the angle at which 
     sunlight hits different parts of the Earth throughout the year, 
     creating seasonal variations in temperature and daylight hours.
  ```

### HuggingFace LoRA Example

```bash
python huggingface_lora.py
```

The HuggingFace example demonstrates:
- Local fine-tuning with LoRA (Low-Rank Adaptation)
- Full control over the training process
- Data stays on your local machine
- Requires GPU for actual training (currently simulation only)

Expected Results:
- Most technically precise answers
- Full control over model behavior
- Customizable response format
- Example output:
  ```
  Q: What causes the seasons on Earth?
  A: The seasons on Earth are caused by the tilt of Earth's axis 
     relative to its orbit around the Sun. As Earth orbits the Sun, 
     different parts of the planet receive sunlight at different angles, 
     creating seasonal variations in temperature and daylight hours.
  ```

## Implementation Structure

Each example follows the same educational structure:

1. **Environment & Data Preparation**: Setup and dataset preparation
2. **Data Format Preparation**: Converting data to the required format
3. **Model Configuration**: Setting up the model architecture
4. **Fine-Tuning Implementation**: The actual training process
5. **Inference and Demonstration**: Using the model to answer questions
6. **Comparison**: Direct comparison with other methods

## Technical Requirements

Different requirements depending on which example you're running:

- For DSPy:
  * OpenAI API key
  * dspy library
  * Python 3.8+
  * Internet connection

- For OpenAI:
  * OpenAI API key
  * openai library
  * Python 3.8+
  * Internet connection
  * OpenAI account with billing enabled

- For HuggingFace:
  * GPU with 16GB+ VRAM
  * CUDA support
  * transformers library
  * peft library
  * bitsandbytes library
  * Python 3.8+
  * ~10GB disk space for models

See the main requirements.txt file for specific versions.

## Educational Purpose

These examples demonstrate the key differences between:

1. **Prompt optimization** (DSPy):
   - No model changes, just better instructions
   - Best for: Quick prototyping, small datasets
   - Advantage: Immediate results, lowest cost
   - Limitation: Bound by base model capabilities

2. **Cloud fine-tuning** (OpenAI):
   - Updating model weights through a service
   - Best for: Production deployment, medium datasets
   - Advantage: Balance of control and convenience
   - Limitation: Cost and data privacy

3. **Local fine-tuning** (HuggingFace):
   - Full control with your own infrastructure
   - Best for: Full customization, large datasets
   - Advantage: Complete control, data privacy
   - Limitation: Complex setup, hardware requirements

## Choosing an Approach

Consider these factors when choosing an approach:
1. **Dataset Size**: 
   - Small (5-10 examples) → DSPy
   - Medium (10-50 examples) → OpenAI
   - Large (50+ examples) → HuggingFace

2. **Time to Results**:
   - Need immediate results → DSPy
   - Can wait hours → OpenAI/HuggingFace

3. **Budget**:
   - Minimal budget → DSPy
   - Medium budget → OpenAI
   - One-time compute cost → HuggingFace

4. **Privacy Requirements**:
   - Standard → DSPy/OpenAI
   - High privacy needs → HuggingFace
