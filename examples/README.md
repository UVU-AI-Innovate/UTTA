# LLM Fine-Tuning Examples

This directory contains examples of different approaches to fine-tuning large language models (LLMs) for educational question answering. These examples are provided under the terms of the LICENSE file included in the repository.

## Overview

This educational project demonstrates three different approaches to improving LLM performance:
- **DSPy**: Prompt optimization without model changes
- **OpenAI**: Cloud-based fine-tuning
- **HuggingFace**: Local fine-tuning with LoRA

## Requirements

All dependencies for these examples are included in the main project's `environment.yml` file. Make sure you have set up the conda environment as described in the main README:

```bash
# Create and activate conda environment from the project root
conda env create -f environment.yml
conda activate utta
```

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

## Cost Considerations

### DSPy
- **API costs**: Requires payment for each API call to the underlying LLM service
- **Usage pattern**: Every optimization and inference request incurs API costs
- **Cost reduction**: Can use cheaper LLMs like Mixtral, Llama 2, or Claude Instant instead of GPT-4
- **Typical cost range**: $0.01-$0.20 per optimization cycle, depending on chosen model

### OpenAI
- **Training costs**: One-time fee for fine-tuning ($0.03-$0.80 per 1K training tokens)
- **Inference costs**: Ongoing API costs for every request ($0.01-$0.12 per 1K tokens)
- **Model options**: GPT-3.5 Turbo is significantly cheaper than GPT-4 for both training and inference
- **Typical cost range**: $5-$50 for training + ongoing inference costs

### HuggingFace
- **One-time compute**: GPU costs for training (either cloud GPU rental or local hardware)
- **Deployment costs**: Only if self-hosting (server/cloud costs)
- **Cost reduction**: Can use smaller open-source models like Llama 2 7B instead of larger variants
- **Typical cost range**: $5-$15 for cloud GPU rental during training, or free if using existing hardware
- **Note for this project**: No cloud costs needed as we have access to a server with GPU

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
- **Note**: Still requires payment for underlying LLM API calls
- **Cost saving option**: Can configure to use cheaper models like Mixtral or Llama 2

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
- **Note**: Incurs both one-time training costs and ongoing API call costs
- **Cost saving option**: Can use GPT-3.5 Turbo instead of GPT-4 for significant cost reduction

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
- **Note**: Requires one-time compute cost (GPU usage) but no ongoing API costs
- **Cost saving option**: Can use smaller open-source models and lower precision (int8/int4)

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

## Educational Purpose

These examples demonstrate the key differences between:

1. **Prompt optimization** (DSPy):
   - No model changes, just better instructions
   - Best for: Quick prototyping, small datasets
   - Advantage: Immediate results, lowest cost
   - Limitation: Bound by base model capabilities
   - **Cost structure**: Requires payment for each API call to the underlying LLM service

2. **Cloud fine-tuning** (OpenAI):
   - Updating model weights through a service
   - Best for: Production deployment, medium datasets
   - Advantage: Balance of control and convenience
   - Limitation: Cost and data privacy
   - **Cost structure**: One-time training fee plus ongoing API costs for inference

3. **Local fine-tuning** (HuggingFace):
   - Full control with your own infrastructure
   - Best for: Full customization, large datasets
   - Advantage: Complete control, data privacy
   - Limitation: Complex setup, hardware requirements
   - **Cost structure**: One-time compute cost, no ongoing API fees

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
   - Minimal budget → DSPy (still requires paying for API calls)
   - Medium budget → OpenAI
   - One-time compute cost → HuggingFace
   - **Cheapest overall**: HuggingFace with existing GPU and small open-source models (recommended for this project since we have GPU server access)
   - **Cheapest without GPU**: DSPy with budget-friendly models like Mixtral or Claude Instant

4. **Privacy Requirements**:
   - Standard → DSPy/OpenAI
   - High privacy needs → HuggingFace
