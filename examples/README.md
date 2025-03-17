# LLM Fine-Tuning Examples

This directory contains examples of different approaches to fine-tuning large language models (LLMs) for educational question answering. These examples are provided under the terms of the LICENSE file included in the repository.

## 📋 Table of Contents
- [LLM Fine-Tuning Examples](#llm-fine-tuning-examples)
  - [📋 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Example Files](#example-files)
    - [Datasets](#datasets)
  - [Three Approaches to LLM Improvement](#three-approaches-to-llm-improvement)
    - [DSPy: Prompt Optimization](#dspy-prompt-optimization)
    - [OpenAI: Cloud Fine-Tuning](#openai-cloud-fine-tuning)
    - [HuggingFace: Local Fine-Tuning with LoRA](#huggingface-local-fine-tuning-with-lora)
  - [Technical Deep Dive: Understanding LoRA](#technical-deep-dive-understanding-lora)
    - [How LoRA Works](#how-lora-works)
    - [Benefits of LoRA](#benefits-of-lora)
    - [LoRA Visualization](#lora-visualization)
  - [Cost Analysis](#cost-analysis)
    - [Side-by-Side Comparison](#side-by-side-comparison)
  - [Alternative LLM Options](#alternative-llm-options)
    - [Mixtral 8x7B](#mixtral-8x7b)
    - [Llama 2](#llama-2)
    - [Claude Instant](#claude-instant)
    - [Choosing the Right Alternative](#choosing-the-right-alternative)
  - [Example Implementation Details](#example-implementation-details)
  - [Decision Guide: Choosing the Right Approach](#decision-guide-choosing-the-right-approach)
    - [At-a-Glance Comparison](#at-a-glance-comparison)
    - [Choose Based On Your Needs](#choose-based-on-your-needs)
      - [Dataset Size:](#dataset-size)
      - [Time Constraints:](#time-constraints)
      - [Budget:](#budget)
      - [Privacy Requirements:](#privacy-requirements)

## Overview

This educational project demonstrates three different approaches to improving LLM performance for question-answering tasks:

1. **DSPy**: Prompt optimization without model changes
2. **OpenAI**: Cloud-based fine-tuning
3. **HuggingFace**: Local fine-tuning with LoRA

Each approach offers different tradeoffs in terms of cost, complexity, and performance.

## Getting Started

### Prerequisites
All dependencies for these examples are included in the main project's `environment.yml` file.

```bash
# Create and activate conda environment from the project root
conda env create -f environment.yml
conda activate utta
```

### Example Files
- **DSPy Optimization**: [dspy_example.py](dspy_example.py)
- **OpenAI Fine-Tuning**: [openai_finetune.py](openai_finetune.py)
- **HuggingFace LoRA**: [huggingface_lora.py](huggingface_lora.py)

### Datasets
All examples use consistent datasets for fair comparison:
- **Base dataset**: 5 educational QA pairs (used in all examples)
- **Extended dataset**: 10-20 educational QA pairs (used in OpenAI and HuggingFace examples)
- Data formats:
  - DSPy uses [small_edu_qa.jsonl](small_edu_qa.jsonl) (created on first run)
  - OpenAI uses [openai_edu_qa_training.jsonl](openai_edu_qa_training.jsonl)

## Three Approaches to LLM Improvement

### DSPy: Prompt Optimization

**Key Features:**
- No model weight changes - only better prompting
- Works with very small datasets (5 examples)
- Uses Chain-of-Thought prompting technique
- API-based approach with immediate results

**How to Run:**
```bash
python dspy_example.py
```

**Cost Considerations:**
- Requires payment for each API call to the underlying LLM service
- Can use cheaper LLMs like Mixtral, Llama 2, or Claude Instant
- Typical cost: $0.01-$0.20 per optimization cycle

**Example Output:**
```
Q: What causes the seasons on Earth?
A: The changing angle of sunlight due to the Earth's tilted axis 
   as it orbits the Sun causes the seasons on Earth.
```

### OpenAI: Cloud Fine-Tuning

**Key Features:**
- Updates actual model weights through cloud service
- Requires medium-sized datasets (10+ examples)
- Balance of control and convenience
- Handles complex questions with more detailed answers

**How to Run:**
```bash
python openai_finetune.py
```

**Cost Considerations:**
- One-time training fee ($0.03-$0.80 per 1K training tokens)
- Ongoing API costs for inference ($0.01-$0.12 per 1K tokens)
- GPT-3.5 Turbo is significantly cheaper than GPT-4
- Typical cost: $5-$50 for training + ongoing inference costs

**Example Output:**
```
Q: What causes the seasons on Earth?
A: The seasons on Earth are caused by the tilt of the Earth's axis 
   as it orbits around the Sun. This tilt changes the angle at which 
   sunlight hits different parts of the Earth throughout the year, 
   creating seasonal variations in temperature and daylight hours.
```

### HuggingFace: Local Fine-Tuning with LoRA

**Key Features:**
- Full local control over model and training process
- Uses LoRA for efficient adaptation of model weights
- Works best with larger datasets (20+ examples)
- Data privacy - all information stays on your machine

**How to Run:**
```bash
python huggingface_lora.py
```

**Cost Considerations:**
- One-time compute cost for GPU usage, no ongoing API fees
- Can use smaller open-source models and lower precision (int8/int4)
- For this project: No cloud costs needed as we have server GPU access

**Example Output:**
```
Q: What causes the seasons on Earth?
A: The seasons on Earth are caused by the tilt of Earth's axis 
   relative to its orbit around the Sun. As Earth orbits the Sun, 
   different parts of the planet receive sunlight at different angles, 
   creating seasonal variations in temperature and daylight hours.
```

## Technical Deep Dive: Understanding LoRA

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that significantly reduces the computational and memory requirements for fine-tuning large language models.

### How LoRA Works
1. **Traditional fine-tuning** updates all model weights (often billions of parameters)
2. **LoRA** instead:
   - Freezes the original model weights
   - Adds small trainable "adapter" matrices using low-rank decomposition
   - Only trains these adapter parameters (typically <1% of original model parameters)

### Benefits of LoRA
- **Memory efficient**: Requires much less GPU VRAM than full fine-tuning (up to 70% reduction)
- **Training efficient**: Trains faster with fewer resources
- **Storage efficient**: The resulting fine-tuned model is much smaller
- **Adaptable**: Can create multiple different adaptations for the same base model
- **Performance comparable**: Often achieves results similar to full fine-tuning

### LoRA Visualization
```
Original Model Weights ──────┐
                             │
                             ├──► Combined for inference
                             │
LoRA Adapter Weights ────────┘
   (trained)
```

## Cost Analysis

### Side-by-Side Comparison

| Cost Factor | DSPy | OpenAI | HuggingFace |
|-------------|------|--------|-------------|
| Initial cost | None | $5-$50 (training) | GPU hardware/rental |
| Ongoing costs | API calls for each request | API calls for each request | None (self-hosted) |
| Cost reduction options | Use cheaper models (Mixtral, Llama) | Use GPT-3.5 instead of GPT-4 | Use smaller models/lower precision |
| Data usage costs | API calls during optimization | Training + API inference | None |
| Pricing model | Pay-per-request | Training + pay-per-request | One-time compute |
| Best for budget | Small, quick experiments | Medium-term projects | Long-term, high-volume usage |
| With GPU server access | Still requires API costs | Still requires API costs | Cheapest overall option |

## Alternative LLM Options

When working with these approaches, you can use different LLMs to optimize for cost, performance, and other factors. Here are details on some recommended alternatives to GPT-4:

### Mixtral 8x7B
- **What it is**: An open-weights mixture-of-experts model developed by Mistral AI
- **Architecture**: Uses a sparse mixture-of-experts approach where only a subset of parameters activate for each token
- **Performance**: Competitive with much larger models while being more efficient
- **Cost**: Can be used via Mistral API (~$0.0002/1K tokens) or self-hosted
- **Best for**: DSPy or HuggingFace approaches where you need strong reasoning at lower cost
- **Availability**: Commercial use allowed

### Llama 2
- **What it is**: Meta's open-source model family available in multiple sizes (7B, 13B, 70B parameters)
- **Architecture**: Traditional transformer architecture with improvements from the original Llama
- **Performance**: The 70B variant approaches GPT-3.5 performance for many tasks
- **Cost**: Free for self-hosting; also available through various APIs
- **Best for**: HuggingFace approach where you can fine-tune locally
- **Sizes**: 
  - 7B: Fastest, lowest memory requirements (16GB VRAM)
  - 13B: Better quality, moderate requirements (24GB VRAM)
  - 70B: Best quality, high requirements (80GB+ VRAM or quantized)
- **Availability**: Commercial use allowed with a license

### Claude Instant
- **What it is**: Anthropic's faster, more cost-effective version of their Claude model
- **Architecture**: Proprietary architecture with constitutional AI alignment
- **Performance**: Good balance of reasoning capabilities and speed
- **Cost**: ~$0.008/1K tokens (input), ~$0.024/1K tokens (output)
- **Best for**: DSPy approach when you need strong reasoning but GPT-4 is too expensive
- **Availability**: Only available through Anthropic's API

### Choosing the Right Alternative

| Factor | Mixtral 8x7B | Llama 2 | Claude Instant |
|--------|-------------|---------|----------------|
| API Cost | Low | Very low/Free | Medium |
| Self-hostable | Yes | Yes | No |
| Reasoning ability | High | Medium-High | High |
| Context length | 32K tokens | 4K tokens | 100K tokens |
| Best paired with | DSPy/HuggingFace | HuggingFace | DSPy |

## Example Implementation Details

All examples follow the same educational structure:

1. **Environment & Data Preparation**: Setup and dataset preparation
2. **Data Format Preparation**: Converting data to the required format
3. **Model Configuration**: Setting up the model architecture
4. **Fine-Tuning Implementation**: The actual training process
5. **Inference and Demonstration**: Using the model to answer questions
6. **Comparison**: Direct comparison with other methods

All examples include safety measures to prevent accidental execution that might incur costs or require significant computing resources.

## Decision Guide: Choosing the Right Approach

### At-a-Glance Comparison

| Feature | DSPy | OpenAI | HuggingFace |
|---------|------|--------|-------------|
| What changes | Prompts | Model weights (cloud) | Model weights (local) |
| Training data needed | Small (5 examples) | Medium (10+ examples) | Large (20+ examples) |
| Setup difficulty | Simple | Simple | Complex |
| Control | Limited | Medium | Full |
| Hardware required | None | None | GPU (16GB+ VRAM) |
| Deployment | API calls | API calls | Self-host |
| Data privacy | Data shared with API | Data shared with API | Data stays local |
| Time to results | Immediate | Hours | Hours (with GPU) |
| Answer quality | Good | Better | Best |

### Choose Based On Your Needs

#### Dataset Size:
- **Small (5-10 examples)** → DSPy
- **Medium (10-50 examples)** → OpenAI
- **Large (50+ examples)** → HuggingFace

#### Time Constraints:
- **Need immediate results** → DSPy
- **Can wait hours** → OpenAI/HuggingFace

#### Budget:
- **Minimal budget** → DSPy (still requires paying for API calls)
- **Medium budget** → OpenAI
- **One-time compute cost** → HuggingFace
- **Cheapest overall**: HuggingFace with existing GPU (recommended for this project since we have GPU server access)
- **Cheapest without GPU**: DSPy with budget-friendly models

#### Privacy Requirements:
- **Standard** → DSPy/OpenAI
- **High privacy needs** → HuggingFace
