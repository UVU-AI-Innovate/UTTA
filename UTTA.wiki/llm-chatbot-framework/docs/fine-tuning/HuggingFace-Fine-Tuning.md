# HuggingFace Fine-Tuning

HuggingFace Fine-Tuning allows local fine-tuning of open-source models using efficient techniques like LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA).

## What It Is

HuggingFace Fine-Tuning allows you to fine-tune open-source models locally using efficient techniques like LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA). This provides complete control over the fine-tuning process and model deployment.

## How It Contributes to the Application

- **Local Deployment**: Enables fully local operation without API dependencies
- **Privacy**: Keeps sensitive training data entirely in your environment
- **Customization**: Provides full control over model architecture and parameters
- **Cost Control**: Eliminates ongoing API costs after initial training
- **Efficiency**: Supports 4-bit and 8-bit quantization for resource-efficient training

## Local Educational AI Infrastructure

HuggingFace fine-tuning enables the framework to operate without relying on external APIs, which is crucial for:

- **Educational Data Privacy**: Keeping student interactions and data entirely local
- **Offline Deployment**: Supporting schools with limited internet connectivity 
- **Cost Control**: Eliminating ongoing API costs for high-volume educational deployments
- **Customization**: Building specialized models for unique educational contexts

## Practical Impact

In the LLM Chatbot Framework, HuggingFace Fine-Tuning:

1. **Enables Self-Hosted Solutions**: Allows schools to run the entire system on their infrastructure
2. **Supports Specialized Educational Models**: Creates models tailored to specific curricula or teaching approaches
3. **Provides Resource Flexibility**: Offers options for deployment across different hardware configurations
4. **Enables Continuous Improvement**: Facilitates ongoing model refinement based on actual classroom usage

## Implementation Details

The HuggingFace fine-tuning support is implemented in `src/llm/fine_tuning/huggingface_tuner.py` and provides:

- `load_model()`: Loads base models with optional quantization
- `prepare_dataset()`: Converts training data to HuggingFace dataset format
- `fine_tune()`: Fine-tunes a model using LoRA/QLoRA
- `save_model()`: Saves fine-tuned models and adapters
- `load_fine_tuned_model()`: Loads a previously fine-tuned model
- `create_handler()`: Creates a handler using the fine-tuned model

## Example Usage

```python
from src.llm.fine_tuning.huggingface_tuner import HuggingFaceTuner

# Initialize the tuner
tuner = HuggingFaceTuner()

# Load a base model (with 4-bit quantization)
model, tokenizer = tuner.load_model(
    "meta-llama/Llama-2-7b-chat-hf",
    load_in_4bit=True
)

# Prepare the dataset
dataset = tuner.prepare_dataset("cache/fine_tuning/education_examples.jsonl")

# Fine-tune the model
output_dir = "cache/fine_tuned_models/llama-2-edu"
tuner.fine_tune(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    output_dir=output_dir,
    epochs=3,
    learning_rate=2e-5,
    lora_r=16
)

# Create a handler using the fine-tuned model
handler = tuner.create_handler(output_dir)
```

## Code Integration

The HuggingFace fine-tuning integration provides tools to seamlessly transition between cloud and local models:

```python
# Example from the framework showing local model integration
tuner = HuggingFaceTuner()
handler = tuner.create_handler("cache/fine_tuned_models/llama-2-edu")
# The handler provides the same interface as cloud-based handlers
```

## Using with the CLI Tool

The framework includes a CLI tool for HuggingFace fine-tuning:

```bash
# Fine-tune a model
python tools/fine_tuning_cli.py huggingface fine-tune --model-name meta-llama/Llama-2-7b-chat-hf --training-file data.jsonl

# List fine-tuned models
python tools/fine_tuning_cli.py huggingface list-models

# Export a model for integration
python tools/fine_tuning_cli.py huggingface export-model --model-dir cache/fine_tuned_models/model-name
```

## Hardware Requirements

HuggingFace fine-tuning has varying hardware requirements depending on model size:

| Model Size | Recommended GPU | Memory (with QLoRA) | Memory (standard LoRA) |
|------------|-----------------|---------------------|-------------------------|
| 7B         | NVIDIA T4/16GB  | ~12GB               | ~20GB                  |
| 13B        | NVIDIA A10/24GB | ~18GB               | ~32GB                  |
| 70B        | NVIDIA A100/80GB| ~60GB               | Not recommended        |

## Troubleshooting

Common issues with HuggingFace fine-tuning:

- **Memory Errors**: Try using quantization (4-bit or 8-bit)
- **CUDA Errors**: Ensure your CUDA toolkit and PyTorch versions are compatible
- **Model Loading Errors**: Check if you have sufficient disk space for model weights
- **Training Stuck**: Reduce batch size or use gradient accumulation

For more details, see the [Fine-Tuning CLI](Fine-Tuning-CLI) and [Troubleshooting](Fine-Tuning-Troubleshooting) pages.

## Related Pages

- [Fine-Tuning Overview](Fine-Tuning-Overview)
- [DSPy Optimization](DSPy-Optimization)
- [OpenAI Fine-Tuning](OpenAI-Fine-Tuning)
- [Fine-Tuning Contributions](Fine-Tuning-Contributions) 