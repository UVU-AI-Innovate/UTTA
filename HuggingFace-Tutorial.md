# HuggingFace Tutorial for UTTA

This tutorial will guide you through using HuggingFace models with the UTTA framework.

## Introduction

HuggingFace provides a vast ecosystem of open-source models that can be leveraged for educational purposes. This guide shows how to integrate and use HuggingFace models within the UTTA framework.

## Prerequisites

Before starting this tutorial, ensure you have:

* Completed the [Environment Setup](Environment-Setup)
* A HuggingFace account and API token (optional, but recommended for some features)
* Basic understanding of the UTTA framework
* Sufficient hardware resources (some models require significant RAM/GPU)

## Setting Up HuggingFace Access

1. **Create a HuggingFace Account**
   * Sign up at [HuggingFace](https://huggingface.co/join)

2. **Generate an API Token** (for API access)
   * Go to [Settings → Access Tokens](https://huggingface.co/settings/tokens)
   * Create a new token
   * Copy and securely store your token

3. **Set Up Environment Variable**
   ```bash
   # Add to your environment or .env file
   export HUGGINGFACE_API_KEY="your-token-here"
   ```

## Basic Usage of HuggingFace Models

### Initializing with HuggingFace Models

```python
from utta import TeachingAssistant

# Initialize with a HuggingFace model
assistant = TeachingAssistant(model="huggingface/google/flan-t5-large")

# Ask a question
response = assistant.ask("What is the difference between mitosis and meiosis?")
print(response)
```

### Using Different Model Types

```python
# Using a conversational model
chat_assistant = TeachingAssistant(model="huggingface/meta-llama/Llama-2-7b-chat-hf")

# Using a specialized educational model
edu_assistant = TeachingAssistant(model="huggingface/google/t5-11b")

# Using a multilingual model for language learning
lang_assistant = TeachingAssistant(model="huggingface/facebook/mbart-large-50-many-to-many-mmt")
```

### Model Configuration Options

```python
# Configure with specific parameters
assistant = TeachingAssistant(
    model="huggingface/google/flan-t5-large",
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2
)

# Update parameters after initialization
assistant.update_model_parameters(temperature=0.5)
```

## Running Models Locally

One advantage of HuggingFace is the ability to run models locally:

```python
# Initialize with a locally downloaded model
local_assistant = TeachingAssistant(
    model="huggingface/local:/path/to/downloaded/model",
    device="cuda"  # or "cpu" for CPU-only inference
)

# For optimized local running
optimized_assistant = TeachingAssistant(
    model="huggingface/local:/path/to/model",
    quantization="8bit",  # Reduce memory requirements
    device="cuda"
)
```

### Downloading Models for Local Use

```python
from utta.utils import ModelDownloader

# Download a model for local use
downloader = ModelDownloader()
model_path = downloader.download("google/flan-t5-large", target_dir="./models")

# Now use the downloaded model
assistant = TeachingAssistant(model=f"huggingface/local:{model_path}")
```

## Advanced Usage

### Model Adapters and LoRA

For more efficient fine-tuning, you can use adapters and LoRA (Low-Rank Adaptation):

```python
from utta.adaptation import LoRAAdapter

# Initialize an adapter
adapter = LoRAAdapter(base_model="meta-llama/Llama-2-7b-hf")

# Train the adapter on your educational data
adapter.train(
    dataset_path="education_examples.jsonl",
    output_dir="./lora_adapter",
    epochs=3
)

# Use the adapted model
adapted_assistant = TeachingAssistant(
    model="huggingface/meta-llama/Llama-2-7b-hf",
    adapter_path="./lora_adapter"
)
```

### Pipeline Integration

HuggingFace pipelines can be integrated for specific tasks:

```python
from utta.pipelines import PipelineIntegrator

# Create a pipeline integrator
integrator = PipelineIntegrator()

# Add a summarization pipeline
integrator.add_pipeline("summarization", model="facebook/bart-large-cnn")

# Add a question-answering pipeline
integrator.add_pipeline("question-answering", model="deepset/roberta-base-squad2")

# Integrate with your assistant
assistant = TeachingAssistant(model="huggingface/google/flan-t5-large")
assistant.integrate_pipelines(integrator)

# Now the assistant can use these pipelines
summary = assistant.summarize("A long passage about the solar system...")
answer = assistant.answer_factual_question("When was the first moon landing?", context="...")
```

## Educational Applications

### Multilingual Learning Assistant

```python
from utta import TeachingAssistant
from utta.education import LanguageLearningModule

# Create a multilingual teaching assistant
assistant = TeachingAssistant(model="huggingface/facebook/mbart-large-50-many-to-many-mmt")

# Add language learning capabilities
lang_module = LanguageLearningModule()
assistant.add_module(lang_module)

# Examples of usage
translation = assistant.translate("Hello, how are you?", source_lang="en", target_lang="es")
grammar_check = assistant.check_grammar("She don't like apples", lang="en")
vocabulary = assistant.get_vocabulary("astronomy", level="intermediate", lang="fr")
```

### Creating Interactive Learning Activities

```python
from utta.education import ActivityGenerator

# Initialize an activity generator
activity_gen = ActivityGenerator(model="huggingface/google/flan-t5-xl")

# Generate different types of activities
matching_activity = activity_gen.create_activity(
    subject="Geography",
    activity_type="matching",
    topic="European Capitals",
    difficulty="middle school"
)

# Generate a fill-in-the-blanks exercise
fill_blanks = activity_gen.create_activity(
    subject="History",
    activity_type="fill_blanks",
    topic="American Revolution",
    difficulty="high school"
)

print(matching_activity)
print(fill_blanks)
```

## Fine-Tuning HuggingFace Models

For customizing models to your specific educational needs:

```python
from utta.fine_tuning import HuggingFaceFineTuner

# Initialize the fine-tuner
tuner = HuggingFaceFineTuner(
    base_model="google/flan-t5-base",
    output_dir="./fine_tuned_model"
)

# Prepare your dataset
tuner.prepare_dataset("education_examples.jsonl")

# Start fine-tuning
tuner.train(
    epochs=3,
    batch_size=8,
    learning_rate=2e-5,
    save_steps=500
)

# Use the fine-tuned model
fine_tuned_assistant = TeachingAssistant(
    model="huggingface/local:./fine_tuned_model"
)
```

## Evaluating Models

```python
from utta.evaluation import ModelEvaluator

# Initialize an evaluator
evaluator = ModelEvaluator()

# Add models to compare
evaluator.add_model(
    TeachingAssistant(model="huggingface/google/flan-t5-large"),
    name="FLAN-T5-Large"
)
evaluator.add_model(
    TeachingAssistant(model="huggingface/meta-llama/Llama-2-7b-chat-hf"),
    name="Llama-2-7B"
)

# Define evaluation tasks
evaluator.add_evaluation_task("factual_knowledge", dataset="education_qa.jsonl")
evaluator.add_evaluation_task("explanation_quality", dataset="concept_explanations.jsonl")

# Run evaluation
results = evaluator.evaluate()
print(results.summary())
print(results.comparison_chart())
```

## Performance Optimization

### Quantization

```python
# Use quantized models for faster inference and lower memory usage
assistant = TeachingAssistant(
    model="huggingface/meta-llama/Llama-2-7b-chat-hf",
    quantization="4bit",  # Options: 4bit, 8bit
    device="cuda"
)
```

### Model Pruning

```python
from utta.optimization import ModelPruner

# Prune a model for efficiency
pruner = ModelPruner(sparsity=0.3)  # 30% sparsity
pruned_model_path = pruner.prune(
    model_path="./models/flan-t5-large",
    output_dir="./pruned_model"
)

# Use the pruned model
assistant = TeachingAssistant(
    model=f"huggingface/local:{pruned_model_path}"
)
```

## Best Practices

1. **Choose the Right Model Size**: Balance performance with resource requirements
2. **Use Quantization**: For faster inference with minimal quality loss
3. **Consider Local Models**: For privacy and reduced API costs
4. **Batch Processing**: Process multiple inputs at once for efficiency
5. **Evaluate Systematically**: Compare models on relevant educational tasks
6. **Version Control**: Keep track of fine-tuned model versions

## Troubleshooting

* **Memory Issues**: Try smaller models or quantization
* **Slow Inference**: Check hardware utilization, consider optimized models
* **Installation Problems**: Check CUDA compatibility for GPU acceleration
* **Model Loading Errors**: Verify paths and model compatibility

## Further Resources

* [HuggingFace Documentation](https://huggingface.co/docs)
* [HuggingFace Model Hub](https://huggingface.co/models)
* [UTTA Documentation](Home)
* [Transformers for Education](https://github.com/UVU-AI-Innovate/UTTA/wiki/Transformers-for-Education) 