# HuggingFace Tutorial

This tutorial guides you through using HuggingFace models with UTTA for creating effective teaching assistants.

## Prerequisites

Before starting this tutorial, ensure you have:
1. Completed the [Environment Setup](Environment-Setup)
2. HuggingFace account and API token
3. Basic understanding of Python and LLMs
4. UTTA installed and configured
5. Sufficient GPU resources (recommended)

## Basic Usage

### 1. Initialize HuggingFace with UTTA

```python
from utta.core import TeachingAssistant
from utta.frameworks.huggingface_framework import HuggingFaceFramework

# Initialize the framework
framework = HuggingFaceFramework()

# Create teaching assistant
assistant = TeachingAssistant(framework=framework)
```

### 2. Model Selection

```python
from utta.config import FrameworkConfig

# Choose a specific model
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Initialize framework with config
framework = HuggingFaceFramework(config=config)
assistant = TeachingAssistant(framework=framework)
```

## Advanced Features

### 1. Custom Model Configuration

```python
from transformers import AutoConfig

# Configure model parameters
model_config = AutoConfig.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    trust_remote_code=True,
    temperature=0.7,
    top_p=0.9,
    max_length=512
)

# Initialize framework with custom config
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    model_config=model_config,
    device="cuda"
)

framework = HuggingFaceFramework(config=config)
```

### 2. Quantization and Optimization

```python
from utta.optimization import load_quantized_model

# Load 4-bit quantized model
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    quantization="4bit",
    device_map="auto"  # Automatically manage model placement
)

# Initialize with optimized model
framework = HuggingFaceFramework(config=config)
```

### 3. Pipeline Configuration

```python
from transformers import pipeline

# Create custom pipeline
generation_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-chat-hf",
    tokenizer="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    max_length=2048,
    do_sample=True,
    temperature=0.7
)

# Use custom pipeline
config = FrameworkConfig(
    pipeline=generation_pipeline
)

framework = HuggingFaceFramework(config=config)
```

## Integration with UTTA

### 1. Custom Tokenization

```python
from transformers import AutoTokenizer

# Configure custom tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    padding_side="left",
    truncation_side="left"
)

# Use custom tokenizer
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer=tokenizer
)

framework = HuggingFaceFramework(config=config)
```

### 2. Memory Management

```python
# Configure memory efficient settings
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    offload_folder="offload",
    max_memory={0: "8GiB", 1: "8GiB", "cpu": "32GiB"}
)

framework = HuggingFaceFramework(config=config)
```

### 3. Multi-GPU Support

```python
# Configure multi-GPU settings
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="balanced",
    max_memory={
        0: "12GiB",  # GPU 0
        1: "12GiB",  # GPU 1
        "cpu": "32GiB"
    }
)

framework = HuggingFaceFramework(config=config)
```

## Best Practices

### 1. Model Selection Guidelines

```python
def select_appropriate_model(task_requirements):
    """Select appropriate model based on requirements."""
    if task_requirements.get("memory_constrained"):
        return "meta-llama/Llama-2-7b-chat-hf"
    elif task_requirements.get("high_performance"):
        return "meta-llama/Llama-2-13b-chat-hf"
    else:
        return "meta-llama/Llama-2-7b-chat-hf"
```

### 2. Resource Management

```python
import torch

def check_system_resources():
    """Check and report system resources."""
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "memory_allocated": torch.cuda.memory_allocated(),
        "memory_reserved": torch.cuda.memory_reserved()
    }
    return gpu_info

# Monitor resources
resources = check_system_resources()
print(f"GPU Memory Used: {resources['memory_allocated'] / 1024**3:.2f} GB")
```

### 3. Performance Optimization

```python
from utta.optimization import optimize_model_settings

def optimize_for_hardware():
    """Optimize model settings for available hardware."""
    if torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).total_memory > 8e9:  # 8GB
            return {
                "quantization": "8bit",
                "device_map": "auto"
            }
        else:
            return {
                "quantization": "4bit",
                "device_map": "auto"
            }
    return {"device": "cpu"}

# Apply optimizations
settings = optimize_for_hardware()
config = FrameworkConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    **settings
)
```

## Example Projects

### 1. Multi-Model Teaching Assistant

```python
class MultiModelTeacher:
    def __init__(self):
        self.models = {
            "small": HuggingFaceFramework(
                config=FrameworkConfig(
                    model_name="meta-llama/Llama-2-7b-chat-hf",
                    quantization="4bit"
                )
            ),
            "large": HuggingFaceFramework(
                config=FrameworkConfig(
                    model_name="meta-llama/Llama-2-13b-chat-hf",
                    quantization="8bit"
                )
            )
        }
    
    def get_response(self, question: str, complexity: str) -> str:
        model = self.models["large"] if complexity == "high" else self.models["small"]
        assistant = TeachingAssistant(framework=model)
        return assistant.answer(question)

# Usage
teacher = MultiModelTeacher()
response = teacher.get_response(
    "Explain quantum computing",
    complexity="high"
)
```

### 2. Resource-Aware Assistant

```python
class ResourceAwareAssistant:
    def __init__(self):
        self.config = self._get_optimal_config()
        self.framework = HuggingFaceFramework(config=self.config)
        self.assistant = TeachingAssistant(framework=self.framework)
    
    def _get_optimal_config(self):
        resources = check_system_resources()
        if resources["available"]:
            memory_gb = resources["memory_reserved"] / 1024**3
            if memory_gb > 16:
                return FrameworkConfig(
                    model_name="meta-llama/Llama-2-13b-chat-hf",
                    quantization="8bit"
                )
        return FrameworkConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            quantization="4bit"
        )
    
    def answer(self, question: str) -> str:
        try:
            return self.assistant.answer(question)
        except RuntimeError as e:
            # Fallback to smaller model if OOM
            self.config = FrameworkConfig(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                quantization="4bit"
            )
            self.framework = HuggingFaceFramework(config=self.config)
            self.assistant = TeachingAssistant(framework=self.framework)
            return self.assistant.answer(question)
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Use model quantization
   - Enable gradient checkpointing
   - Implement memory efficient attention
   ```python
   config = FrameworkConfig(
       model_name="meta-llama/Llama-2-7b-chat-hf",
       quantization="4bit",
       gradient_checkpointing=True,
       attention_implementation="flash"
   )
   ```

2. **Slow Inference**
   - Enable batch processing
   - Use optimized attention mechanisms
   - Implement caching
   ```python
   config = FrameworkConfig(
       model_name="meta-llama/Llama-2-7b-chat-hf",
       batch_size=4,
       use_cache=True,
       attention_implementation="flash"
   )
   ```

3. **Model Loading Issues**
   - Check HuggingFace token
   - Verify model compatibility
   - Monitor download progress
   ```python
   from huggingface_hub import HfApi
   
   def verify_model_access(model_name: str) -> bool:
       api = HfApi()
       try:
           api.model_info(model_name)
           return True
       except Exception as e:
           print(f"Error accessing model: {e}")
           return False
   ```

## Next Steps

1. Explore the [OpenAI Tutorial](OpenAI-Tutorial) for comparison
2. Learn about [Dataset Preparation](Dataset-Preparation)
3. Contribute to the [UTTA Project](Contributing) 