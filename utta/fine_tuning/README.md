# UTTA Fine-Tuning Module

This module provides tools for fine-tuning and optimizing language models for educational applications.

## Components

### OpenAIFineTuner

`OpenAIFineTuner` handles fine-tuning of OpenAI models:

- Data preparation and formatting
- Fine-tuning job submission and monitoring
- Fine-tuned model evaluation
- Job and model management

### DSPyOptimizer

`DSPyOptimizer` provides prompt optimization using DSPy:

- Bootstrap few-shot optimization
- Parameter-efficient tuning
- Evaluation of optimized prompts
- Saving and loading optimized modules

### HuggingFaceFineTuner

`HuggingFaceFineTuner` handles fine-tuning of open-source models from HuggingFace:

- Parameter-efficient fine-tuning with LoRA/QLoRA
- Training pipeline configuration
- Model saving and exporting
- Integration with Hugging Face's Accelerate library

## Usage

### OpenAI Fine-Tuning

```python
from utta.fine_tuning import OpenAIFineTuner

# Initialize the tuner
tuner = OpenAIFineTuner(base_model="gpt-3.5-turbo")

# Prepare data
data = [
    {"question": "What is differentiated instruction?", "answer": "Differentiated instruction is..."},
    # More examples...
]

# Start fine-tuning
job_id = tuner.start_fine_tuning("training_data.jsonl")
```

### DSPy Optimization

```python
from utta.fine_tuning import DSPyOptimizer
import dspy

# Define your DSPy module
class TeacherResponder(dspy.Module):
    def forward(self, question):
        return question

# Initialize the optimizer
optimizer = DSPyOptimizer(TeacherResponder())

# Optimize using examples
optimizer.optimize(train_data, num_iterations=3)
```

## Configuration

Each fine-tuning component can be configured through:

- Constructor parameters
- Environment variables (for API keys)
- Configuration files 