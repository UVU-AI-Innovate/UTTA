# Fine-Tuning

This directory contains components for fine-tuning language models for pedagogical applications.

## Purpose

The fine-tuning module provides tools to optimize language models for educational contexts by:

1. Creating specialized training datasets
2. Training models with pedagogical examples
3. Optimizing for teacher-like responses
4. Evaluating improvements in teaching capability

## Components

### Pedagogical Trainer

The `PedagogicalTrainer` class (`trainer.py`) provides:

- Training data preparation from various sources
- DSPy-based optimization of language models
- Custom evaluation metrics for teaching quality
- Model persistence for trained models

## Key Features

- **DSPy Integration**: Uses DSPy's optimization frameworks to fine-tune models
- **Training Data Management**: Prepares and processes training examples
- **Pedagogical Evaluation**: Custom metrics focused on teaching effectiveness
- **Context Integration**: Training examples include relevant context materials
- **Custom Signatures**: DSPy signatures optimized for pedagogical tasks

### Usage Example

```python
from fine_tuning.trainer import PedagogicalTrainer

# Initialize trainer
trainer = PedagogicalTrainer(model_name="gpt-3.5-turbo")

# Prepare training data
training_data = trainer.prepare_data("data/training_examples/")

# Train the model
optimized_model = trainer.train(
    training_data=training_data,
    num_iterations=5
)

# Use the optimized model
response = optimized_model.generate(
    prompt="Explain photosynthesis to a 3rd grader",
    context=["Photosynthesis is how plants make food using sunlight."]
)
```

## Implementation Details

The fine-tuning module uses:
- DSPy ChainOfThoughtOptimizer for model optimization
- Custom evaluation metrics for pedagogical quality
- Structured training examples with context
- Efficient model serialization for persistence 