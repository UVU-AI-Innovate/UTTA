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

## How DSPy Fine-Tuning Works

The system uses DSPy's prompt optimization approach rather than traditional weight-based fine-tuning:

1. **Chain of Thought Optimization**: Uses `ChainOfThoughtOptimizer` to improve prompts through iterative testing
   - Runs for 3 optimization rounds by default
   - Tests multiple prompt variations to find the most effective pedagogical patterns
   - Preserves the base model weights while significantly enhancing teaching quality

2. **Teaching Signature Definition**: Creates a structured input/output signature:
   ```python
   class TeachingSignature(dspy.Signature):
       query = dspy.InputField()  # Student question
       context = dspy.InputField() # Reference material
       metadata = dspy.InputField() # Teaching context
       response = dspy.OutputField() # Teaching response
   ```

3. **Pedagogical Evaluation**: During optimization, responses are evaluated on:
   - **Length Ratio**: Ensuring appropriate response length (30% of score)
   - **Context Usage**: Measuring factual alignment with reference materials (40% of score)
   - **Pedagogical Elements**: Counting teaching techniques like questions, examples, explanations (30% of score)

4. **Optimization Process Flow**:
   - Create training signatures from example dialogues
   - Optimize prompts through multiple rounds of testing
   - Evaluate each candidate prompt on pedagogical metrics
   - Save the optimized model for inference

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

## Evaluation During Training

During the optimization process, candidate prompts are evaluated on:

1. **Context Usage Evaluation**:
   - Calculates word overlap between response and reference context
   - Ensures factual accuracy and relevant information inclusion
   - Weights this metric higher (40%) as factual correctness is critical

2. **Pedagogical Elements Count**:
   - Identifies key pedagogical markers:
     - Questions: "?", "how", "what", "why"
     - Examples: "for example", "like", "such as"
     - Explanations: "because", "therefore"
     - Engagement: "let's", "try", "imagine"
   - Normalizes scores across categories

3. **Combined Training Metrics**:
   - Weighted average of individual metrics to guide optimization
   - Ensures balance between factual accuracy and teaching quality
   - Optimizes for maximum pedagogical effectiveness while preserving information accuracy 