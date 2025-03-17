# Evaluation

This directory contains components for evaluating model outputs and performance.

## Purpose

The evaluation module provides:

1. Frameworks for evaluating LLM outputs
2. Methods for comparing model responses against reference materials
3. Tools for measuring pedagogical effectiveness
4. Utilities for batch evaluation across multiple examples

## Components

This directory works closely with the `metrics` directory, which contains more specific automated metrics for teaching responses. The evaluation module focuses on:

- Model performance evaluation
- Test set creation and management
- Automated evaluation pipelines
- Benchmarking against human evaluations

## Usage

Evaluation components can be used to:

1. Measure model improvement during fine-tuning
2. Compare different LLM configurations
3. Create and run test sets
4. Generate performance reports

### Usage Example

```python
from evaluation.evaluator import ModelEvaluator
from llm.dspy.handler import EnhancedDSPyLLMInterface

# Initialize model and evaluator
model = EnhancedDSPyLLMInterface()
evaluator = ModelEvaluator()

# Run evaluation on test set
results = evaluator.evaluate_model(
    model=model,
    test_set="data/evaluation/test_set.json",
    metrics=["clarity", "accuracy", "engagement"]
)

# Generate report
evaluator.generate_report(results, output_path="evaluation_results.json")
``` 