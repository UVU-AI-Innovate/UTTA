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

## Comprehensive Evaluation Metrics

The system implements the `AutomatedMetricsEvaluator` class with multiple assessment dimensions:

### 1. Clarity Score (0-1.0)
Evaluates how understandable the teaching response is:
- **Readability**: Uses Flesch Reading Ease score to assess text complexity
- **Sentence Structure**: Analyzes average sentence length (optimum: 15-25 words)
- **Vocabulary Complexity**: Identifies and penalizes excessive use of complex words
- **Calculation**: Combined score of readability, length, and complexity metrics

### 2. Engagement Score (0-1.0)
Measures how well the response encourages student interaction:
- **Engagement Markers**: Counts phrases like "let's try", "imagine", "think about"
- **Question Frequency**: Tracks the number of questions posed to the student
- **Calculation**: Average of normalized engagement marker and question scores

### 3. Context Alignment Score (0-1.0)
Assesses alignment between response and reference materials:
- **Semantic Similarity**: Uses sentence embeddings to calculate cosine similarity
- **Ensures factual correctness and relevant information usage
- **Calculation**: Direct cosine similarity between response and context embeddings

### 4. Age-Appropriate Score (0-1.0)
Determines if the response matches the student's grade level:
- **Target Grade Detection**: Extracts grade level from student profile
- **Reading Level Analysis**: Calculates text complexity using established metrics
- **Calculation**: 1.0 - min(abs(reading_level - target_grade) / target_grade, 1.0)

### 5. Pedagogical Elements Score (0-1.0)
Evaluates teaching techniques present in the response:
- **Engagement**: Phrases that activate student thinking
- **Scaffolding**: Sequential structuring of concepts
- **Examples**: Concrete illustrations of abstract concepts
- **Understanding Checks**: Verification of student comprehension
- **Calculation**: Average score across all pedagogical categories

## Evaluation Process

The evaluation process follows these steps:

1. **Response Generation**: Produce teaching responses with the model
2. **Multi-Dimensional Scoring**: Apply all metrics to evaluate responses
3. **Overall Score Calculation**: Weighted average of individual metrics
4. **Feedback Generation**: Automated suggestions for improvement
5. **Comparative Analysis**: Before/after comparison for fine-tuning effectiveness

### Evaluation Output Format

The evaluation produces a structured report like this:

```json
{
  "overall_score": 0.78,
  "clarity_score": 0.82,
  "engagement_score": 0.75,
  "context_score": 0.85,
  "age_appropriate_score": 0.79,
  "pedagogical_score": 0.70,
  "feedback": [
    "Add more interactive elements like questions",
    "Include more scaffolding examples"
  ]
}
```

## Visual Reporting

Evaluation results are displayed with:

1. **Comparative Tables**: Side-by-side before/after metrics
2. **Score Visualizations**: Bar charts showing improvement across dimensions
3. **Detailed Breakdown**: Per-category analysis of teaching quality
4. **Improvement Highlights**: Emphasis on areas of significant enhancement

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