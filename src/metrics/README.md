# Metrics Module

This directory contains the components for evaluating the quality of responses provided by the UTTA system.

## Overview

The metrics module provides automated evaluation capabilities for assessing teaching responses based on various educational criteria such as clarity, engagement, and accuracy.

## Components

### AutomatedMetricsEvaluator

This class implements the functionality for assessing teaching responses and providing quantitative metrics:

- **Clarity Assessment**: Evaluates how clearly the content is explained for the target grade level
- **Engagement Metrics**: Measures how engaging and interesting the content is for students
- **Accuracy Analysis**: Verifies the factual correctness of the provided information
- **Age Appropriateness**: Ensures the language and concepts are appropriate for the target age group

## Usage

```python
from src.metrics.automated_metrics import AutomatedMetricsEvaluator

# Initialize the evaluator
evaluator = AutomatedMetricsEvaluator()

# Evaluate a response
metrics = evaluator.evaluate_response(
    response="Photosynthesis is like a plant's way of making food...",
    prompt="Explain photosynthesis to a 5th grader",
    grade_level=5,
    context=["Photosynthesis is the process by which plants use sunlight..."]
)

# Access the metrics
print(f"Clarity score: {metrics.clarity}")
print(f"Engagement score: {metrics.engagement}")
```

## Implementation Details

The evaluator uses LLM-based assessment to analyze responses along the defined metrics. It leverages the same LLM interface used for response generation but with specially crafted prompts designed to evaluate educational content.

When the LLM API is unavailable, the evaluator gracefully degrades to provide estimated metrics based on text statistics like readability scores, vocabulary complexity, and sentence structure. 