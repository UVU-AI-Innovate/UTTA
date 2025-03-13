# LLM Evaluation Framework

The LLM evaluation framework provides comprehensive tools for assessing chatbot performance through automated metrics, human evaluation, and systematic benchmarking.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Components](#components)
  - [Automated Metrics](#automated-metrics)
  - [Human Evaluation](#human-evaluation)
  - [Benchmark Suite](#benchmark-suite)
- [Usage Examples](#usage-examples)
- [Best Practices](#best-practices)

## Overview

The evaluation framework consists of three main components:
1. **Automated Metrics**: Quantitative evaluation using industry-standard metrics
2. **Human Evaluation**: Structured qualitative assessment through human annotators
3. **Benchmark Suite**: Systematic testing with predefined test cases

## Installation

Ensure you have the required dependencies installed:

```bash
pip install rouge-score bert-score sacrebleu nltk pyyaml numpy
```

These dependencies are already included in the project's `requirements.txt`.

## Components

### Automated Metrics

The `AutomatedMetrics` class provides the following metrics:

- **ROUGE Scores**: Measures overlap between generated and reference texts
  - ROUGE-1: Unigram overlap
  - ROUGE-2: Bigram overlap
  - ROUGE-L: Longest common subsequence
- **BERTScore**: Semantic similarity using BERT embeddings
- **BLEU**: Translation quality metric
- **METEOR**: Machine translation evaluation metric

Example usage:

```python
from evaluation.metrics.automated_metrics import AutomatedMetrics

metrics = AutomatedMetrics()
scores = metrics.evaluate_response(
    generated_response="Hello! How can I help you today?",
    reference_response="Hi! How may I assist you?"
)

print(scores)
# Output: {
#     'rouge1_f1': 0.75,
#     'rouge2_f1': 0.5,
#     'rougeL_f1': 0.75,
#     'bert_score': 0.85,
#     'bleu': 0.65,
#     'meteor': 0.70
# }
```

### Human Evaluation

The `HumanEvaluation` class facilitates structured human assessment:

Default evaluation criteria:
1. Response Relevance (1-5)
2. Response Accuracy (1-5)
3. Language Quality (1-5)
4. Task Completion (1-5)
5. Overall Quality (1-5)

Example usage:

```python
from evaluation.human.human_evaluation import HumanEvaluation

evaluator = HumanEvaluation()

# Create evaluation form
form = evaluator.create_evaluation_form([
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level programming language..."}
])

# Save evaluation scores
evaluator.save_evaluation(
    evaluator_id="evaluator_1",
    form_data=form,
    scores={
        "Response Relevance (1-5)": 5,
        "Response Accuracy (1-5)": 4,
        "Language Quality (1-5)": 5,
        "Task Completion (1-5)": 4,
        "Overall Quality (1-5)": 4
    },
    comments="Good explanation, could include more examples"
)

# Aggregate multiple evaluations
aggregate_scores = evaluator.aggregate_evaluations(form["conversation_id"])
```

### Benchmark Suite

The `BenchmarkSuite` class enables systematic testing using predefined test cases:

Example usage:

```python
from evaluation.benchmarks.benchmark_suite import BenchmarkSuite

benchmark = BenchmarkSuite()

# Create a new benchmark set
test_cases = [
    {
        "input": "What is machine learning?",
        "expected_output": "Machine learning is a subset of artificial intelligence...",
        "category": "technical_explanation"
    }
]
benchmark.create_benchmark_set(test_cases, "ml_concepts_benchmark")

# Run benchmarks
results = benchmark.run_benchmark(
    chatbot=your_chatbot_instance,
    benchmark_name="ml_concepts_benchmark",
    output_file="ml_benchmark_results.json"
)

# Compare multiple benchmark runs
comparison = benchmark.compare_benchmarks([
    "ml_benchmark_results_v1.json",
    "ml_benchmark_results_v2.json"
])
```

## Best Practices

1. **Automated Metrics**
   - Use multiple metrics for comprehensive evaluation
   - Consider domain-specific metrics when applicable
   - Track metrics over time to measure improvements

2. **Human Evaluation**
   - Use multiple evaluators for each conversation
   - Provide clear scoring guidelines
   - Include qualitative feedback through comments
   - Regularly analyze aggregated results

3. **Benchmarking**
   - Create diverse test cases covering different scenarios
   - Include edge cases and error conditions
   - Maintain separate benchmark sets for different capabilities
   - Version control your benchmark sets

4. **General Tips**
   - Combine automated and human evaluation for best results
   - Document evaluation criteria and procedures
   - Regularly update benchmark sets with new test cases
   - Track and analyze trends over time 