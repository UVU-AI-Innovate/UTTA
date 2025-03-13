# Dataset Preparation Guide

This guide explains how to prepare and format datasets for training and evaluating UTTA teaching assistants.

## Overview

UTTA supports various dataset formats and sources for:
- Training custom teaching assistants
- Evaluating teaching effectiveness
- Fine-tuning language models
- Testing response quality

## Dataset Structure

### Basic Format

```json
{
    "conversations": [
        {
            "id": "conv_001",
            "messages": [
                {
                    "role": "student",
                    "content": "What is a variable in Python?"
                },
                {
                    "role": "assistant",
                    "content": "A variable is a container that stores a value..."
                }
            ],
            "metadata": {
                "topic": "Python Basics",
                "difficulty": "beginner"
            }
        }
    ]
}
```

### Extended Format

```json
{
    "conversations": [
        {
            "id": "conv_002",
            "messages": [
                {
                    "role": "student",
                    "content": "How do I use a for loop?",
                    "context": {
                        "prior_knowledge": ["variables", "basic syntax"],
                        "learning_style": "visual"
                    }
                },
                {
                    "role": "assistant",
                    "content": "A for loop is used to iterate over a sequence...",
                    "examples": [
                        {
                            "code": "for i in range(5):\n    print(i)",
                            "output": "0\n1\n2\n3\n4"
                        }
                    ],
                    "follow_up_questions": [
                        "What about while loops?",
                        "Can you nest loops?"
                    ]
                }
            ],
            "metadata": {
                "topic": "Control Flow",
                "difficulty": "beginner",
                "tags": ["loops", "iteration", "basics"]
            }
        }
    ]
}
```

## Data Collection

### 1. Manual Collection

```python
from utta.data import DataCollector

# Initialize collector
collector = DataCollector()

# Add conversation
collector.add_conversation(
    student_question="What is inheritance in OOP?",
    assistant_response="Inheritance is a mechanism that allows a class...",
    metadata={
        "topic": "Object-Oriented Programming",
        "difficulty": "intermediate"
    }
)

# Save dataset
collector.save("oop_dataset.json")
```

### 2. Automated Collection

```python
from utta.data import AutomaticCollector
from utta.sources import StackOverflowSource, DocumentationSource

# Initialize collector with sources
collector = AutomaticCollector([
    StackOverflowSource(tags=["python", "beginner"]),
    DocumentationSource(url="https://docs.python.org/3/")
])

# Collect data
dataset = collector.collect(
    topics=["variables", "functions", "classes"],
    num_examples=100
)

# Save dataset
dataset.save("python_basics_dataset.json")
```

### 3. Data Augmentation

```python
from utta.data import DataAugmenter

# Initialize augmenter
augmenter = DataAugmenter()

# Load existing dataset
dataset = augmenter.load("original_dataset.json")

# Apply augmentations
augmented_dataset = augmenter.augment(
    dataset,
    techniques=[
        "paraphrase",
        "add_examples",
        "vary_difficulty"
    ]
)

# Save augmented dataset
augmented_dataset.save("augmented_dataset.json")
```

## Data Preprocessing

### 1. Basic Preprocessing

```python
from utta.preprocessing import Preprocessor

# Initialize preprocessor
preprocessor = Preprocessor()

# Load and preprocess dataset
dataset = preprocessor.load("raw_dataset.json")
processed_dataset = preprocessor.process(
    dataset,
    steps=[
        "remove_duplicates",
        "clean_text",
        "validate_format"
    ]
)

# Save processed dataset
processed_dataset.save("processed_dataset.json")
```

### 2. Advanced Preprocessing

```python
from utta.preprocessing import AdvancedPreprocessor
from utta.validation import DataValidator

# Initialize preprocessor with custom settings
preprocessor = AdvancedPreprocessor(
    text_cleaner_config={
        "remove_html": True,
        "fix_unicode": True,
        "standardize_quotes": True
    },
    validator_config={
        "min_length": 50,
        "max_length": 1000,
        "required_fields": ["content", "metadata"]
    }
)

# Process dataset
processed_dataset = preprocessor.process(
    "raw_dataset.json",
    additional_steps=[
        "normalize_difficulty_levels",
        "extract_code_snippets",
        "categorize_topics"
    ]
)
```

## Data Validation

### 1. Schema Validation

```python
from utta.validation import SchemaValidator

# Define schema
schema = {
    "type": "object",
    "properties": {
        "conversations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "messages", "metadata"]
            }
        }
    }
}

# Validate dataset
validator = SchemaValidator(schema)
is_valid = validator.validate("dataset.json")
```

### 2. Content Quality Checks

```python
from utta.validation import QualityChecker

# Initialize checker
checker = QualityChecker()

# Check dataset quality
quality_report = checker.check(
    "dataset.json",
    checks=[
        "spelling",
        "grammar",
        "code_validity",
        "content_relevance"
    ]
)

# Print report
print(quality_report.summary())
```

## Dataset Splitting

### 1. Basic Splitting

```python
from utta.data import DatasetSplitter

# Initialize splitter
splitter = DatasetSplitter()

# Split dataset
train, val, test = splitter.split(
    "dataset.json",
    ratios=[0.7, 0.15, 0.15]
)

# Save splits
train.save("train.json")
val.save("val.json")
test.save("test.json")
```

### 2. Stratified Splitting

```python
from utta.data import StratifiedSplitter

# Initialize splitter
splitter = StratifiedSplitter(
    stratify_by=["difficulty", "topic"]
)

# Split dataset
splits = splitter.split(
    "dataset.json",
    ratios=[0.7, 0.15, 0.15],
    ensure_representation=True
)
```

## Best Practices

### 1. Data Quality

- Ensure consistent formatting
- Validate all entries
- Remove duplicate content
- Balance topic distribution
- Include diverse difficulty levels

### 2. Metadata

- Add detailed topic tags
- Include difficulty levels
- Mark prerequisites
- Add source information
- Include timestamps

### 3. Content Guidelines

- Keep responses focused
- Include practical examples
- Add follow-up questions
- Maintain consistent style
- Include code snippets where relevant

## Example Workflow

```python
from utta.data import DataPipeline

# Define pipeline
pipeline = DataPipeline([
    ("collect", AutomaticCollector()),
    ("preprocess", Preprocessor()),
    ("augment", DataAugmenter()),
    ("validate", QualityChecker()),
    ("split", DatasetSplitter())
])

# Configure pipeline
config = {
    "collect": {
        "sources": ["stackoverflow", "documentation"],
        "num_examples": 1000
    },
    "preprocess": {
        "steps": ["clean", "deduplicate", "normalize"]
    },
    "augment": {
        "techniques": ["paraphrase", "add_examples"]
    },
    "validate": {
        "checks": ["quality", "format", "content"]
    },
    "split": {
        "ratios": [0.7, 0.15, 0.15]
    }
}

# Run pipeline
train, val, test = pipeline.run(config)
```

## Next Steps

1. Learn how to use your dataset in the [DSPy Tutorial](DSPy-Tutorial)
2. Explore model training in the [OpenAI Tutorial](OpenAI-Tutorial)
3. Contribute to the [UTTA Project](Contributing) 