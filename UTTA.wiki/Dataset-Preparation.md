# Dataset Preparation Guide

This guide covers how to prepare and format datasets for training and evaluating UTTA models.

## Introduction

High-quality datasets are essential for effectively training and evaluating AI teaching assistants. This guide will help you understand how to create, format, and preprocess data for use with UTTA.

## Types of Datasets

### Educational Q&A Datasets

Educational question and answer datasets typically include:
- Student questions
- Expert answers
- Optional metadata (subject area, grade level, etc.)

### Teaching Dialogue Datasets

Conversational datasets modeling teacher-student interactions:
- Multi-turn dialogues
- Student questions/misconceptions
- Teacher explanations and guidance

### Assessment Datasets

Datasets for evaluating model performance:
- Questions with reference answers
- Rubrics for evaluation
- Multiple difficulty levels

## Data Format

UTTA supports several data formats:

### JSONL Format (Recommended)

```jsonl
{"question": "What is photosynthesis?", "answer": "Photosynthesis is the process used by plants to convert light energy into chemical energy...", "subject": "Biology", "grade_level": "High School"}
{"question": "How do I solve for x in 2x + 5 = 13?", "answer": "To solve for x, first subtract 5 from both sides: 2x = 8. Then divide both sides by 2: x = 4.", "subject": "Mathematics", "grade_level": "Middle School"}
```

### CSV Format

```csv
question,answer,subject,grade_level
"What is photosynthesis?","Photosynthesis is the process used by plants to convert light energy into chemical energy...","Biology","High School"
"How do I solve for x in 2x + 5 = 13?","To solve for x, first subtract 5 from both sides: 2x = 8. Then divide both sides by 2: x = 4.","Mathematics","Middle School"
```

### Conversation Format

For dialogue datasets:

```json
{
  "conversations": [
    {
      "turns": [
        {"role": "student", "content": "I don't understand why water is a polar molecule."},
        {"role": "teacher", "content": "Great question! Water (H₂O) is polar because of the uneven distribution of electrons..."},
        {"role": "student", "content": "So that's why oil and water don't mix?"},
        {"role": "teacher", "content": "Exactly! Oil molecules are non-polar, meaning..."}
      ],
      "metadata": {
        "subject": "Chemistry",
        "grade_level": "High School"
      }
    }
  ]
}
```

## Creating Educational Datasets

### From Existing Sources

Several educational datasets can be adapted for UTTA:
- [SciQ](https://allenai.org/data/sciq)
- [AI2 Reasoning Challenge (ARC)](https://allenai.org/data/arc)
- [OpenBookQA](https://allenai.org/data/open-book-qa)

### Generating with LLMs

You can use LLMs to generate educational content:

```python
from utta.data import DatasetGenerator

# Initialize a generator
generator = DatasetGenerator(model="openai/gpt-4")

# Generate a math Q&A dataset
math_dataset = generator.generate(
    subject="Mathematics",
    grade_levels=["Middle School", "High School"],
    topics=["Algebra", "Geometry", "Calculus"],
    num_examples=100
)

# Save the dataset
math_dataset.save("math_qa_dataset.jsonl")
```

### Human-in-the-Loop Data Collection

For highest quality:
1. Start with LLM-generated examples
2. Have educators review and edit
3. Collect real student questions
4. Iterate and expand the dataset

## Data Preprocessing

### Cleaning

```python
from utta.data import DataCleaner

# Initialize cleaner
cleaner = DataCleaner()

# Clean dataset
cleaned_data = cleaner.clean("raw_dataset.jsonl")

# Save cleaned data
cleaned_data.save("cleaned_dataset.jsonl")
```

### Augmentation

```python
from utta.data import DataAugmenter

# Initialize augmenter
augmenter = DataAugmenter()

# Augment dataset with paraphrased questions
augmented_data = augmenter.augment(
    "dataset.jsonl",
    methods=["paraphrase", "difficulty_variation"],
    multiplier=2
)

# Save augmented data
augmented_data.save("augmented_dataset.jsonl")
```

## Dataset Splitting

```python
from utta.data import DatasetSplitter

# Initialize splitter
splitter = DatasetSplitter()

# Split dataset
train, val, test = splitter.split(
    "full_dataset.jsonl",
    ratios=[0.7, 0.15, 0.15],
    stratify_by="subject"
)

# Save splits
train.save("train.jsonl")
val.save("val.jsonl")
test.save("test.jsonl")
```

## Dataset Analysis

```python
from utta.data import DatasetAnalyzer

# Initialize analyzer
analyzer = DatasetAnalyzer()

# Analyze dataset
stats = analyzer.analyze("dataset.jsonl")

# Print statistics
print(stats.summary())
print(stats.distributions())
print(stats.quality_issues())
```

## Best Practices

1. **Diverse Content**: Include a wide range of topics, difficulty levels, and question types
2. **Balance**: Ensure balanced representation across subjects and grade levels
3. **Quality Control**: Manually review samples for accuracy and educational value
4. **Metadata**: Include rich metadata to enable targeted training and evaluation
5. **Versioning**: Maintain clear dataset versions as you iterate and improve

## Example Workflow

1. Collect initial data from existing sources
2. Clean and preprocess the data
3. Augment with LLM-generated examples
4. Have educators review and refine
5. Split into train/val/test sets
6. Use for training and evaluation

## Resources

- [Sample Datasets Repository](https://github.com/UVU-AI-Innovate/UTTA/tree/main/datasets)
- [Data Preparation Scripts](https://github.com/UVU-AI-Innovate/UTTA/tree/main/tools/data_preparation)
- [Data Quality Guidelines](https://github.com/UVU-AI-Innovate/UTTA/wiki/Data-Quality-Guidelines) 