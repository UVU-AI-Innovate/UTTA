# Dataset Preparation

This guide covers the preparation and formatting of datasets for use with the UTTA framework, particularly for training and evaluating educational AI models.

## Types of Educational Datasets

UTTA supports several types of educational datasets:

1. **Question-Answer Pairs**: For training assistants to answer educational questions
2. **Explanations**: For teaching complex concepts at various educational levels
3. **Feedback Examples**: For learning to provide constructive feedback on student work
4. **Dialogues**: Multi-turn educational conversations between teachers and students
5. **Educational Content**: Examples of lessons, lectures, and educational materials

## Dataset Format Requirements

### Basic JSON Format

The most common format for UTTA datasets is JSONL (JSON Lines), where each line is a valid JSON object:

```json
{"question": "What is photosynthesis?", "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy..."}
{"question": "What causes the seasons?", "answer": "The seasons are caused by Earth's tilted axis as it orbits the sun..."}
```

### Structured Data Format

For more complex educational scenarios, use structured data:

```json
{
  "id": "biology-101",
  "metadata": {
    "subject": "Biology",
    "grade_level": "high school",
    "difficulty": "intermediate"
  },
  "content": {
    "question": "Explain the difference between mitosis and meiosis.",
    "exemplar_answer": "Mitosis and meiosis are both processes of cell division, but they differ in several key ways. Mitosis produces two genetically identical daughter cells and is used for growth and repair. Meiosis produces four genetically diverse cells with half the chromosome count, used in sexual reproduction..."
  }
}
```

### Dialogue Format

For conversational educational data:

```json
{
  "id": "math-tutoring-1",
  "metadata": {
    "subject": "Mathematics",
    "topic": "Algebra",
    "grade_level": "middle school"
  },
  "conversation": [
    {"role": "student", "content": "I don't understand how to solve for x in 2x + 5 = 13."},
    {"role": "teacher", "content": "Let's break this down step by step. We want to isolate x on one side of the equation. First, what should we do with the 5?"},
    {"role": "student", "content": "Subtract 5 from both sides?"},
    {"role": "teacher", "content": "Exactly! So we get 2x = 8. Now what's the next step?"},
    {"role": "student", "content": "Divide both sides by 2?"},
    {"role": "teacher", "content": "Perfect! So x = 4. Always remember, whatever operation you do to one side of the equation, you must do to the other."}
  ]
}
```

## Creating Educational Datasets

### From Existing Materials

1. **Convert textbooks and lecture materials**:
   ```python
   from utta.data import TextbookConverter
   
   converter = TextbookConverter()
   dataset = converter.from_pdf("biology_textbook.pdf", sections=True)
   dataset.save("biology_dataset.jsonl")
   ```

2. **Extract from educational videos**:
   ```python
   from utta.data import VideoTranscriptExtractor
   
   extractor = VideoTranscriptExtractor()
   dataset = extractor.from_youtube("VIDEO_ID", segment=True)
   dataset.save("video_lessons.jsonl")
   ```

### Manual Creation

For highest quality training data, create examples manually with educational experts:

1. Define the educational objectives
2. Create a diverse set of examples covering different aspects
3. Ensure appropriate difficulty levels and educational standards
4. Include metadata about subject, grade level, etc.

### Using the UTTA Dataset Generator

```python
from utta import Assistant
from utta.data import DatasetGenerator

# Create a dataset generator with a powerful model
generator = DatasetGenerator(model="openai/gpt-4")

# Generate a dataset of math word problems
dataset = generator.generate(
    topic="Algebra Word Problems",
    grade_levels=["elementary", "middle", "high"],
    num_examples=50,
    template="Create a word problem about {topic} for {grade_level} students."
)

# Save the dataset
dataset.save("algebra_word_problems.jsonl")
```

## Data Augmentation

To increase dataset size and diversity:

```python
from utta.data import DataAugmenter

# Initialize the augmenter
augmenter = DataAugmenter()

# Load your base dataset
base_dataset = augmenter.load("science_questions.jsonl")

# Apply augmentation techniques
augmented_dataset = augmenter.augment(
    base_dataset,
    techniques=["paraphrase", "difficulty_variation", "format_change"],
    scale_factor=3  # Triple the dataset size
)

# Save the augmented dataset
augmented_dataset.save("augmented_science_questions.jsonl")
```

## Data Preprocessing

### Cleaning Educational Data

```python
from utta.data import DataCleaner

# Initialize the cleaner
cleaner = DataCleaner()

# Load a dataset
dataset = cleaner.load("raw_educational_data.jsonl")

# Clean the dataset
cleaned_dataset = cleaner.clean(
    dataset,
    operations=[
        "remove_duplicates",
        "fix_formatting",
        "standardize_notation",
        "check_factual_accuracy"
    ]
)

# Save the cleaned dataset
cleaned_dataset.save("clean_educational_data.jsonl")
```

### Formatting for Specific Models

```python
from utta.data import DataFormatter

# Initialize the formatter
formatter = DataFormatter()

# Load your dataset
dataset = formatter.load("educational_qa.jsonl")

# Format for specific model types
openai_dataset = formatter.format_for_model(dataset, model_type="openai")
hf_dataset = formatter.format_for_model(dataset, model_type="huggingface")

# Save formatted datasets
openai_dataset.save("openai_educational_qa.jsonl")
hf_dataset.save("hf_educational_qa.jsonl")
```

## Dataset Splitting and Validation

```python
from utta.data import DatasetSplitter

# Initialize the splitter
splitter = DatasetSplitter()

# Load your dataset
dataset = splitter.load("educational_content.jsonl")

# Split into train, validation, and test sets
splits = splitter.split(
    dataset,
    ratios=[0.7, 0.15, 0.15],  # train, validation, test
    stratify_by="subject"  # Ensure balanced subjects in each split
)

# Save the splits
splits["train"].save("edu_content_train.jsonl")
splits["validation"].save("edu_content_val.jsonl")
splits["test"].save("edu_content_test.jsonl")
```

## Dataset Quality Assessment

```python
from utta.data import QualityChecker

# Initialize the quality checker
checker = QualityChecker()

# Load your dataset
dataset = checker.load("math_problems.jsonl")

# Run quality checks
quality_report = checker.evaluate(
    dataset,
    checks=[
        "completeness",
        "consistency",
        "educational_value",
        "accuracy",
        "grade_level_appropriateness"
    ]
)

# View the report
print(quality_report.summary())
print(quality_report.recommendations())
```

## Best Practices

1. **Educational Alignment**: Ensure content matches educational standards and curricula
2. **Diversity**: Include a wide range of difficulties, topics, and pedagogical approaches
3. **Metadata Rich**: Add detailed metadata about subject, grade level, learning objectives
4. **Balanced Representation**: Ensure diverse cultural and social perspectives
5. **Expert Validation**: Have educational experts review dataset content for accuracy
6. **Privacy Protection**: Remove any personally identifiable information from student data
7. **Iterative Improvement**: Continuously update and refine datasets based on model performance

## Sample Educational Datasets

UTTA provides several sample datasets to get started:

- `utta.datasets.math_qa`: Mathematical questions and answers
- `utta.datasets.science_explanations`: Scientific concept explanations
- `utta.datasets.history_essays`: History essay prompts with example responses
- `utta.datasets.language_feedback`: Language learning exercises with feedback examples

To use them:

```python
from utta.datasets import math_qa

# Load the dataset
dataset = math_qa.load()

# Explore the dataset
print(f"Dataset size: {len(dataset)}")
print(f"Sample entry: {dataset[0]}")

# Use for training
from utta.training import Trainer
trainer = Trainer(dataset=dataset)
```

## Further Resources

- [Educational Data Standards](link-to-educational-standards)
- [Sample Datasets Repository](link-to-repository)
- [Data Augmentation Techniques](link-to-augmentation-guide)

For more information on how to use datasets with UTTA, see:
- [DSPy Tutorial](DSPy-Tutorial)
- [OpenAI Tutorial](OpenAI-Tutorial)
- [HuggingFace Tutorial](HuggingFace-Tutorial) 