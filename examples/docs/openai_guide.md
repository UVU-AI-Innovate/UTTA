# OpenAI Integration Guide for Educational Applications

## Quick Start

1. Install required packages:
```bash
pip install openai python-dotenv
```

2. Set up your API key:
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
```

## Core Features

### 1. Teaching Assistant
```python
def get_educational_response(question, subject, grade_level):
    """Generate educational responses for students."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"You are an expert {subject} teacher for {grade_level} students."},
            {"role": "user", "content": question}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content
```

### 2. Student Assessment
```python
def assess_student_response(student_answer, correct_answer):
    """Evaluate student responses and provide feedback."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an experienced educational assessor."},
            {"role": "user", "content": f"Evaluate this answer:\n{student_answer}\nCorrect answer:\n{correct_answer}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content
```

## Dataset Management

### Dataset Types

1. **In-Code Examples** (in `openai_finetune.py`):
   - Basic teaching examples (`SAMPLE_DIALOGUES`)
   - Complex scenarios (`ADDITIONAL_DIALOGUES`)
   - Test cases (`TEST_SCENARIOS`)

2. **External Files** (in `datasets/` directory):
   - Training data (`train.jsonl`)
   - Validation data (`val.jsonl`)
   - Subject-specific files (in `subjects/` subdirectories)

### Adding New Data

1. **Simple Method**: Add to existing files
```python
# In openai_finetune.py
SAMPLE_DIALOGUES.append({
    "messages": [
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": "Your question"},
        {"role": "assistant", "content": "Your response"}
    ]
})
```

2. **File Method**: Create new JSONL files
```plaintext
datasets/
└── subjects/
    └── your_subject/
        └── your_dataset.jsonl
```

Example format:
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user", "content": "Student question"},
        {"role": "assistant", "content": "Teacher response"}
    ],
    "metadata": {
        "subject": "math",
        "difficulty": "beginner",
        "tags": ["algebra"]
    }
}
```

3. **Using DatasetManager**:
```python
from openai_finetune import DatasetManager

dataset = DatasetManager()
dataset.add_dialogues_from_file("your_dataset.jsonl")
dataset.save_to_file("train.jsonl")
```

## Best Practices

1. **Data Quality**:
   - Use accurate content
   - Write clear explanations
   - Include real examples
   - Test before adding

2. **Cost Management**:
   - Start with small datasets
   - Use GPT-3.5-Turbo for basic tasks
   - Use GPT-4 for complex explanations
   - Cache common responses

## Resources

- [OpenAI Documentation](https://platform.openai.com/docs)
- [Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Example Code](../openai_finetune.py) 