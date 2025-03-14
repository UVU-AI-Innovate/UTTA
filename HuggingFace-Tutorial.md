# HuggingFace Tutorial

This tutorial will guide you through using HuggingFace models with the UTTA framework to create effective educational AI applications.

## Prerequisites

Before you begin, make sure you have:
- Completed the [Environment Setup](Environment-Setup)
- A HuggingFace account and API token (for accessing gated models)
- Basic understanding of transformer models

## Setting Up HuggingFace with UTTA

### API Token Configuration

If you're planning to use gated models that require authentication:

1. Create a HuggingFace account at [huggingface.co](https://huggingface.co)
2. Generate an API token from your account settings
3. Add your token to your environment:
   ```bash
   export HUGGINGFACE_API_KEY="your-huggingface-token"
   ```
   
   Or in your `.env` file:
   ```
   HUGGINGFACE_API_KEY=your-huggingface-token
   ```

### Installing Dependencies

Install the HuggingFace integration:

```bash
pip install utta[huggingface]
```

## Basic Usage with HuggingFace Models

### Creating a Simple Assistant with HuggingFace

```python
from utta import Assistant
from utta.models import HuggingFaceModel

# Initialize with a HuggingFace model
model = HuggingFaceModel(model_id="google/flan-t5-large")
assistant = Assistant(model=model)

# Ask a question
response = assistant.answer("Explain the concept of photosynthesis.")
print(response)
```

### Using Different Model Types

UTTA supports various HuggingFace model architectures:

```python
from utta.models import HuggingFaceModel

# Using a text generation model
llama_model = HuggingFaceModel(model_id="meta-llama/Llama-2-7b-chat-hf")

# Using a text-to-text model
t5_model = HuggingFaceModel(model_id="google/flan-t5-xl")

# Using a causal language model
gpt_neo_model = HuggingFaceModel(model_id="EleutherAI/gpt-neo-2.7B")
```

### Customizing Generation Parameters

```python
from utta.models import HuggingFaceModel

# Create a model with custom parameters
model = HuggingFaceModel(
    model_id="google/flan-t5-large",
    max_length=500,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2,
    num_return_sequences=1
)

assistant = Assistant(model=model)
```

## Advanced HuggingFace Features

### Using Specialized Educational Models

```python
from utta import Assistant
from utta.models import HuggingFaceModel

# Using a model fine-tuned for educational content
education_model = HuggingFaceModel(model_id="utta/t5-base-education-tuned")
edu_assistant = Assistant(model=education_model)

# Example for a math tutor
response = edu_assistant.answer("How do I solve quadratic equations?")
print(response)
```

### Implementing Domain-Specific Prompting

```python
from utta import Assistant
from utta.models import HuggingFaceModel
from utta.prompts import EducationalPromptTemplate

# Create a specialized prompt template for science education
science_template = EducationalPromptTemplate(
    template="""
    As a science educator, please explain the following concept in a way that's 
    appropriate for a {grade_level} student:
    
    Concept: {concept}
    
    Include:
    - A simple definition
    - Real-world examples
    - An analogy to help understand
    - A simple experiment they could try at home
    """,
    input_variables=["concept", "grade_level"]
)

# Initialize model
model = HuggingFaceModel(model_id="google/flan-t5-xl")

# Create assistant with the template
science_assistant = Assistant(
    model=model,
    prompt_template=science_template
)

# Use the assistant with template variables
response = science_assistant.answer(
    concept="Density",
    grade_level="5th grade"
)

print(response)
```

### Using Local Models

One advantage of HuggingFace is the ability to run models locally:

```python
from utta import Assistant
from utta.models import LocalHuggingFaceModel

# Use a model that will run on your local machine
local_model = LocalHuggingFaceModel(
    model_id="google/flan-t5-base",  # Smaller model that can run locally
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create an assistant with the local model
local_assistant = Assistant(model=local_model)

# This will run entirely on your machine without API calls
response = local_assistant.answer("Explain Newton's laws of motion.")
print(response)
```

## Creating Educational Applications

### Subject-Specific Tutor

```python
from utta import Assistant
from utta.models import HuggingFaceModel
from utta.prompts import TutorPromptTemplate

# Create a specialized history tutor
history_prompt = TutorPromptTemplate(
    subject="history",
    template="""
    As a history tutor, I want you to help the student understand:
    
    {question}
    
    Provide historical context, key dates, important figures, and the significance
    of this historical event or period. If relevant, mention different historical
    interpretations.
    """,
    input_variables=["question"]
)

# Initialize with an appropriate model
model = HuggingFaceModel(model_id="meta-llama/Llama-2-13b-chat-hf")

# Create the history tutor
history_tutor = Assistant(
    model=model,
    prompt_template=history_prompt
)

# Use the tutor
response = history_tutor.answer(question="What caused World War I?")
print(response)
```

### Quiz Generator

```python
from utta import QuizGenerator
from utta.models import HuggingFaceModel

# Initialize a quiz generator with a HuggingFace model
model = HuggingFaceModel(model_id="meta-llama/Llama-2-7b-chat-hf")
quiz_gen = QuizGenerator(model=model)

# Generate a quiz
quiz = quiz_gen.create_quiz(
    topic="Cell Biology",
    grade_level="high school",
    num_questions=5,
    question_types=["multiple_choice", "true_false"]
)

# Display the quiz
for i, question in enumerate(quiz.questions):
    print(f"Q{i+1}: {question.text}")
    if question.type == "multiple_choice":
        for j, option in enumerate(question.options):
            print(f"  {chr(65+j)}) {option}")
    print(f"Answer: {question.answer}\n")
```

## Performance Optimization

### Quantization for Local Models

```python
from utta import Assistant
from utta.models import LocalHuggingFaceModel

# Use a quantized model for better performance on local hardware
model = LocalHuggingFaceModel(
    model_id="TheBloke/Llama-2-7B-Chat-GGML",
    quantization="4bit",  # Options include 4bit, 8bit depending on model
    context_length=2048,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

assistant = Assistant(model=model)
```

### Batching for Multiple Questions

```python
from utta import Assistant
from utta.models import HuggingFaceModel

model = HuggingFaceModel(model_id="google/flan-t5-large")
assistant = Assistant(model=model)

# Process multiple questions efficiently
questions = [
    "What is photosynthesis?",
    "Explain the water cycle.",
    "How do batteries work?"
]

# Process in a batch
responses = assistant.batch_answer(questions)

for question, response in zip(questions, responses):
    print(f"Q: {question}")
    print(f"A: {response}")
    print()
```

## Best Practices

1. **Select the right model size**:
   - Larger models (>7B parameters) for complex educational tasks
   - Smaller models (<2B parameters) for simpler tasks or resource-constrained environments

2. **Consider specialized models**:
   - Subject-specific fine-tuned models often outperform general-purpose models
   - Instruction-tuned models (like Flan-T5) tend to follow educational prompts better

3. **Balance quality and speed**:
   - For interactive tutoring, prioritize response time
   - For content generation, prioritize quality and accuracy

4. **Quantization considerations**:
   - 4-bit quantization: Fastest, but may reduce quality
   - 8-bit quantization: Good balance of speed and quality
   - 16-bit or 32-bit: Highest quality but resource intensive

## Further Resources

- [HuggingFace Documentation](https://huggingface.co/docs)
- [UTTA Model Reference](link-to-utta-model-docs)
- [HuggingFace Model Hub](https://huggingface.co/models)

For other model integrations, see:
- [OpenAI Tutorial](OpenAI-Tutorial)
- [DSPy Tutorial](DSPy-Tutorial) 