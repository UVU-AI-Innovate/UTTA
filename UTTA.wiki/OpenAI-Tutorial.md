# OpenAI Tutorial

This tutorial guides you through using OpenAI's models with UTTA for creating effective teaching assistants.

## Prerequisites

Before starting this tutorial, ensure you have:
1. Completed the [Environment Setup](Environment-Setup)
2. OpenAI API key configured
3. Basic understanding of Python and LLMs
4. UTTA installed and configured

## Basic Usage

### 1. Initialize OpenAI with UTTA

```python
from utta.core import TeachingAssistant
from utta.frameworks.openai_framework import OpenAIFramework

# Initialize the framework
framework = OpenAIFramework()

# Create teaching assistant
assistant = TeachingAssistant(framework=framework)
```

### 2. Simple Question-Answering

```python
# Ask a question
response = assistant.answer("What is object-oriented programming?")
print(response)

# Ask with context
context = """
Object-oriented programming (OOP) is a programming paradigm based on the concept of objects,
which can contain data and code. The data is in the form of fields (often known as attributes),
and the code is in the form of procedures (often known as methods).
"""

response = assistant.answer(
    "What are the main components of OOP?",
    context=context
)
print(response)
```

## Advanced Features

### 1. Custom Configuration

```python
from utta.config import FrameworkConfig

# Create custom configuration
config = FrameworkConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.0,
    presence_penalty=0.0
)

# Initialize framework with config
framework = OpenAIFramework(config=config)
assistant = TeachingAssistant(framework=framework)
```

### 2. System Messages

```python
# Define teaching style
system_message = """
You are a patient and knowledgeable computer science teacher.
Always:
1. Start with basic concepts
2. Provide relevant examples
3. Include practice exercises
4. Encourage critical thinking
"""

# Create assistant with custom system message
assistant = TeachingAssistant(
    framework=framework,
    system_message=system_message
)
```

### 3. Function Calling

```python
from typing import List, Dict

def generate_practice_problems(topic: str, difficulty: str) -> List[Dict]:
    """Generate practice problems for a given topic."""
    response = assistant.framework.call_function(
        "generate_problems",
        {
            "topic": topic,
            "difficulty": difficulty,
            "num_problems": 3
        }
    )
    return response["problems"]

# Usage
problems = generate_practice_problems("Python Lists", "intermediate")
for problem in problems:
    print(f"Question: {problem['question']}")
    print(f"Answer: {problem['answer']}\n")
```

## Integration with UTTA

### 1. Custom Prompts

```python
from utta.prompts import PromptTemplate

# Create custom prompt template
teaching_prompt = PromptTemplate(
    template="""
    Topic: {topic}
    Student Level: {level}
    Question: {question}
    
    Please provide:
    1. A clear explanation
    2. Relevant examples
    3. Common misconceptions
    4. Practice exercises
    """
)

# Use custom prompt
response = assistant.answer(
    question="How do Python decorators work?",
    prompt_template=teaching_prompt,
    topic="Python Advanced Features",
    level="intermediate"
)
```

### 2. Conversation Management

```python
# Start a teaching session
session = assistant.start_session()

# Have a conversation
responses = []
responses.append(session.send("What is a variable in Python?"))
responses.append(session.send("Can you give me an example?"))
responses.append(session.send("What about different data types?"))

# Review conversation history
history = session.get_history()
for message in history:
    print(f"{message['role']}: {message['content']}\n")
```

### 3. Error Handling

```python
from utta.exceptions import ModelError, RateLimitError

try:
    response = assistant.answer("Complex question about quantum computing")
except ModelError as e:
    print(f"Model error: {e}")
except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")
    # Implement retry logic or fallback
```

## Best Practices

### 1. Token Management

```python
from utta.utils import count_tokens

# Check token count before sending
question = "Long and complex question..."
token_count = count_tokens(question)

if token_count > 2000:
    # Split into smaller chunks or summarize
    pass
```

### 2. Cost Optimization

```python
# Use cheaper models for simple tasks
config_simple = FrameworkConfig(
    model_name="gpt-3.5-turbo",
    max_tokens=100
)

# Use more powerful models for complex tasks
config_complex = FrameworkConfig(
    model_name="gpt-4",
    max_tokens=500
)

# Switch based on task complexity
def get_appropriate_config(question: str) -> FrameworkConfig:
    complexity = assess_complexity(question)
    return config_complex if complexity > 0.7 else config_simple
```

### 3. Response Validation

```python
from utta.validation import validate_response

def ensure_quality_response(response: str) -> bool:
    checks = [
        "contains_example",
        "appropriate_length",
        "relevant_content"
    ]
    
    return validate_response(response, checks)
```

## Example Projects

### 1. Interactive Coding Tutor

```python
class CodingTutor:
    def __init__(self):
        self.assistant = TeachingAssistant(framework=OpenAIFramework())
        self.session = self.assistant.start_session()
    
    def review_code(self, code: str) -> str:
        prompt = f"""
        Please review this code:
        ```python
        {code}
        ```
        Provide:
        1. Code quality assessment
        2. Potential improvements
        3. Best practices suggestions
        """
        return self.session.send(prompt)
    
    def explain_error(self, error_message: str) -> str:
        return self.session.send(
            f"Please explain this error and how to fix it: {error_message}"
        )

# Usage
tutor = CodingTutor()
code = "def factorial(n): return 1 if n <= 1 else n*factorial(n-1)"
review = tutor.review_code(code)
```

### 2. Concept Mapper

```python
class ConceptMapper:
    def __init__(self):
        self.assistant = TeachingAssistant(framework=OpenAIFramework())
    
    def create_concept_map(self, topic: str) -> dict:
        prompt = f"""
        Create a concept map for: {topic}
        Include:
        1. Main concepts
        2. Relationships between concepts
        3. Prerequisites
        4. Advanced applications
        """
        response = self.assistant.answer(prompt)
        return self.parse_concept_map(response)
    
    def parse_concept_map(self, response: str) -> dict:
        # Implementation to convert response to structured format
        pass

# Usage
mapper = ConceptMapper()
python_concepts = mapper.create_concept_map("Python Object-Oriented Programming")
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**
   - Implement exponential backoff
   - Use token bucket rate limiting
   - Monitor usage patterns

2. **Response Quality**
   - Adjust temperature and top_p
   - Refine system messages
   - Implement validation checks

3. **Cost Management**
   - Monitor token usage
   - Use appropriate models
   - Implement caching

## Next Steps

1. Explore the [DSPy Tutorial](DSPy-Tutorial) for comparison
2. Learn about [Dataset Preparation](Dataset-Preparation)
3. Contribute to the [UTTA Project](Contributing) 