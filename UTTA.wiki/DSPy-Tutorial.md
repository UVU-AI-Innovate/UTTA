# DSPy Tutorial

This tutorial guides you through using DSPy with UTTA for creating and customizing teaching assistants.

## What is DSPy?

[DSPy](https://github.com/stanfordnlp/dspy) is a framework for solving complex language tasks by programming with foundation models. It provides:
- Declarative programming with language models
- Automatic prompt optimization
- Systematic evaluation and improvement

## Prerequisites

Before starting this tutorial, ensure you have:
1. Completed the [Environment Setup](Environment-Setup)
2. Basic understanding of Python and LLMs
3. UTTA installed and configured
4. OpenAI API key set up

## Basic Usage

### 1. Initialize DSPy with UTTA

```python
from utta.core import TeachingAssistant
from utta.frameworks.dspy_framework import DSPyFramework

# Initialize the framework
framework = DSPyFramework()

# Create teaching assistant
assistant = TeachingAssistant(framework=framework)
```

### 2. Define a Basic Module

```python
import dspy

class TeachingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        response = self.gen(question=question)
        return response.answer

# Use the module
module = TeachingModule()
result = module.forward("What is the capital of France?")
print(result)
```

## Advanced Features

### 1. Custom Prompts

```python
class CustomTeachingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("""
        Given a student's question about {topic}, provide:
        1. A clear explanation
        2. An example
        3. A practice question
        
        Question: {question}
        """)
    
    def forward(self, question, topic):
        response = self.gen(question=question, topic=topic)
        return response.answer

# Usage
custom_module = CustomTeachingModule()
result = custom_module.forward(
    question="How does inheritance work?",
    topic="Object-Oriented Programming"
)
```

### 2. Teleprompters

```python
# Define a teleprompter for automatic prompt optimization
teleprompter = dspy.Teleprompter()

# Train on example data
examples = [
    ("What is a variable?", "A variable is a container for storing data..."),
    ("Explain functions.", "A function is a reusable block of code...")
]

# Optimize prompts
teleprompter.train(examples)
```

### 3. Signature Chains

```python
class ComplexTeachingModule(dspy.Module):
    def __init__(self):
        super().__init__()
        
        # Define a chain of operations
        self.analyze = dspy.Predict("question -> difficulty, prerequisites")
        self.explain = dspy.ChainOfThought("question, difficulty -> explanation")
        self.example = dspy.Predict("explanation -> example")
    
    def forward(self, question):
        # Execute the chain
        analysis = self.analyze(question=question)
        explanation = self.explain(
            question=question,
            difficulty=analysis.difficulty
        )
        example = self.example(explanation=explanation.explanation)
        
        return {
            'explanation': explanation.explanation,
            'example': example.example,
            'difficulty': analysis.difficulty,
            'prerequisites': analysis.prerequisites
        }
```

## Integration with UTTA

### 1. Custom Framework Configuration

```python
from utta.frameworks.dspy_framework import DSPyFramework
from utta.config import FrameworkConfig

# Create custom configuration
config = FrameworkConfig(
    model_name="gpt-4",
    temperature=0.7,
    max_tokens=500
)

# Initialize framework with config
framework = DSPyFramework(config=config)
```

### 2. Evaluation and Metrics

```python
from utta.evaluation import Evaluator
from utta.metrics import AccuracyMetric, RelevanceMetric

# Create evaluator
evaluator = Evaluator([
    AccuracyMetric(),
    RelevanceMetric()
])

# Evaluate responses
results = evaluator.evaluate(
    predictions=[assistant.answer("What is Python?")],
    references=["Python is a high-level programming language..."]
)
```

## Best Practices

1. **Modular Design**
   - Break down complex tasks into smaller modules
   - Use clear and descriptive module names
   - Keep modules focused and single-purpose

2. **Prompt Engineering**
   - Start with simple prompts
   - Use teleprompters for optimization
   - Include examples in prompts when helpful

3. **Error Handling**
   ```python
   try:
       response = module.forward(question)
   except dspy.InputError as e:
       print(f"Invalid input: {e}")
   except dspy.ModelError as e:
       print(f"Model error: {e}")
   ```

4. **Performance Optimization**
   - Cache responses when appropriate
   - Use batching for multiple queries
   - Monitor token usage

## Example Projects

### 1. Basic Question-Answering

```python
class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, context, question):
        return self.qa(context=context, question=question).answer

# Usage
qa = QAModule()
context = "Python was created by Guido van Rossum..."
question = "Who created Python?"
answer = qa.forward(context, question)
```

### 2. Interactive Teaching Assistant

```python
class InteractiveTeacher(dspy.Module):
    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought("answer -> understanding_level")
        self.explain = dspy.ChainOfThought("question, understanding_level -> explanation")
    
    def forward(self, question, previous_answer=None):
        if previous_answer:
            understanding = self.assess(answer=previous_answer)
            level = understanding.understanding_level
        else:
            level = "beginner"
        
        return self.explain(
            question=question,
            understanding_level=level
        ).explanation
```

## Troubleshooting

### Common Issues

1. **Model Errors**
   - Check API key configuration
   - Verify model availability
   - Monitor rate limits

2. **Prompt Issues**
   - Review prompt format
   - Check for missing variables
   - Ensure clear instructions

3. **Integration Problems**
   - Verify UTTA version compatibility
   - Check framework configuration
   - Review error messages

## Next Steps

1. Explore the [OpenAI Tutorial](OpenAI-Tutorial) for comparison
2. Learn about [Dataset Preparation](Dataset-Preparation)
3. Contribute to the [UTTA Project](Contributing) 