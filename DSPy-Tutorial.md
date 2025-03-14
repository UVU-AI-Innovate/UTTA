# DSPy Tutorial for UTTA

This tutorial will guide you through using DSPy for prompt optimization in UTTA.

## What is DSPy?

[DSPy](https://github.com/stanfordnlp/dspy) is a framework for programming with foundation models, developed by Stanford AI Lab. In UTTA, we use DSPy for:

1. Optimizing prompts for educational tasks
2. Creating consistent response patterns
3. Improving the quality and accuracy of model outputs

## Prerequisites

Before starting this tutorial, make sure you have:

* Completed the [Environment Setup](Environment-Setup)
* Basic understanding of prompt engineering
* Basic knowledge of Python

## Installation

DSPy is included in the UTTA requirements, but if you need to install it separately:

```bash
pip install dspy-ai
```

## Basic DSPy Usage in UTTA

### 1. Simple DSPy Integration

```python
from utta import TeachingAssistant
from utta.optimization import DSPyOptimizer

# Initialize the teaching assistant
assistant = TeachingAssistant(model="openai/gpt-3.5-turbo")

# Create a DSPy optimizer
optimizer = DSPyOptimizer()

# Define a simple educational task
task = {
    "description": "Explain complex concepts to high school students",
    "examples": [
        {"input": "What is quantum computing?", "output": "Quantum computing uses quantum bits or 'qubits' that can exist in multiple states at once, unlike regular computer bits that are either 0 or 1. This allows quantum computers to solve certain problems much faster..."}
    ]
}

# Optimize the assistant for the task
optimizer.optimize(assistant, task)

# Now use the optimized assistant
response = assistant.ask("Can you explain black holes in simple terms?")
print(response)
```

### 2. Creating a DSPy Module

For more control, you can create custom DSPy modules:

```python
import dspy

# Define a custom module for educational explanations
class EducationalExplainer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.gen = dspy.ChainOfThought("question -> explanation")
    
    def forward(self, question):
        # Generate a thoughtful explanation
        explanation = self.gen(question=question).explanation
        return explanation

# Use it in UTTA
from utta import TeachingAssistant
from utta.optimization import DSPyModuleIntegrator

# Create and optimize the module
explainer = EducationalExplainer()
integrator = DSPyModuleIntegrator()
integrator.integrate(assistant, explainer)

# Use the enhanced assistant
response = assistant.ask("How does photosynthesis work?")
print(response)
```

## Advanced DSPy Optimization

### Teleprompters for Educational Content

DSPy's teleprompters are powerful tools for optimizing prompts based on example data:

```python
import dspy
from dspy.teleprompt import BootstrapFewShot

# Prepare example data
examples = [
    dspy.Example(
        question="What causes seasons on Earth?",
        explanation="Seasons are caused by Earth's tilted axis as it orbits the Sun. This tilt makes different parts of Earth receive more direct sunlight at different times of the year, creating seasons."
    ),
    # Add more examples...
]

# Create a teleprompter
teleprompter = BootstrapFewShot(metric=dspy.metrics.Answer())

# Optimize your DSPy module
optimized_explainer = teleprompter.compile(
    EducationalExplainer(),
    train_data=examples,
    num_trials=10
)

# Integrate with UTTA
assistant.integrate_dspy_module(optimized_explainer)
```

### Creating Multi-step Educational Workflows

For complex educational tasks, you can create multi-step workflows:

```python
class EducationalWorkflow(dspy.Module):
    def __init__(self):
        super().__init__()
        # Assess the student's knowledge level
        self.assess = dspy.Predict("question -> knowledge_level")
        # Generate an appropriate explanation
        self.explain = dspy.ChainOfThought("question, knowledge_level -> explanation")
        # Create follow-up questions
        self.follow_up = dspy.Predict("question, explanation -> follow_up_questions")
    
    def forward(self, question):
        # Assess knowledge level
        knowledge = self.assess(question=question).knowledge_level
        # Generate explanation
        explanation = self.explain(question=question, knowledge_level=knowledge).explanation
        # Create follow-up questions
        follow_ups = self.follow_up(question=question, explanation=explanation).follow_up_questions
        
        return {
            "knowledge_level": knowledge,
            "explanation": explanation,
            "follow_up_questions": follow_ups
        }
```

## Evaluation and Iteration

To evaluate and improve your DSPy modules:

```python
# Create evaluation data
eval_data = [
    dspy.Example(question="How do vaccines work?"),
    # Add more evaluation examples...
]

# Define evaluation metrics
metrics = [
    dspy.metrics.Accuracy(),
    dspy.metrics.InformationRecall()
]

# Run evaluation
results = dspy.evaluate(optimized_explainer, eval_data, metrics)
print(results)
```

## Integration with UTTA's Evaluation System

```python
from utta.evaluation import Evaluator

# Create an evaluator
evaluator = Evaluator()

# Add your DSPy-optimized assistant
evaluator.add_model(assistant, name="DSPy-Optimized")

# Add a baseline model for comparison
baseline = TeachingAssistant(model="openai/gpt-3.5-turbo")
evaluator.add_model(baseline, name="Baseline")

# Run evaluation
results = evaluator.evaluate(task="explanation", dataset="education_examples.jsonl")
print(results)
```

## Best Practices

1. **Start Small**: Begin with simple modules before building complex workflows
2. **Use Real Examples**: Collect real educational exchanges for training data
3. **Iterate**: Continuously evaluate and refine your DSPy modules
4. **Combine Approaches**: Mix DSPy optimization with other techniques like fine-tuning
5. **Focus on Educational Goals**: Tailor your modules to specific learning objectives

## Common Issues and Solutions

* **Overfitting**: If your module works well on training data but not new queries, add more diverse examples
* **Inconsistent Output Format**: Use structured prediction modules like `dspy.TypedPredictor`
* **Poor Performance**: Try different teleprompters or increase the number of optimization trials

## Advanced Topics

* [DSPy Compiler Options](https://github.com/stanfordnlp/dspy#advanced-compiler-options)
* [Custom Metrics Development](https://github.com/stanfordnlp/dspy#metrics)
* [Integration with Other LLM Frameworks](https://github.com/stanfordnlp/dspy#integrations)

## Resources

* [Official DSPy Documentation](https://github.com/stanfordnlp/dspy)
* [Stanford AI Lab Research](https://ai.stanford.edu/research/)
* [UTTA Documentation](Home) 