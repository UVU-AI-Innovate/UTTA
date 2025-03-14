# DSPy Tutorial

This tutorial will guide you through using the DSPy framework with UTTA for optimizing language model prompts and building powerful educational applications.

## What is DSPy?

[DSPy](https://github.com/stanfordnlp/dspy) is a framework developed by Stanford NLP that provides a systematic way to optimize language model prompting. It treats prompt engineering as a programming problem rather than a trial-and-error process, allowing for more reliable and optimized prompts.

## Installation

If you haven't already set up UTTA with DSPy support, run:

```bash
pip install utta[dspy]
```

Or install DSPy separately:

```bash
pip install dspy-ai
```

## Basic DSPy Concepts

DSPy revolves around a few core concepts:
- **Signatures**: Define the inputs and outputs of language model calls
- **Modules**: Reusable components that implement specific functionalities
- **Pipelines**: Sequences of modules that work together
- **Teleprompters**: Optimize prompts automatically based on examples

## Using DSPy with UTTA

UTTA provides integrations with DSPy to help create optimized educational assistants. Here's how to use them:

### Creating a Basic Educational Assistant with DSPy

```python
import dspy
from utta.dspy_integration import EducationalSignature, TeachingModule

# Configure DSPy with your preferred LM
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo"))

# Define a signature for explanations
class ExplainConcept(EducationalSignature):
    """Explain a concept at the appropriate educational level."""
    concept = dspy.InputField()
    educational_level = dspy.InputField(desc="The student's educational level (elementary, middle, high, undergraduate, graduate)")
    explanation = dspy.OutputField()

# Create a teaching module
explainer = TeachingModule(ExplainConcept)

# Use the module
result = explainer(concept="Photosynthesis", educational_level="middle")
print(result.explanation)
```

### Optimizing Prompts with Examples

```python
import dspy
from utta.dspy_integration import EducationalOptimizer

# Define some examples
examples = [
    {"concept": "Gravity", "educational_level": "elementary", "explanation": "Gravity is what pulls things down to the Earth. It's like an invisible hand that pulls everything towards the ground. That's why when you drop something, it falls down instead of floating away."},
    {"concept": "Gravity", "educational_level": "undergraduate", "explanation": "Gravity is a fundamental force described by Newton's law of universal gravitation and Einstein's theory of general relativity. It's the force that attracts objects with mass toward each other, proportional to the product of their masses and inversely proportional to the square of the distance between them."}
]

# Create and train the optimizer
optimizer = EducationalOptimizer(ExplainConcept, examples)
optimized_explainer = optimizer.optimize()

# Use the optimized module
result = optimized_explainer(concept="Cellular Respiration", educational_level="high")
print(result.explanation)
```

## Advanced DSPy Features for Education

### Creating a Multi-Step Educational Pipeline

```python
import dspy
from utta.dspy_integration import AssessmentSignature, FeedbackSignature

# Define signatures for a complete teaching pipeline
class GenerateQuestion(EducationalSignature):
    """Generate a question to test understanding of a concept."""
    concept = dspy.InputField()
    difficulty = dspy.InputField(desc="easy, medium, or hard")
    question = dspy.OutputField()
    answer = dspy.OutputField()

class ProvideFeedback(FeedbackSignature):
    """Provide helpful feedback on a student's answer."""
    question = dspy.InputField()
    correct_answer = dspy.InputField()
    student_answer = dspy.InputField()
    feedback = dspy.OutputField()
    
# Create a complete pipeline
class TeachingPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.question_generator = dspy.Predict(GenerateQuestion)
        self.feedback_provider = dspy.Predict(ProvideFeedback)
    
    def forward(self, concept, difficulty, student_answer):
        # Generate a question and correct answer
        qa_result = self.question_generator(concept=concept, difficulty=difficulty)
        
        # Provide feedback on the student's answer
        feedback_result = self.feedback_provider(
            question=qa_result.question,
            correct_answer=qa_result.answer,
            student_answer=student_answer
        )
        
        return dspy.Prediction(
            question=qa_result.question,
            correct_answer=qa_result.answer,
            feedback=feedback_result.feedback
        )

# Use the pipeline
teaching_pipeline = TeachingPipeline()
result = teaching_pipeline(
    concept="Photosynthesis",
    difficulty="medium",
    student_answer="It's the process where plants make energy from sunlight."
)

print(f"Question: {result.question}")
print(f"Correct Answer: {result.correct_answer}")
print(f"Feedback: {result.feedback}")
```

## Best Practices

1. **Use domain-specific examples**: The quality of your optimization depends on the quality of examples
2. **Educational levels matter**: Be explicit about educational levels in your signatures
3. **Chain modules thoughtfully**: Break complex educational tasks into logical steps
4. **Evaluate regularly**: Test your optimized prompts with real educational use cases

## Further Resources

- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [UTTA DSPy Integration Reference](link-to-utta-dspy-docs)

For more information on how to use UTTA with various frameworks, check out our other tutorials:
- [OpenAI Tutorial](OpenAI-Tutorial)
- [HuggingFace Tutorial](HuggingFace-Tutorial) 