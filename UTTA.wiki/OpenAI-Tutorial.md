# OpenAI Tutorial for UTTA

This tutorial will guide you through using OpenAI models with the UTTA framework.

## Introduction

OpenAI's models like GPT-3.5 and GPT-4 provide powerful capabilities for educational applications. This guide will show you how to effectively leverage these models within UTTA for teaching and training purposes.

## Prerequisites

Before starting this tutorial, ensure you have:

* Completed the [Environment Setup](Environment-Setup)
* An OpenAI API key
* Basic understanding of the UTTA framework

## Setting Up OpenAI Access

1. **Create an OpenAI Account**
   * Sign up at [OpenAI](https://platform.openai.com/)
   * Navigate to the API section

2. **Generate an API Key**
   * Go to "API Keys" in your account
   * Click "Create new secret key"
   * Copy and store your key securely

3. **Set Up Environment Variable**
   ```bash
   # Add to your environment or .env file
   export OPENAI_API_KEY="your-key-here"
   ```

## Basic Usage of OpenAI Models

### Initializing with OpenAI Models

```python
from utta import TeachingAssistant

# Initialize with GPT-3.5-Turbo
assistant = TeachingAssistant(model="openai/gpt-3.5-turbo")

# Or initialize with GPT-4
advanced_assistant = TeachingAssistant(model="openai/gpt-4")

# Ask a question
response = assistant.ask("What are the key principles of machine learning?")
print(response)
```

### Configuring Model Parameters

```python
# Configure with specific parameters
assistant = TeachingAssistant(
    model="openai/gpt-4",
    temperature=0.7,
    max_tokens=500,
    top_p=0.95
)

# You can also update parameters after initialization
assistant.update_model_parameters(temperature=0.5)
```

## Advanced OpenAI Features

### System Messages for Role Definition

```python
# Define a specific teaching persona
assistant = TeachingAssistant(
    model="openai/gpt-3.5-turbo",
    system_message="You are a university professor specializing in computer science. Explain concepts thoroughly with relevant examples and ask students questions to check understanding."
)
```

### Function Calling

```python
from utta.tools import MathTool, WebSearchTool

# Add tools to your assistant
assistant = TeachingAssistant(model="openai/gpt-4")
assistant.add_tool(MathTool())
assistant.add_tool(WebSearchTool())

# Now the assistant can use these tools
response = assistant.ask("Calculate the derivative of x^3 + 2x^2 - 5x + 3")
print(response)
```

### Using Different OpenAI Models

UTTA supports different OpenAI models:

```python
# Using the most recent GPT-4 model
assistant_gpt4 = TeachingAssistant(model="openai/gpt-4")

# Using GPT-3.5-Turbo for faster, more cost-effective responses
assistant_gpt35 = TeachingAssistant(model="openai/gpt-3.5-turbo")

# Using a specific model version
assistant_specific = TeachingAssistant(model="openai/gpt-4-0613")
```

## Educational Applications

### Creating Personalized Explanations

```python
def generate_explanation(concept, grade_level, learning_style):
    prompt = f"Explain {concept} to a {grade_level} student with a {learning_style} learning style."
    
    assistant = TeachingAssistant(model="openai/gpt-4")
    response = assistant.ask(prompt)
    
    return response

# Generate explanations for different students
visual_learner = generate_explanation("photosynthesis", "8th grade", "visual")
auditory_learner = generate_explanation("photosynthesis", "8th grade", "auditory")
```

### Creating Interactive Quizzes

```python
from utta.education import QuizGenerator

# Initialize a quiz generator
quiz_gen = QuizGenerator(model="openai/gpt-4")

# Generate a quiz on a specific topic
quiz = quiz_gen.create_quiz(
    topic="World War II",
    difficulty="high school",
    num_questions=5,
    question_types=["multiple_choice", "short_answer"]
)

# Print the quiz
for i, question in enumerate(quiz.questions):
    print(f"Q{i+1}: {question.text}")
    if question.type == "multiple_choice":
        for j, option in enumerate(question.options):
            print(f"  {chr(65+j)}) {option}")
```

## Cost Management

OpenAI API usage incurs costs. Here's how to manage them:

```python
from utta.utils import TokenCounter

# Initialize a token counter
counter = TokenCounter()

# Check token usage for a prompt
prompt = "Explain the concept of photosynthesis in detail."
tokens = counter.count_tokens(prompt, model="openai/gpt-4")
print(f"This prompt will use approximately {tokens} tokens.")

# Estimate cost
estimated_cost = counter.estimate_cost(tokens, model="openai/gpt-4")
print(f"Estimated cost: ${estimated_cost:.4f}")
```

## Evaluating Responses

```python
from utta.evaluation import ResponseEvaluator

# Initialize an evaluator
evaluator = ResponseEvaluator()

# Evaluate a response against reference material
score = evaluator.evaluate(
    model_response="Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
    reference="Photosynthesis is the process by which plants, algae, and some bacteria convert light energy, usually from the sun, into chemical energy in the form of glucose or other sugars.",
    criteria=["accuracy", "completeness", "clarity"]
)

print(f"Evaluation score: {score}")
print(f"Feedback: {evaluator.get_feedback()}")
```

## Fine-Tuning OpenAI Models

For advanced customization, you can fine-tune OpenAI models:

```python
from utta.fine_tuning import OpenAIFineTuner

# Initialize the fine-tuner
tuner = OpenAIFineTuner(
    base_model="gpt-3.5-turbo",
    dataset_path="education_examples.jsonl"
)

# Start the fine-tuning process
job_id = tuner.start_fine_tuning()
print(f"Fine-tuning job started with ID: {job_id}")

# Check status
status = tuner.check_status(job_id)
print(f"Status: {status}")

# When complete, use the fine-tuned model
assistant = TeachingAssistant(model=f"openai/{tuner.get_model_name()}")
```

## Best Practices

1. **Be Specific in Prompts**: Clear instructions lead to better responses
2. **Use System Messages**: Set the educational context and expectations
3. **Start with GPT-3.5**: Use the more cost-effective model for development
4. **Monitor Token Usage**: Keep track of costs, especially with larger models
5. **Implement Safeguards**: Add checks to ensure content is appropriate for educational use
6. **Verify Information**: Always verify factual accuracy of model outputs

## Troubleshooting

* **Rate Limits**: If you encounter rate limits, implement exponential backoff
* **Token Limits**: Break long prompts into smaller chunks
* **Cost Management**: Set budget alerts and limitations in your application
* **Version Changes**: Stay updated on model version changes

## Further Resources

* [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
* [UTTA Documentation](Home)
* [OpenAI Model Capabilities](https://platform.openai.com/docs/models) 