# OpenAI Tutorial

This tutorial will guide you through using OpenAI models with the UTTA framework for creating effective educational assistants and tools.

## Prerequisites

Before you begin, make sure you have:
- Completed the [Environment Setup](Environment-Setup)
- An OpenAI API key
- Basic understanding of prompt engineering

## Setting Up OpenAI with UTTA

### API Key Configuration

First, ensure your OpenAI API key is properly configured:

1. Add your API key to your environment variables:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

2. Or add it to a `.env` file in your project root:
   ```
   OPENAI_API_KEY=your-openai-api-key
   ```

3. Load the environment variables in your code:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### Installing Dependencies

Make sure you have the necessary dependencies:

```bash
pip install utta[openai]
```

## Basic Usage with OpenAI Models

### Creating a Simple Teaching Assistant

```python
from utta import Assistant
from utta.models import OpenAIModel

# Initialize with a specific OpenAI model
model = OpenAIModel(model_name="gpt-4")
assistant = Assistant(model=model)

# Ask a question
response = assistant.answer("What is the Pythagorean theorem?")
print(response)
```

### Customizing OpenAI Parameters

You can customize various parameters for OpenAI models:

```python
from utta.models import OpenAIModel

# Create a model with custom parameters
model = OpenAIModel(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=500,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.0
)

assistant = Assistant(model=model)
```

## Advanced OpenAI Features

### Using System Messages for Educational Contexts

```python
from utta import Assistant
from utta.models import OpenAIModel

# Create a model with educational system prompt
model = OpenAIModel(
    model_name="gpt-4",
    system_message="You are an expert mathematics tutor helping high school students. \
                   Provide clear, step-by-step explanations and include examples \
                   that illustrate key concepts. Encourage critical thinking and \
                   guide students to solutions rather than providing answers directly."
)

math_tutor = Assistant(model=model)
response = math_tutor.answer("How do I solve systems of linear equations?")
print(response)
```

### Function Calling for Structured Educational Outputs

```python
from utta import Assistant
from utta.models import OpenAIModel
from utta.tools import MathProblemSolver, ConceptDiagramGenerator

# Define functions that the model can call
functions = [
    {
        "name": "solve_math_problem",
        "description": "Solves a mathematical problem step by step",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The math problem to solve"
                },
                "grade_level": {
                    "type": "string",
                    "description": "The grade level of the student (elementary, middle, high, college)"
                }
            },
            "required": ["problem"]
        }
    },
    {
        "name": "generate_concept_diagram",
        "description": "Generates a description for a diagram explaining a concept",
        "parameters": {
            "type": "object",
            "properties": {
                "concept": {
                    "type": "string",
                    "description": "The concept to explain visually"
                },
                "complexity": {
                    "type": "string",
                    "enum": ["basic", "intermediate", "advanced"],
                    "description": "The complexity level of the diagram"
                }
            },
            "required": ["concept"]
        }
    }
]

# Create model with function calling
model = OpenAIModel(
    model_name="gpt-4",
    functions=functions,
    function_call="auto"
)

# Initialize the assistant with tools
assistant = Assistant(
    model=model,
    tools={
        "solve_math_problem": MathProblemSolver(),
        "generate_concept_diagram": ConceptDiagramGenerator()
    }
)

# The assistant will automatically use function calling when appropriate
response = assistant.answer("Can you help me solve 2x + 5 = 13?")
print(response)
```

### Using OpenAI's Vision Models for Educational Content

```python
from utta import Assistant
from utta.models import OpenAIVisionModel
import base64

# Function to encode image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load the image
image_data = encode_image("path/to/math_problem.jpg")

# Create vision model
vision_model = OpenAIVisionModel(model_name="gpt-4-vision-preview")

# Create assistant with vision capabilities
vision_assistant = Assistant(model=vision_model)

# Analyze an image containing educational content
response = vision_assistant.analyze_image(
    image=image_data,
    prompt="This is a photo of a math problem. Can you solve it step by step?"
)

print(response)
```

## Best Practices for Educational Applications

1. **Use appropriate temperature settings**:
   - Lower (0.0-0.3) for factual questions and problem-solving
   - Medium (0.3-0.7) for explanations and examples
   - Higher (0.7-1.0) for creative educational content

2. **Select the right model for the task**:
   - `gpt-4` for complex problems and nuanced explanations
   - `gpt-3.5-turbo` for general educational content and faster responses
   - Vision models for analyzing diagrams, equations, and educational images

3. **Craft effective system messages**:
   - Specify the educational role (tutor, teacher, coach)
   - Define the target audience's educational level
   - Outline the appropriate tone and teaching style
   - Include relevant pedagogical approaches

4. **Handle model uncertainty appropriately**:
   - Teach students to critically evaluate information
   - Allow the model to express uncertainty when appropriate
   - Implement citation or reference requirements

## Troubleshooting Common Issues

- **Rate Limiting**: Implement appropriate retries and backoff strategies
- **Token Limits**: Break complex educational content into manageable chunks
- **Cost Management**: Use simpler models for initial interactions, reserve GPT-4 for complex tasks

## Example Educational Applications

### Interactive Math Tutor

```python
from utta import Assistant
from utta.models import OpenAIModel
from utta.tools import MathProblemSolver

# Create a specialized math tutor
math_tutor = Assistant(
    model=OpenAIModel(
        model_name="gpt-4",
        system_message="You are a patient and encouraging math tutor who specializes in helping students understand concepts, not just get answers."
    ),
    tools={"solve_math_problem": MathProblemSolver()}
)

def tutor_session():
    print("Math Tutor: Hi! I'm your math tutor. What would you like help with today?")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Math Tutor: Goodbye! Keep practicing!")
            break
            
        response = math_tutor.answer(user_input)
        print(f"Math Tutor: {response}")

# Run the tutor session
if __name__ == "__main__":
    tutor_session()
```

## Further Resources

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [UTTA Model Reference](link-to-utta-model-docs)
- [Educational Prompt Engineering Guide](link-to-prompt-engineering-guide)

For other model integrations, see:
- [HuggingFace Tutorial](HuggingFace-Tutorial)
- [DSPy Tutorial](DSPy-Tutorial) 