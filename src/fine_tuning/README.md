# Fine-Tuning

This directory contains components for fine-tuning language models for pedagogical applications.

## Purpose

The fine-tuning module provides tools to optimize language models for educational contexts by:

1. Creating specialized training datasets
2. Training models with pedagogical examples
3. Optimizing for teacher-like responses
4. Evaluating improvements in teaching capability

## Components

### Pedagogical Trainer

The `PedagogicalTrainer` class (`trainer.py`) provides:

- Training data preparation from various sources
- DSPy-based optimization of language models
- Custom evaluation metrics for teaching quality
- Model persistence for trained models

## What is DSPy Fine-Tuning?

DSPy fine-tuning represents a fundamentally different approach from traditional model fine-tuning:

- **Traditional Fine-Tuning**: Updates the model's weights through gradient descent to adapt the entire model's behavior, requiring significant computational resources and large datasets.
  
- **DSPy Prompt Optimization**: Instead of changing model weights, DSPy optimizes the prompts used to communicate with the model. This approach is more efficient, requires less data, and works with models you access through APIs.

DSPy is a framework that enables building LLM applications with sophisticated prompting patterns, including chains of thought, multi-step reasoning, and self-refinement. The framework treats these patterns as programmatic modules that can be optimized through search rather than static templates.

## Comparing DSPy to Other Fine-Tuning Approaches

| Feature | DSPy | OpenAI Fine-Tuning | HuggingFace LoRA |
|---------|------|-------------------|------------------|
| **What changes** | Prompts | Model weights (cloud) | Model weights (local) |
| **Training data needed** | Small (10-50 examples) | Medium (50-100+ examples) | Large (100-1000+ examples) |
| **Cost structure** | Pay per API call | Training fee + inference | One-time compute |
| **Technical complexity** | Low | Medium | High |
| **Control** | Limited | Medium | Full |
| **Data privacy** | Shared with API | Shared with API | Stays local |
| **Time to results** | Minutes | Hours | Hours to days |
| **Typical use case** | Quick experiments, limited data | Production with moderate data | Full control, large data |

## Technical Architecture

DSPy's architecture has several key components:

1. **Signatures**: Define the structure of inputs and outputs for language model interactions
   ```python
   class TeachingSignature(dspy.Signature):
       query = dspy.InputField()  # Student question
       context = dspy.InputField() # Reference material
       metadata = dspy.InputField() # Teaching context
       response = dspy.OutputField() # Teaching response
   ```

2. **Modules**: Reusable components that implement specific prompting patterns
   - `Predict`: Basic prompting for direct answers
   - `ChainOfThought`: Encourages step-by-step reasoning
   - `Retrieve`: Information retrieval module
   - `KeyphraseSelector`: Extracts important phrases

3. **Programs**: Compositions of modules that define complex tasks
   ```python
   # Example of a DSPy program for teaching responses
   class TeacherProgram(dspy.Module):
       def __init__(self):
           self.reasoner = dspy.ChainOfThought(TeachingSignature)
           
       def forward(self, query, context, metadata):
           # Generate teaching response with reasoning
           response = self.reasoner(query=query, context=context, metadata=metadata)
           return response
   ```

4. **Teleprompters**: Optimizers that learn to improve prompts through experimentation
   - Test different prompt formulations
   - Evaluate outcomes using custom metrics
   - Select the best performing prompt variations

## Optimization Process

Our implementation uses DSPy's `ChainOfThoughtOptimizer` with a systematic workflow:

1. **Initialization Phase**:
   - Define a teaching signature with structured inputs and outputs
   - Create a ChainOfThought module using this signature
   - Prepare a dataset of exemplar question-context-response triples

2. **Prompt Generation Phase**:
   - Generate multiple candidate prompts for each template position
   - Emphasize different aspects of teaching (engagement, examples, explanations)
   - Explore prompt space methodically, not randomly

3. **Evaluation Phase**:
   - Test each candidate prompt on the training examples
   - Evaluate generated responses using pedagogical metrics
   - Calculate a weighted score to determine prompt quality

4. **Selection and Refinement Phase**:
   - Select the best performing prompts for the next round
   - Learn from successful patterns and refine them
   - Repeat for the specified number of rounds (default: 3)

5. **Compilation Phase**:
   - Compile the final optimal prompts into the LLM interaction template
   - Encode pedagogical knowledge and teaching strategies in the prompts
   - Save the compiled template for inference

### Optimization Algorithm

The DSPy optimizer uses a sophisticated evolutionary approach to improve prompts:

```
ALGORITHM OptimizeTeachingPrompts:
    Input: Examples E, initial module M, metric function f
    Output: Optimized module M'
    
    1. Initialize candidates C ← {M}
    2. For r = 1 to MAX_ROUNDS:
        3. Generate P new prompts for each template position
        4. For each new prompt template t:
            5. Test t on examples E
            6. Compute score s = f(predictions, targets)
            7. Add (t, s) to candidates C
        8. Select top-k best performing candidates
        9. Extract common patterns from successful candidates
        10. Generate new candidates emphasizing successful patterns
    11. Return best candidate M' from C
```

This approach allows DSPy to discover prompts that encode pedagogical expertise without changing model weights.

## Evaluation Metrics

Our implementation uses several metrics to assess teaching quality during optimization:

### 1. Context Usage Evaluation (40% of Score)

This metric measures how effectively the model incorporates reference materials:

```python
def _evaluate_context_usage(self, response: str, context: str) -> float:
    """Evaluate how well the response uses the given context."""
    # Simple word overlap metric
    response_words = set(response.lower().split())
    context_words = set(context.lower().split())
    
    overlap = len(response_words.intersection(context_words))
    total = len(context_words)
    
    return overlap / total if total > 0 else 0.0
```

This word overlap approach ensures the model uses information from the reference materials while generating responses.

### 2. Pedagogical Elements Detection (30% of Score)

This metric identifies specific teaching techniques:

```python
def _count_pedagogical_elements(self, response: str) -> float:
    """Count pedagogical elements in the response."""
    elements = {
        "question": ["?", "how", "what", "why", "can you"],
        "example": ["for example", "like", "such as"],
        "explanation": ["because", "therefore", "this means"],
        "engagement": ["let's", "try", "imagine"]
    }
    
    count = 0
    response_lower = response.lower()
    
    for element_type, markers in elements.items():
        for marker in markers:
            if marker in response_lower:
                count += 1
                break
    
    return min(count / len(elements), 1.0)
```

By recognizing these teaching patterns, we can optimize for responses that employ pedagogically sound techniques.

### 3. Length Ratio (30% of Score)

Ensures appropriate response length by comparing the generated response to the target response length:

```python
length_ratio = min(
    len(predicted.response) / len(target.response),
    len(target.response) / len(predicted.response)
)
```

This ratio helps prevent responses that are too brief or excessively lengthy.

## Cost Considerations

The current DSPy implementation uses paid OpenAI API calls:

1. **API Costs**: Each optimization round requires multiple API calls to OpenAI's GPT-3.5 Turbo
   - The default configuration makes calls through 4 parallel threads
   - Each training example processed incurs API charges
   - Running the full 3 optimization rounds multiplies these costs

2. **Cost Factors**:
   - Number of training examples in your dataset
   - Length of context and responses in tokens
   - Number of optimization rounds (currently set to 3)
   - OpenAI's pricing (approximately $0.002 per 1K tokens for GPT-3.5 Turbo)

3. **Optimization Tips**:
   - Start with a small, high-quality training set (5-10 examples)
   - Consider reducing max_rounds for initial experiments
   - Monitor token usage during optimization
   - Save and reuse successful optimized prompts

**Note**: Each time you run the training process, you will incur OpenAI API charges based on the volume of tokens processed during optimization.

## Implementation Details

The fine-tuning module uses:
- DSPy ChainOfThoughtOptimizer for model optimization
- Custom evaluation metrics for pedagogical quality
- Structured training examples with context
- Efficient model serialization for persistence

## Usage Example

```python
from fine_tuning.trainer import PedagogicalTrainer

# Initialize trainer
trainer = PedagogicalTrainer(model_name="gpt-3.5-turbo")

# Prepare training data
training_data = trainer.prepare_data("data/training_examples/")

# Train the model
optimized_model = trainer.train(
    training_data=training_data,
    num_iterations=5
)

# Use the optimized model
response = optimized_model.generate(
    prompt="Explain photosynthesis to a 3rd grader",
    context=["Photosynthesis is how plants make food using sunlight."]
)
```

## Best Practices for DSPy Fine-Tuning

1. **Creating Effective Training Data**:
   - Include diverse student questions (simple to complex)
   - Provide comprehensive context information for factual grounding
   - Include exemplary teaching responses that demonstrate pedagogical techniques
   - Annotate examples with appropriate metadata (grade level, subject, etc.)

2. **Optimizing for Performance**:
   - Balance your metrics to focus on both factual accuracy and teaching quality
   - Include examples from different domains to improve generalization
   - Create test cases separate from training data to evaluate optimization results

3. **Production Deployment**:
   - Save optimized prompts and reuse them to avoid repeated optimization costs
   - Monitor performance in production and collect new examples for future optimization
   - Consider caching common responses to reduce API usage 