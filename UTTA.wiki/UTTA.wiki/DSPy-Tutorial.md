# DSPy Optimization Tutorial

This hands-on tutorial will walk you through implementing DSPy optimization for your own use case. For conceptual understanding and comparisons with other methods, see [DSPy Optimization](DSPy-Optimization.md).

## Prerequisites

- Completed [Environment Setup](Environment-Setup.md#dspy-setup)
- OpenAI API key with billing configured
- Prepared dataset (see [Dataset Preparation](Dataset-Preparation.md))

## Implementation Steps

### Step 1: Define Your Task

Create `task_definition.py`:
```python
import dspy
from typing import List, Dict

# Define your task signature
class QASignature(dspy.Signature):
    """Signature for question-answering task"""
    input = dspy.InputField(desc="question to answer")
    output = dspy.OutputField(desc="detailed answer to the question")

# Define your predictor
class QAPredictor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(QASignature)
    
    def forward(self, question: str) -> str:
        result = self.qa(input=question)
        return result.output
```

### Step 2: Prepare Training Data

Create `prepare_data.py`:
```python
import json
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_examples(file_path: str) -> List[Dict]:
    """Load and format training examples"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    examples = []
    for item in data:
        examples.append({
            'input': item['question'],
            'output': item['answer']
        })
    
    logger.info(f"Loaded {len(examples)} examples")
    return examples

def create_train_eval_split(examples: List[Dict], eval_size: float = 0.2):
    """Split examples into train and evaluation sets"""
    split_idx = int(len(examples) * (1 - eval_size))
    return examples[:split_idx], examples[split_idx:]
```

### Step 3: Optimization Process

Create `optimize.py`:
```python
import dspy
from dspy.teleprompt import BootstrapFewShot
import logging
from task_definition import QAPredictor
from prepare_data import load_examples, create_train_eval_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Optimizer:
    def __init__(self, openai_key: str):
        # Configure DSPy
        dspy.settings.configure(lm=dspy.OpenAI(api_key=openai_key))
        
        # Initialize components
        self.predictor = QAPredictor()
        self.teleprompter = BootstrapFewShot(metric=dspy.Answer_F1())
    
    def optimize(self, train_data: List[Dict], eval_data: List[Dict]):
        """Run optimization process"""
        logger.info("Starting optimization...")
        
        # Compile predictor
        compiled_predictor = self.teleprompter.compile(
            self.predictor,
            trainset=train_data,
            evalset=eval_data
        )
        
        logger.info("Optimization complete!")
        return compiled_predictor

def main():
    # Load and split data
    examples = load_examples('processed_data/dspy_data.json')
    train_data, eval_data = create_train_eval_split(examples)
    
    # Run optimization
    optimizer = Optimizer(openai_key="your-key-here")
    optimized_predictor = optimizer.optimize(train_data, eval_data)
    
    # Save optimized predictor
    dspy.save(optimized_predictor, 'optimized_predictor.dspy')

if __name__ == "__main__":
    main()
```

### Step 4: Using the Optimized Model

Create `inference.py`:
```python
import dspy
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedQA:
    def __init__(self, model_path: str):
        self.predictor = dspy.load(model_path)
    
    def answer_question(self, question: str) -> str:
        """Generate answer using optimized predictor"""
        try:
            return self.predictor(question)
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def run_test_cases(self, test_cases: List[Dict]) -> List[Dict]:
        """Run multiple test cases"""
        results = []
        for case in test_cases:
            question = case['question']
            expected = case['answer']
            generated = self.answer_question(question)
            
            results.append({
                'question': question,
                'expected': expected,
                'generated': generated
            })
        
        return results

def main():
    # Test cases
    test_cases = [
        {
            "question": "What is photosynthesis?",
            "answer": "Photosynthesis is the process by which plants convert sunlight into energy."
        },
        {
            "question": "Explain Newton's first law.",
            "answer": "Newton's first law states that an object remains at rest or in motion unless acted upon by an external force."
        }
    ]
    
    # Run tests
    qa = OptimizedQA('optimized_predictor.dspy')
    results = qa.run_test_cases(test_cases)
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Test results saved to test_results.json")

if __name__ == "__main__":
    main()
```

## Running the Tutorial

1. **Define Task**:
   ```bash
   # Create task definition
   python task_definition.py
   ```

2. **Prepare Data**:
   ```bash
   # Format and split data
   python prepare_data.py
   ```

3. **Run Optimization**:
   ```bash
   # Optimize predictor
   python optimize.py
   ```

4. **Test Model**:
   ```bash
   # Run inference
   python inference.py
   ```

## Troubleshooting Guide

[Previous troubleshooting content remains...]

## Next Steps

1. Explore advanced topics:
   - [Custom Metrics](Evaluation-Metrics.md)
   - [Advanced Teleprompters](Advanced-DSPy.md)
   - [Production Best Practices](Model-Deployment.md)

2. Try other fine-tuning methods:
   - [OpenAI Tutorial](OpenAI-Tutorial.md)
   - [HuggingFace Tutorial](HuggingFace-Tutorial.md)

## Resources

- [DSPy API Reference](https://github.com/stanfordnlp/dspy)
- [DSPy Example Projects](https://github.com/stanfordnlp/dspy/tree/main/examples) 