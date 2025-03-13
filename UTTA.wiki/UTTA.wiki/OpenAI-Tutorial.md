# OpenAI Fine-Tuning Tutorial

This hands-on tutorial will walk you through implementing OpenAI fine-tuning for your own use case. For conceptual understanding and comparisons with other methods, see [OpenAI Fine-Tuning](OpenAI-Fine-Tuning.md).

## Prerequisites

- Completed [Environment Setup](Environment-Setup.md#openai-setup)
- OpenAI API key with billing configured
- Prepared dataset (see [Dataset Preparation](Dataset-Preparation.md))

## Implementation Steps

### Step 1: Data Preparation

Create `prepare_training_data.py`:
```python
import json
import tiktoken
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingDataPreparator:
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 4096
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def format_qa_pair(self, question: str, answer: str) -> dict:
        """Format Q&A pair for fine-tuning"""
        return {
            "messages": [
                {"role": "system", "content": "You are a helpful educational assistant."},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
    
    def validate_example(self, example: dict) -> bool:
        """Validate a single training example"""
        total_tokens = sum(
            self.count_tokens(msg["content"]) 
            for msg in example["messages"]
        )
        return total_tokens <= self.max_tokens
    
    def prepare_data(self, input_file: str, output_file: str):
        """Prepare training data for fine-tuning"""
        logger.info(f"Reading data from {input_file}")
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        valid_examples = []
        stats = {"total": 0, "valid": 0, "invalid": 0}
        
        for item in data:
            stats["total"] += 1
            example = self.format_qa_pair(item["question"], item["answer"])
            
            if self.validate_example(example):
                valid_examples.append(example)
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
        
        # Save formatted data
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            for example in valid_examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"Statistics: {stats}")
        logger.info(f"Saved {stats['valid']} examples to {output_file}")

if __name__ == "__main__":
    preparator = TrainingDataPreparator()
    preparator.prepare_data('processed_data/dspy_data.json', 'training_data.jsonl')

### Step 2: Training Process

Create `run_finetuning.py`:
```python
from openai import OpenAI
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self):
        self.client = OpenAI()
    
    def create_job(self, training_file: str, model: str = "gpt-3.5-turbo"):
        """Create fine-tuning job"""
        try:
            # Upload training file
            with open(training_file, 'rb') as f:
                upload = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            logger.info(f"File uploaded: {upload.id}")
            
            # Create fine-tuning job
            job = self.client.fine_tuning.jobs.create(
                training_file=upload.id,
                model=model
            )
            logger.info(f"Fine-tuning job created: {job.id}")
            
            return job.id
        
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            return None
    
    def monitor_progress(self, job_id: str):
        """Monitor fine-tuning progress"""
        while True:
            try:
                job = self.client.fine_tuning.jobs.retrieve(job_id)
                logger.info(f"Status: {job.status}")
                
                if job.status in ["succeeded", "failed"]:
                    return job
                
                time.sleep(60)  # Check every minute
            
            except Exception as e:
                logger.error(f"Error monitoring job: {e}")
                return None

if __name__ == "__main__":
    tuner = FineTuner()
    job_id = tuner.create_job('training_data.jsonl')
    if job_id:
        final_job = tuner.monitor_progress(job_id)
        if final_job and final_job.status == "succeeded":
            logger.info(f"Fine-tuning complete! Model: {final_job.fine_tuned_model}")

### Step 3: Model Usage

Create `test_model.py`:
```python
from openai import OpenAI
import json
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(self, model_id: str):
        self.client = OpenAI()
        self.model_id = model_id
    
    def generate_answer(self, question: str) -> str:
        """Generate answer using fine-tuned model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful educational assistant."},
                    {"role": "user", "content": question}
                ]
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return None
    
    def run_test_cases(self, test_cases: List[Dict]) -> Dict:
        """Run multiple test cases"""
        results = []
        
        for case in test_cases:
            question = case["question"]
            expected = case["answer"]
            generated = self.generate_answer(question)
            
            results.append({
                "question": question,
                "expected": expected,
                "generated": generated
            })
        
        return results

def main():
    # Load your fine-tuned model ID
    model_id = "ft:gpt-3.5-turbo:..."  # Replace with your model ID
    
    # Create test cases
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
    tester = ModelTester(model_id)
    results = tester.run_test_cases(test_cases)
    
    # Save results
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Test results saved to test_results.json")

if __name__ == "__main__":
    main()

### Step 4: Production Deployment

[Previous deployment content remains...]

## Troubleshooting Guide

[Previous troubleshooting content remains...]

## Next Steps

1. Explore advanced topics:
   - [Hyperparameter Tuning](Hyperparameter-Tuning.md)
   - [Model Evaluation](Evaluation-Metrics.md)
   - [Production Best Practices](Model-Deployment.md)

2. Try other fine-tuning methods:
   - [DSPy Tutorial](DSPy-Tutorial.md)
   - [HuggingFace Tutorial](HuggingFace-Tutorial.md)

## Resources

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/fine-tuning)
- [OpenAI Cookbook Examples](https://github.com/openai/openai-cookbook) 