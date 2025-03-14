#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating DSPy optimization for educational question answering.
Using a simplified approach with direct OpenAI API calls.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key with:")
    print("export OPENAI_API_KEY=your-api-key-here")
    sys.exit(1)

from openai import OpenAI

def load_jsonl(file_path):
    """Load data from a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    
def save_jsonl(data, file_path):
    """Save data to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    """Run the educational QA optimization example."""
    print("Running educational QA optimization with OpenAI...")
    
    # Initialize the OpenAI client
    client = OpenAI()
    
    print("Loading educational QA dataset...")
    # Load example dataset
    data_path = Path(__file__).parents[1] / "data" / "educational_qa_sample.jsonl"
    
    if not data_path.exists():
        print(f"Dataset not found at {data_path}. Creating sample dataset...")
        # Create a sample dataset if it doesn't exist
        sample_data = [
            {
                "question": "What is photosynthesis?",
                "answer": "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods with the help of chlorophyll. It converts carbon dioxide and water into glucose and oxygen."
            },
            {
                "question": "Who was Marie Curie?",
                "answer": "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize, the first person to win Nobel Prizes in two different scientific fields, and the first woman to become a professor at the University of Paris."
            },
            {
                "question": "What is the Pythagorean theorem?",
                "answer": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the lengths of the other two sides. It is expressed as a² + b² = c², where c is the length of the hypotenuse and a and b are the lengths of the other two sides."
            },
            {
                "question": "What caused World War I?",
                "answer": "World War I was caused by a complex set of factors including militarism, alliances, imperialism, and nationalism. The immediate trigger was the assassination of Archduke Franz Ferdinand of Austria-Hungary in Sarajevo in June 1914, which set off a chain of diplomatic and military decisions that led to the outbreak of war."
            },
            {
                "question": "What is the water cycle?",
                "answer": "The water cycle, also known as the hydrologic cycle, is the continuous movement of water on, above, and below the surface of the Earth. It involves processes such as evaporation, transpiration, condensation, precipitation, and runoff, which circulate water throughout Earth's systems."
            }
        ]
        data_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(sample_data, data_path)
        print(f"Created sample dataset at {data_path}")
    
    # Load the dataset
    dataset = load_jsonl(data_path)
    
    # Split into train and test sets (80/20 split)
    random.shuffle(dataset)
    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples.")
    
    # Create a system prompt for educational QA
    system_prompt = """
    You are an educational assistant specialized in answering questions clearly and accurately.
    Provide comprehensive, well-structured answers that are appropriate for educational contexts.
    Include relevant facts, examples, and explanations to help understanding.
    """
    
    # Test with a few examples
    print("\nTesting educational QA with your OpenAI API key...")
    for i, example in enumerate(test_data[:2]):
        question = example["question"]
        gold_answer = example["answer"]
        
        # Get prediction using the OpenAI API
        print(f"\nExample {i+1}:")
        print(f"Question: {question}")
        print(f"Gold Answer: {gold_answer}")
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=1024
        )
        
        # Extract and print the response
        ai_answer = response.choices[0].message.content
        print(f"AI Answer: {ai_answer}")
    
    # Save results
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save a simple report
    report = {
        "model": "gpt-3.5-turbo",
        "test_questions": len(test_data),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "examples": [
            {
                "question": example["question"],
                "gold_answer": example["answer"]
            } for example in test_data[:2]
        ]
    }
    
    with open(output_dir / "educational_qa_results.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'educational_qa_results.json'}")
    print("Successfully completed the educational QA test with your OpenAI API key!")

if __name__ == "__main__":
    main() 