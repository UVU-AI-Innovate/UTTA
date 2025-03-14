#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating DSPy optimization for educational question answering.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parents[2]))

from src.dspy.optimizer import DSPyOptimizer
from src.dspy.models import EducationalQAModel
from src.utils import load_jsonl, save_jsonl

def main():
    """Run the DSPy optimization example."""
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with:")
        print("export OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
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
    train_size = int(len(dataset) * 0.8)
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]
    
    print(f"Loaded {len(train_data)} training examples and {len(test_data)} test examples.")
    
    # Initialize the DSPy optimizer and model
    print("Initializing DSPy optimizer...")
    model = EducationalQAModel()
    optimizer = DSPyOptimizer(
        model=model,
        model_name="gpt-3.5-turbo",  # Base model to use
        max_tokens=1024,
        temperature=0.0,  # Use deterministic outputs for optimization
    )
    
    # Optimize the model
    print("Optimizing the model with DSPy...")
    optimizer.optimize(
        train_data=train_data,
        num_iterations=2,  # Reduced for example purposes
        metric="answer_quality"
    )
    
    # Evaluate on test data
    print("Evaluating optimized model on test data...")
    results = optimizer.evaluate(test_data)
    
    print("\nOptimization Results:")
    print(f"Initial Accuracy: {results['initial_accuracy']:.2f}")
    print(f"Optimized Accuracy: {results['optimized_accuracy']:.2f}")
    print(f"Improvement: {results['improvement']:.2f}%")
    
    # Show example predictions
    print("\nExample Predictions:")
    for i, example in enumerate(test_data[:2]):  # Show first 2 test examples
        question = example["question"]
        gold_answer = example["answer"]
        
        # Get prediction from optimized model
        optimized_answer = optimizer.predict(question)
        
        print(f"\nExample {i+1}:")
        print(f"Question: {question}")
        print(f"Gold Answer: {gold_answer}")
        print(f"Optimized Answer: {optimized_answer}")
    
    # Save the optimized model
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    optimizer.save(output_dir / "optimized_educational_qa_model.json")
    print(f"\nOptimized model saved to {output_dir / 'optimized_educational_qa_model.json'}")

if __name__ == "__main__":
    main() 