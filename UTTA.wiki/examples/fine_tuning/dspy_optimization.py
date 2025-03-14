#!/usr/bin/env python
"""
DSPy Optimization Example with Educational Question Answering.

This script demonstrates how to use DSPy for optimizing prompts for
educational question answering without changing model weights.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import random

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dspy import EducationalQAModel, DSPyOptimizer
from src.utils.dataset import load_jsonl, split_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="DSPy Optimization for Educational QA")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="examples/data/educational_qa_sample.jsonl",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use for optimization"
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/dspy_optimization",
        help="Directory to save results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key with:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        return
    
    print("\n" + "="*50)
    print("DSPy Optimization for Educational QA")
    print("="*50)
    
    # Step 1: Load and split the dataset
    print("\nStep 1: Loading and preparing the dataset...")
    dataset = load_jsonl(args.dataset_path)
    
    if not dataset:
        print(f"Error: Could not load dataset from {args.dataset_path}")
        return
    
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
    
    # Split into train and test sets (80% train, 20% test)
    train_data, test_data = split_dataset(dataset, test_size=0.2, seed=args.seed)
    print(f"Split into {len(train_data)} training and {len(test_data)} testing examples")
    
    # Step 2: Initialize the model and optimizer
    print("\nStep 2: Initializing model and optimizer...")
    model = EducationalQAModel()
    
    optimizer = DSPyOptimizer(
        model=model,
        model_name=args.model_name,
        max_tokens=1024,
        temperature=0.7
    )
    
    # Step 3: Run the optimization
    print("\nStep 3: Running DSPy optimization...")
    print(f"Using model: {args.model_name}")
    print(f"Running {args.num_iterations} iterations of optimization")
    
    optimization_results = optimizer.optimize(
        train_data=train_data,
        num_iterations=args.num_iterations,
        metric="answer_quality"
    )
    
    # Step 4: Evaluate on test data
    print("\nStep 4: Evaluating on test data...")
    evaluation_results = optimizer.evaluate(test_data)
    
    print("\nEvaluation Results:")
    print(f"Average Answer Quality: {evaluation_results['average_quality']:.2f}")
    print(f"Average Token Overlap: {evaluation_results['average_overlap']:.2f}")
    
    # Step 5: Save the optimized model and results
    print("\nStep 5: Saving optimized model and results...")
    optimizer.save(os.path.join(args.output_dir, "optimized_dspy_model.json"))
    
    # Save evaluation results
    results = {
        "model_name": args.model_name,
        "dataset": args.dataset_path,
        "num_train_examples": len(train_data),
        "num_test_examples": len(test_data),
        "num_iterations": args.num_iterations,
        "optimization_results": optimization_results,
        "evaluation_results": evaluation_results,
        "saved_model_path": os.path.join(args.output_dir, "optimized_dspy_model.json")
    }
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Step 6: Run a demo prediction
    print("\nStep 6: Demo prediction...")
    if test_data:
        test_example = test_data[0]
        print("\nQuestion:")
        print(test_example["question"])
        
        print("\nGenerating answer using optimized DSPy model...")
        prediction = optimizer.predict(test_example["question"])
        
        print("\nGenerated Answer:")
        print(prediction["answer"])
        
        print("\nReference Answer:")
        print(test_example["answer"])
    
    print("\n" + "="*50)
    print("DSPy Optimization Complete!")
    print("="*50)
    print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()