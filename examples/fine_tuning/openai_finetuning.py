#!/usr/bin/env python
"""
OpenAI Fine-Tuning Example with Educational Question Answering.

This script demonstrates how to use the OpenAI API to fine-tune a model
for educational question answering.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.openai import OpenAIFineTuner
from src.utils.dataset import load_jsonl, split_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI Fine-Tuning for Educational QA")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="examples/data/educational_qa_sample.jsonl",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI base model to fine-tune"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are an educational assistant that provides comprehensive, accurate, and instructive answers to questions.",
        help="System prompt for fine-tuning"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/openai_finetuning",
        help="Directory to save results"
    )
    parser.add_argument(
        "--wait_for_completion",
        action="store_true",
        help="Wait for the fine-tuning job to complete (can take hours)"
    )
    parser.add_argument(
        "--max_wait_time",
        type=int,
        default=3600,
        help="Maximum time to wait for job completion in seconds (default: 1 hour)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key with:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        return
    
    print("\n" + "="*50)
    print("OpenAI Fine-Tuning for Educational QA")
    print("="*50)
    
    # Step 1: Load the dataset
    print("\nStep 1: Loading the dataset...")
    dataset = load_jsonl(args.dataset_path)
    
    if not dataset:
        print(f"Error: Could not load dataset from {args.dataset_path}")
        return
    
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
    
    # Step 2: Initialize the fine-tuner
    print("\nStep 2: Initializing OpenAI fine-tuner...")
    fine_tuner = OpenAIFineTuner(
        base_model=args.base_model,
        n_epochs=args.n_epochs
    )
    
    # Step 3: Prepare the data for fine-tuning
    print("\nStep 3: Preparing data for fine-tuning...")
    prepared_data = fine_tuner.prepare_data(
        dataset=dataset,
        system_prompt=args.system_prompt
    )
    
    # Calculate token count and estimated cost
    token_info = fine_tuner.calculate_token_count(prepared_data)
    print(f"\nDataset Statistics:")
    print(f"  Number of examples: {token_info['n_examples']}")
    print(f"  Total tokens: {token_info['total_tokens']}")
    print(f"  Average tokens per example: {token_info['tokens_per_example']:.1f}")
    print(f"  Estimated cost: ${token_info['estimated_cost']:.2f}")
    
    # Save prepared data to JSONL file
    prepared_data_path = os.path.join(args.output_dir, "prepared_data.jsonl")
    with open(prepared_data_path, "w", encoding="utf-8") as f:
        for example in prepared_data:
            f.write(json.dumps(example) + "\n")
    
    print(f"Prepared data saved to {prepared_data_path}")
    
    # Step 4: Start the fine-tuning job
    print("\nStep 4: Starting fine-tuning job...")
    print(f"Base model: {args.base_model}")
    print(f"Number of epochs: {args.n_epochs}")
    
    job_id = fine_tuner.start_finetuning(
        data_path=prepared_data_path
    )
    
    print(f"Fine-tuning job started with ID: {job_id}")
    
    # Save the configuration
    config = {
        "job_id": job_id,
        "base_model": args.base_model,
        "system_prompt": args.system_prompt,
        "n_epochs": args.n_epochs,
        "dataset_path": args.dataset_path,
        "num_examples": len(dataset),
        "token_info": token_info,
        "timestamp": int(time.time())
    }
    
    with open(os.path.join(args.output_dir, "finetuning_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Step 5: Wait for completion if requested
    if args.wait_for_completion:
        print("\nStep 5: Waiting for fine-tuning job to complete...")
        print(f"This may take a while. Maximum wait time: {args.max_wait_time} seconds")
        
        status = fine_tuner.wait_for_completion(
            job_id=job_id,
            max_wait_time=args.max_wait_time
        )
        
        print(f"\nFine-tuning job status: {status['status']}")
        
        if status['status'] == 'succeeded':
            print(f"Fine-tuned model name: {status['fine_tuned_model']}")
            
            # Save the final status
            with open(os.path.join(args.output_dir, "job_result.json"), "w") as f:
                json.dump(status, f, indent=2)
            
            # Step 6: Demo the fine-tuned model
            if status.get('fine_tuned_model'):
                print("\nStep 6: Demo of fine-tuned model")
                
                if dataset:
                    test_example = dataset[0]
                    print("\nQuestion:")
                    print(test_example["question"])
                    
                    print("\nGenerating answer using fine-tuned model...")
                    answer = fine_tuner.generate(
                        model_name=status['fine_tuned_model'],
                        question=test_example["question"],
                        system_prompt=args.system_prompt
                    )
                    
                    print("\nGenerated Answer:")
                    print(answer)
                    
                    print("\nReference Answer:")
                    print(test_example["answer"])
        else:
            print("Job did not complete successfully within the waiting time.")
    else:
        print("\nStep 5: Job monitoring instructions")
        print("The fine-tuning job is running in the background.")
        print("You can check its status with the following command:")
        print(f"  python -c \"from openai import OpenAI; client = OpenAI(); print(client.fine_tuning.jobs.retrieve('{job_id}'))\"")
        print("\nOr by using the OpenAI web interface.")
    
    print("\n" + "="*50)
    print("OpenAI Fine-Tuning Setup Complete!")
    print("="*50)
    print(f"\nJob configuration saved to: {args.output_dir}/finetuning_config.json")

if __name__ == "__main__":
    main() 