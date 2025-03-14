#!/usr/bin/env python
"""
LLM Fine-Tuning Toolkit - Command Line Interface

This script provides a unified command-line interface for using all three
fine-tuning approaches: DSPy, OpenAI, and HuggingFace.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Fine-Tuning Toolkit - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Main arguments
    parser.add_argument(
        "approach",
        choices=["dspy", "openai", "huggingface", "generate-data"],
        help="Fine-tuning approach to use or data generation"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="examples/data/educational_qa_sample.jsonl",
        help="Path to the dataset file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save results (default depends on approach)"
    )
    
    # Generate dataset arguments
    parser.add_argument(
        "--num_examples",
        type=int,
        default=30,
        help="Number of examples to generate (for generate-data only)"
    )
    
    # Common fine-tuning arguments
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are an educational assistant that provides comprehensive, accurate, and instructive answers to questions.",
        help="System prompt for fine-tuning"
    )
    
    # DSPy specific arguments
    parser.add_argument(
        "--dspy_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model to use for DSPy optimization"
    )
    
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=3,
        help="Number of optimization iterations for DSPy"
    )
    
    # OpenAI specific arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="gpt-3.5-turbo",
        help="Base model for OpenAI fine-tuning"
    )
    
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=3,
        help="Number of epochs for OpenAI fine-tuning"
    )
    
    parser.add_argument(
        "--wait_for_completion",
        action="store_true",
        help="Wait for OpenAI fine-tuning job to complete"
    )
    
    # HuggingFace specific arguments
    parser.add_argument(
        "--hf_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID to fine-tune"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for HuggingFace fine-tuning"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for HuggingFace fine-tuning"
    )
    
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for HuggingFace fine-tuning"
    )
    
    # General arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        if args.approach == "dspy":
            args.output_dir = "results/dspy_optimization"
        elif args.approach == "openai":
            args.output_dir = "results/openai_finetuning"
        elif args.approach == "huggingface":
            args.output_dir = "results/huggingface_finetuning"
        elif args.approach == "generate-data":
            args.output_dir = "examples/data"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for OpenAI API key if needed
    if args.approach in ["dspy", "openai"] and not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key with:")
        print("  export OPENAI_API_KEY=your_api_key_here")
        return
    
    # Construct command based on approach
    if args.approach == "generate-data":
        script_path = "examples/create_educational_dataset.py"
        cmd = [
            sys.executable, script_path,
            "--num_examples", str(args.num_examples),
            "--output_path", f"{args.output_dir}/educational_qa_sample.jsonl",
            "--seed", str(args.seed)
        ]
        
    elif args.approach == "dspy":
        script_path = "examples/dspy_optimization.py"
        cmd = [
            sys.executable, script_path,
            "--dataset_path", args.dataset_path,
            "--model_name", args.dspy_model,
            "--num_iterations", str(args.num_iterations),
            "--output_dir", args.output_dir,
            "--seed", str(args.seed)
        ]
        
    elif args.approach == "openai":
        script_path = "examples/openai_finetuning.py"
        cmd = [
            sys.executable, script_path,
            "--dataset_path", args.dataset_path,
            "--base_model", args.base_model,
            "--n_epochs", str(args.n_epochs),
            "--system_prompt", args.system_prompt,
            "--output_dir", args.output_dir
        ]
        
        if args.wait_for_completion:
            cmd.append("--wait_for_completion")
        
    elif args.approach == "huggingface":
        script_path = "examples/huggingface_finetuning.py"
        cmd = [
            sys.executable, script_path,
            "--dataset_path", args.dataset_path,
            "--base_model", args.hf_model,
            "--output_dir", args.output_dir,
            "--num_epochs", str(args.n_epochs),
            "--learning_rate", str(args.learning_rate),
            "--batch_size", str(args.batch_size),
            "--system_prompt", args.system_prompt,
            "--seed", str(args.seed)
        ]
        
        if args.use_4bit:
            cmd.append("--use_4bit")
    
    # Print command if verbose
    if args.verbose:
        print(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        return
    
    print("\nCommand completed successfully.")
    
    # Print next steps based on approach
    if args.approach == "generate-data":
        print(f"\nDataset generated at: {args.output_dir}/educational_qa_sample.jsonl")
        print("\nNext steps:")
        print("  - Use the dataset for fine-tuning with one of the approaches:")
        print("    python finetune.py dspy --dataset_path YOUR_DATASET_PATH")
        print("    python finetune.py openai --dataset_path YOUR_DATASET_PATH")
        print("    python finetune.py huggingface --dataset_path YOUR_DATASET_PATH")
        
    elif args.approach == "dspy":
        print(f"\nDSPy optimization results saved to: {args.output_dir}")
        print("\nNext steps:")
        print("  - Examine the optimized prompts in the results directory")
        print("  - Try different optimization metrics or iterations")
        print("  - Use the optimized model in your application")
        
    elif args.approach == "openai":
        print(f"\nOpenAI fine-tuning job started. Configuration saved to: {args.output_dir}")
        print("\nNext steps:")
        print("  - Wait for the fine-tuning job to complete (can take hours)")
        print("  - Check the status in the OpenAI web interface")
        print("  - Use the fine-tuned model with your OpenAI API key")
        
    elif args.approach == "huggingface":
        print(f"\nHuggingFace fine-tuned model saved to: {args.output_dir}")
        print("\nNext steps:")
        print("  - Use the fine-tuned model in your application")
        print("  - Try different hyperparameters or base models")
        print("  - Deploy the model using HuggingFace's model serving options")

if __name__ == "__main__":
    main() 