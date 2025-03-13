#!/usr/bin/env python
"""
HuggingFace Fine-Tuning Example with Educational Question Answering.

This script demonstrates how to use the HuggingFace Transformers library
with LoRA to fine-tune a model for educational question answering.
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.huggingface import HuggingFaceFineTuner
from src.utils.dataset import load_jsonl, split_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace Fine-Tuning for Educational QA")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="examples/data/educational_qa_sample.jsonl",
        help="Path to the dataset file"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model ID to fine-tune (default is a smaller model for faster demo)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/huggingface_finetuning",
        help="Directory to save the fine-tuned model and results"
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are an educational assistant that provides comprehensive, accurate, and instructive answers to questions.",
        help="System prompt for fine-tuning"
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization (recommended for larger models)"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.1,
        help="Portion of data to use for testing (0.0-1.0)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("HuggingFace Fine-Tuning for Educational QA")
    print("="*50)
    
    # Step 1: Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nStep 1: Checking hardware...")
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: GPU not available. Fine-tuning will be slow.")
        if args.use_4bit:
            print("Warning: 4-bit quantization requested but running on CPU. This will be slow.")
    
    # Step 2: Load and prepare the dataset
    print("\nStep 2: Loading and preparing the dataset...")
    dataset = load_jsonl(args.dataset_path)
    
    if not dataset:
        print(f"Error: Could not load dataset from {args.dataset_path}")
        return
    
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
    
    # Split into train and test sets
    train_data, test_data = split_dataset(dataset, test_size=args.test_split, seed=args.seed)
    print(f"Split into {len(train_data)} training and {len(test_data)} testing examples")
    
    # Step 3: Initialize the fine-tuner
    print("\nStep 3: Initializing fine-tuner...")
    print(f"Base model: {args.base_model}")
    
    fine_tuner = HuggingFaceFineTuner(
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_4bit=args.use_4bit
    )
    
    # Step 4: Prepare data for fine-tuning
    print("\nStep 4: Preparing data for fine-tuning...")
    
    train_dataset = fine_tuner.prepare_data(
        dataset=train_data,
        system_prompt=args.system_prompt
    )
    
    test_dataset = None
    if test_data:
        test_dataset = fine_tuner.prepare_data(
            dataset=test_data,
            system_prompt=args.system_prompt
        )
    
    # Step 5: Run the fine-tuning
    print("\nStep 5: Starting fine-tuning...")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    
    results = fine_tuner.fine_tune(
        train_dataset=train_dataset,
        val_dataset=test_dataset,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # Step 6: Save results and model info
    print("\nStep 6: Saving results...")
    
    # Save training metrics
    metrics_path = os.path.join(args.output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    
    # Save model info
    model_info = {
        "base_model": args.base_model,
        "output_dir": args.output_dir,
        "lora_config": {
            "r": args.lora_r,
            "alpha": args.lora_alpha
        },
        "training_params": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.num_epochs,
            "system_prompt": args.system_prompt
        },
        "dataset": {
            "path": args.dataset_path,
            "train_examples": len(train_data),
            "test_examples": len(test_data) if test_data else 0
        },
        "hardware": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None
        }
    }
    
    with open(os.path.join(args.output_dir, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Step 7: Run a demo prediction
    print("\nStep 7: Demo prediction...")
    if test_data:
        test_example = test_data[0]
        print("\nQuestion:")
        print(test_example["question"])
        
        print("\nGenerating answer using fine-tuned model...")
        answer = fine_tuner.generate(
            question=test_example["question"],
            system_prompt=args.system_prompt
        )
        
        print("\nGenerated Answer:")
        print(answer)
        
        print("\nReference Answer:")
        print(test_example["answer"])
    
    print("\n" + "="*50)
    print("HuggingFace Fine-Tuning Complete!")
    print("="*50)
    print(f"\nFine-tuned model saved to: {args.output_dir}")
    print(f"Model info saved to: {args.output_dir}/model_info.json")

if __name__ == "__main__":
    main() 