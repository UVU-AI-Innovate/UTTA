#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to compare the three different fine-tuning approaches.

This script demonstrates how to use all three fine-tuning approaches
(DSPy, OpenAI, and HuggingFace) on the same educational QA dataset.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent))

from src.dspy.optimizer import DSPyOptimizer
from src.dspy.models import EducationalQAModel
from src.openai.finetuner import OpenAIFineTuner
from src.huggingface.finetuner import HuggingFaceFineTuner
from src.utils import load_jsonl, save_jsonl, split_dataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare different fine-tuning approaches")
    parser.add_argument(
        "--approach", 
        type=str, 
        choices=["dspy", "openai", "huggingface", "all"], 
        default="all",
        help="Which fine-tuning approach to run"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="examples/data/educational_qa_sample.jsonl",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="comparison_results",
        help="Output directory for results"
    )
    return parser.parse_args()

def run_dspy_optimization(dataset, output_dir):
    """Run DSPy optimization."""
    print("\n" + "="*50)
    print("Running DSPy Optimization")
    print("="*50)
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with:")
        print("export OPENAI_API_KEY=your-api-key-here")
        return
    
    # Split dataset
    train_data, test_data = split_dataset(dataset)
    print(f"Using {len(train_data)} examples for training and {len(test_data)} for testing")
    
    # Initialize the DSPy optimizer and model
    print("Initializing DSPy optimizer...")
    model = EducationalQAModel()
    optimizer = DSPyOptimizer(
        model=model,
        model_name="gpt-3.5-turbo",
        max_tokens=1024,
        temperature=0.0,
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
    
    # Save the optimized model
    output_path = Path(output_dir) / "dspy" / "optimized_model.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer.save(output_path)
    
    # Save results
    results_path = Path(output_dir) / "dspy" / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return results

def run_openai_finetuning(dataset, output_dir):
    """Run OpenAI fine-tuning."""
    print("\n" + "="*50)
    print("Running OpenAI Fine-Tuning")
    print("="*50)
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key with:")
        print("export OPENAI_API_KEY=your-api-key-here")
        return
    
    # Initialize the OpenAI fine-tuner
    print("Initializing OpenAI fine-tuner...")
    finetuner = OpenAIFineTuner(
        base_model="gpt-3.5-turbo",
        suffix="educational-qa",
        n_epochs=3
    )
    
    # Prepare the dataset for fine-tuning
    print("Preparing dataset for fine-tuning...")
    prepared_data = finetuner.prepare_data(
        dataset=dataset,
        system_prompt="You are an educational assistant that provides accurate, detailed, and helpful answers to student questions.",
        input_key="question",
        output_key="answer"
    )
    
    # Save the prepared data
    output_path = Path(output_dir) / "openai"
    output_path.mkdir(parents=True, exist_ok=True)
    prepared_data_path = output_path / "prepared_data.jsonl"
    
    with open(prepared_data_path, "w") as f:
        for item in prepared_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Prepared data saved to {prepared_data_path}")
    
    # Start the fine-tuning job
    print("Starting fine-tuning job...")
    job_id = finetuner.start_finetuning(prepared_data_path)
    
    print(f"Fine-tuning job started with ID: {job_id}")
    print("Fine-tuning is running in the background. This may take some time.")
    print("You can check the status of your fine-tuning job with:")
    print(f"  openai api fine_tuning.jobs get -i {job_id}")
    
    # Save the fine-tuning configuration
    model_name = f"ft:{finetuner.base_model}:{finetuner.suffix}"
    config = {
        "base_model": finetuner.base_model,
        "suffix": finetuner.suffix,
        "job_id": job_id,
        "model_name": model_name,
        "system_prompt": "You are an educational assistant that provides accurate, detailed, and helpful answers to student questions."
    }
    
    config_path = output_path / "finetuning_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Fine-tuning configuration saved to {config_path}")
    
    return {"job_id": job_id, "model_name": model_name}

def run_huggingface_finetuning(dataset, output_dir):
    """Run HuggingFace fine-tuning."""
    print("\n" + "="*50)
    print("Running HuggingFace Fine-Tuning")
    print("="*50)
    
    try:
        import torch
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if device == "cpu":
            print("Warning: Training on CPU will be slow. Consider using a GPU for faster training.")
        
        # Initialize the HuggingFace fine-tuner
        print("Initializing HuggingFace fine-tuner...")
        output_path = Path(output_dir) / "huggingface" / "finetuned-model"
        finetuner = HuggingFaceFineTuner(
            base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Using a smaller model for example purposes
            output_dir=str(output_path),
            device=device,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            use_8bit=True if device == "cuda" else False,
        )
        
        # Prepare the dataset for fine-tuning
        print("Preparing dataset for fine-tuning...")
        train_dataset = finetuner.prepare_data(
            dataset=dataset,
            system_prompt="You are an educational assistant that provides accurate, detailed, and helpful answers to student questions.",
            input_key="question",
            output_key="answer",
            train_test_split=0.8
        )
        
        # Start fine-tuning
        print("Starting fine-tuning...")
        finetuner.train(
            train_dataset=train_dataset,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_ratio=0.03,
            max_grad_norm=0.3,
            logging_steps=10,
            save_total_limit=3,
            save_strategy="steps",
            save_steps=50,
        )
        
        print("Fine-tuning complete!")
        
        # Save model information
        model_info = {
            "base_model": finetuner.base_model,
            "output_dir": finetuner.output_dir,
            "lora_config": {
                "r": finetuner.lora_r,
                "alpha": finetuner.lora_alpha,
                "dropout": finetuner.lora_dropout,
            },
            "system_prompt": "You are an educational assistant that provides accurate, detailed, and helpful answers to student questions."
        }
        
        info_path = output_path / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {info_path}")
        
        return model_info
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Please make sure you have installed the required dependencies:")
        print("pip install transformers datasets peft accelerate bitsandbytes torch")
        return None

def main():
    """Main function to compare fine-tuning approaches."""
    args = parse_args()
    
    # Load dataset
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        return 1
    
    dataset = load_jsonl(data_path)
    print(f"Loaded {len(dataset)} examples from {data_path}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run selected approaches
    if args.approach in ["dspy", "all"]:
        results["dspy"] = run_dspy_optimization(dataset, output_dir)
    
    if args.approach in ["openai", "all"]:
        results["openai"] = run_openai_finetuning(dataset, output_dir)
    
    if args.approach in ["huggingface", "all"]:
        results["huggingface"] = run_huggingface_finetuning(dataset, output_dir)
    
    # Save overall results
    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print(f"Comparison results saved to {results_path}")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 