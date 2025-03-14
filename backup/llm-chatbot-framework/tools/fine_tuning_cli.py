#!/usr/bin/env python
"""
Fine-Tuning CLI Tool

This command-line tool provides utilities for managing fine-tuning 
jobs for both OpenAI and HuggingFace models.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# Set up path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import fine-tuning modules
from src.llm.fine_tuning.openai_tuner import OpenAIFineTuner
from src.llm.fine_tuning.huggingface_tuner import HuggingFaceFineTuner
from src.llm.fine_tuning.dspy_optimizer import DSPyOptimizer
from src.llm.dspy.handler import DSPyLLMHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning_cli.log"),
        logging.StreamHandler()
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning management CLI")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # OpenAI commands
    openai_parser = subparsers.add_parser("openai", help="OpenAI fine-tuning commands")
    openai_subparsers = openai_parser.add_subparsers(dest="openai_command", help="OpenAI command")
    
    # OpenAI: Start job
    openai_start = openai_subparsers.add_parser("start", help="Start a fine-tuning job")
    openai_start.add_argument("--file", required=True, help="JSONL file with training data")
    openai_start.add_argument("--model", default="gpt-3.5-turbo", help="Base model to fine-tune")
    openai_start.add_argument("--suffix", help="Custom suffix for the model name")
    
    # OpenAI: List jobs
    openai_list = openai_subparsers.add_parser("list-jobs", help="List fine-tuning jobs")
    
    # OpenAI: Check status
    openai_status = openai_subparsers.add_parser("status", help="Check job status")
    openai_status.add_argument("--job-id", required=True, help="Fine-tuning job ID")
    
    # OpenAI: List models
    openai_models = openai_subparsers.add_parser("list-models", help="List fine-tuned models")
    
    # OpenAI: Cancel job
    openai_cancel = openai_subparsers.add_parser("cancel", help="Cancel a job")
    openai_cancel.add_argument("--job-id", required=True, help="Fine-tuning job ID")
    
    # OpenAI: Delete model
    openai_delete = openai_subparsers.add_parser("delete-model", help="Delete a fine-tuned model")
    openai_delete.add_argument("--model-id", required=True, help="Model ID to delete")
    
    # HuggingFace commands
    hf_parser = subparsers.add_parser("huggingface", help="HuggingFace fine-tuning commands")
    hf_subparsers = hf_parser.add_subparsers(dest="hf_command", help="HuggingFace command")
    
    # HuggingFace: Fine-tune
    hf_finetune = hf_subparsers.add_parser("fine-tune", help="Fine-tune a HuggingFace model")
    hf_finetune.add_argument("--file", required=True, help="JSONL file with training data")
    hf_finetune.add_argument("--model", default="microsoft/phi-2", help="Base model to fine-tune")
    hf_finetune.add_argument("--output-dir", default="models/fine_tuned", help="Output directory")
    hf_finetune.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    hf_finetune.add_argument("--batch-size", type=int, default=8, help="Batch size")
    hf_finetune.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    hf_finetune.add_argument("--use-lora", action="store_true", help="Use LoRA for fine-tuning")
    hf_finetune.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    hf_finetune.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    hf_finetune.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    hf_finetune.add_argument("--use-4bit", action="store_true", help="Use 4-bit quantization")
    
    # HuggingFace: List models
    hf_list = hf_subparsers.add_parser("list-models", help="List fine-tuned models")
    hf_list.add_argument("--output-dir", default="models/fine_tuned", help="Models directory")
    
    # HuggingFace: Delete model
    hf_delete = hf_subparsers.add_parser("delete-model", help="Delete a fine-tuned model")
    hf_delete.add_argument("--model-path", required=True, help="Path to model to delete")
    
    # HuggingFace: Export model
    hf_export = hf_subparsers.add_parser("export-model", help="Export a model for framework integration")
    hf_export.add_argument("--model-path", required=True, help="Path to model to export")
    hf_export.add_argument("--export-dir", help="Export directory")
    
    # DSPy optimization commands
    dspy_parser = subparsers.add_parser("dspy", help="DSPy optimization commands")
    dspy_subparsers = dspy_parser.add_subparsers(dest="dspy_command", help="DSPy command")
    
    # DSPy: Optimize
    dspy_optimize = dspy_subparsers.add_parser("optimize", help="Optimize a DSPy module")
    dspy_optimize.add_argument("--examples", required=True, help="JSON file with training examples")
    dspy_optimize.add_argument("--module", required=True, help="Module name to optimize")
    dspy_optimize.add_argument("--metric", default="exact_match", help="Evaluation metric")
    dspy_optimize.add_argument("--method", default="bootstrap_few_shot", 
                             help="Optimization method (bootstrap_few_shot, bootstrap_random_search, bootstrap_iterative_bisect)")
    dspy_optimize.add_argument("--max-demos", type=int, default=5, help="Maximum number of demonstrations")
    dspy_optimize.add_argument("--model", default="gpt-3.5-turbo", help="LLM to use for optimization")
    
    # DSPy: Save module
    dspy_save = dspy_subparsers.add_parser("save", help="Save an optimized module")
    dspy_save.add_argument("--module", required=True, help="Module name to save")
    dspy_save.add_argument("--path", required=True, help="Path to save to")
    
    # DSPy: Load module
    dspy_load = dspy_subparsers.add_parser("load", help="Load an optimized module")
    dspy_load.add_argument("--module", required=True, help="Module name to load")
    dspy_load.add_argument("--path", required=True, help="Path to load from")
    
    # Preparation commands
    prep_parser = subparsers.add_parser("prepare", help="Prepare data for fine-tuning")
    prep_subparsers = prep_parser.add_subparsers(dest="prep_command", help="Preparation command")
    
    # Prepare: Convert conversations
    prep_convert = prep_subparsers.add_parser("convert", help="Convert conversations to fine-tuning format")
    prep_convert.add_argument("--input", required=True, help="Input JSON file with conversations")
    prep_convert.add_argument("--output", required=True, help="Output JSONL file")
    prep_convert.add_argument("--format", choices=["openai", "huggingface"], default="openai", 
                              help="Output format for fine-tuning")
    
    return parser.parse_args()

def handle_openai_commands(args):
    """Handle OpenAI-related commands"""
    try:
        tuner = OpenAIFineTuner()
        
        if args.openai_command == "start":
            job_id = tuner.start_fine_tuning(
                training_file=args.file,
                model=args.model,
                suffix=args.suffix
            )
            print(f"Started fine-tuning job with ID: {job_id}")
            
        elif args.openai_command == "list-jobs":
            jobs = tuner.list_fine_tuning_jobs()
            print(f"Found {len(jobs)} fine-tuning jobs:")
            for job in jobs:
                print(f"  {job['id']} - Model: {job['model']}, Status: {job['status']}")
            
        elif args.openai_command == "status":
            status = tuner.check_fine_tuning_status(args.job_id)
            print(f"Job {args.job_id} status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            
        elif args.openai_command == "list-models":
            models = tuner.list_fine_tuned_models()
            print(f"Found {len(models)} fine-tuned models:")
            for model in models:
                print(f"  {model['id']} - Created: {model['created_at']}")
            
        elif args.openai_command == "cancel":
            success = tuner.cancel_fine_tuning_job(args.job_id)
            if success:
                print(f"Successfully cancelled job {args.job_id}")
            else:
                print(f"Failed to cancel job {args.job_id}")
            
        elif args.openai_command == "delete-model":
            success = tuner.delete_fine_tuned_model(args.model_id)
            if success:
                print(f"Successfully deleted model {args.model_id}")
            else:
                print(f"Failed to delete model {args.model_id}")
                
        else:
            print("Unknown OpenAI command. Use --help for usage information.")
            
    except Exception as e:
        logging.error(f"Error executing OpenAI command: {str(e)}")
        print(f"Error: {str(e)}")

def handle_huggingface_commands(args):
    """Handle HuggingFace-related commands"""
    try:
        # For fine-tune command, use model and output_dir from args
        if args.hf_command == "fine-tune":
            tuner = HuggingFaceFineTuner(
                model_name=args.model,
                output_dir=args.output_dir
            )
            
            # Load training data
            with open(args.file, 'r') as f:
                conversations = [json.loads(line) for line in f]
            
            # Prepare data
            dataset = tuner.prepare_data(conversations)
            
            # Fine-tune model
            output_path = tuner.fine_tune(
                dataset=dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_lora=args.use_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                use_8bit=args.use_8bit,
                use_4bit=args.use_4bit
            )
            
            print(f"Fine-tuning complete. Model saved to: {output_path}")
            
        # For other commands, initialize with default or provided output_dir
        else:
            output_dir = getattr(args, "output_dir", "models/fine_tuned")
            tuner = HuggingFaceFineTuner(output_dir=output_dir)
            
            if args.hf_command == "list-models":
                models = tuner.list_fine_tuned_models()
                print(f"Found {len(models)} fine-tuned models:")
                for model in models:
                    path = model.get("path", "")
                    name = path.split("/")[-1] if path else model.get("name", "Unknown")
                    timestamp = model.get("timestamp", "Unknown")
                    if isinstance(timestamp, float):
                        timestamp = datetime.fromtimestamp(timestamp).isoformat()
                    print(f"  {name} - Created: {timestamp}")
                    print(f"    Path: {path}")
                    
            elif args.hf_command == "delete-model":
                success = tuner.delete_fine_tuned_model(args.model_path)
                if success:
                    print(f"Successfully deleted model at {args.model_path}")
                else:
                    print(f"Failed to delete model at {args.model_path}")
                    
            elif args.hf_command == "export-model":
                export_dir = args.export_dir
                output_path = tuner.export_model_for_integration(args.model_path, export_dir)
                print(f"Model exported to: {output_path}")
                
            else:
                print("Unknown HuggingFace command. Use --help for usage information.")
                
    except Exception as e:
        logging.error(f"Error executing HuggingFace command: {str(e)}")
        print(f"Error: {str(e)}")

def handle_dspy_commands(args):
    """Handle DSPy optimization commands"""
    try:
        # Create LLM handler and optimizer
        handler = DSPyLLMHandler(model_name=getattr(args, "model", "gpt-3.5-turbo"))
        optimizer = DSPyOptimizer(llm_handler=handler)
        
        if args.dspy_command == "optimize":
            # Load examples from file
            with open(args.examples, 'r') as f:
                examples = json.load(f)
            
            # Optimize the module
            optimized_module = optimizer.optimize_module(
                module_name=args.module,
                training_examples=examples,
                metric=args.metric,
                method=args.method,
                max_demos=args.max_demos
            )
            
            print(f"Successfully optimized module {args.module}")
            
        elif args.dspy_command == "save":
            # Save the optimized module
            success = optimizer.save_optimized_module(args.module, args.path)
            if success:
                print(f"Successfully saved optimized module {args.module} to {args.path}")
            else:
                print(f"Failed to save optimized module {args.module}")
                
        elif args.dspy_command == "load":
            # Load the optimized module
            loaded_module = optimizer.load_optimized_module(args.module, args.path)
            if loaded_module:
                print(f"Successfully loaded optimized module {args.module} from {args.path}")
            else:
                print(f"Failed to load optimized module {args.module}")
                
        else:
            print("Unknown DSPy command. Use --help for usage information.")
            
    except Exception as e:
        logging.error(f"Error executing DSPy command: {str(e)}")
        print(f"Error: {str(e)}")

def handle_preparation_commands(args):
    """Handle data preparation commands"""
    try:
        if args.prep_command == "convert":
            # Load input conversations
            with open(args.input, 'r') as f:
                conversations = json.load(f)
            
            # Convert based on format
            if args.format == "openai":
                tuner = OpenAIFineTuner()
                output_file = tuner.prepare_data(conversations, args.output)
                print(f"Converted {len(conversations)} conversations to OpenAI format at {output_file}")
                
            elif args.format == "huggingface":
                # For HuggingFace, we still need to save the JSONL file
                # but in a format compatible with HuggingFace's Dataset
                valid_conversations = []
                for conv in conversations:
                    if "messages" in conv and isinstance(conv["messages"], list):
                        valid_conversations.append(conv)
                
                with open(args.output, 'w') as f:
                    for conv in valid_conversations:
                        f.write(json.dumps(conv) + "\n")
                        
                print(f"Converted {len(valid_conversations)} conversations to HuggingFace format at {args.output}")
                
            else:
                print(f"Unknown format: {args.format}")
                
        else:
            print("Unknown preparation command. Use --help for usage information.")
            
    except Exception as e:
        logging.error(f"Error executing preparation command: {str(e)}")
        print(f"Error: {str(e)}")

def main():
    args = parse_args()
    
    if args.command == "openai":
        handle_openai_commands(args)
    elif args.command == "huggingface":
        handle_huggingface_commands(args)
    elif args.command == "dspy":
        handle_dspy_commands(args)
    elif args.command == "prepare":
        handle_preparation_commands(args)
    else:
        print("Please specify a command. Use --help for usage information.")

if __name__ == "__main__":
    main() 