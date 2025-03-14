"""
Example script demonstrating the LLM evaluation framework.
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from src.evaluation.metrics.automated_metrics import AutomatedMetrics
from src.evaluation.metrics.system_metrics import SystemMetrics
from src.evaluation.human.human_evaluation import HumanEvaluation
from src.evaluation.benchmarks.benchmark_suite import BenchmarkSuite
from config import (
    load_token, list_available_models, get_model_config,
    AVAILABLE_MODELS
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM Evaluation Framework Demo')
    parser.add_argument(
        '--model',
        choices=list(AVAILABLE_MODELS.keys()),
        help='Model to use for evaluation'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    return parser.parse_args()

def load_model(model_key=None):
    """
    Load a model from Hugging Face Hub with optimized multi-GPU configuration.
    Uses 4-bit quantization and mixed precision for efficient loading.
    """
    # Initialize system metrics tracking
    metrics = SystemMetrics()
    metrics.print_gpu_stats("Before model loading")
    
    # Get token from config
    token = load_token()
    if not token:
        raise ValueError("Hugging Face token not found. Please set it in config/.env")
    
    # Get model configuration
    config = get_model_config(model_key)
    model_name = config["model_name"]
    
    # Configure 4-bit quantization with optimal settings
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, config["compute_dtype"]),
        bnb_4bit_use_double_quant=config["double_quant"],
        bnb_4bit_quant_type=config["quant_type"]
    )
    
    # Create offload directory if needed
    if config["offload_folder"]:
        os.makedirs(config["offload_folder"], exist_ok=True)
    
    # Load tokenizer and model
    print(f"\nLoading model: {model_name}")
    print(f"Using token: {token[:8]}...")
    print("\nModel Configuration:")
    print(f"- Device Map: {config['device_map']}")
    print(f"- Max Memory: {config['max_memory']}")
    print(f"- 4-bit Quantization: {config['load_in_4bit']}")
    print(f"- Compute Dtype: {config['compute_dtype']}")
    print(f"- Quantization Type: {config['quant_type']}")
    print(f"- Double Quantization: {config['double_quant']}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=config["trust_remote_code"]
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=config["device_map"],
        max_memory=config["max_memory"],
        torch_dtype=getattr(torch, config["compute_dtype"]),
        token=token,
        trust_remote_code=config["trust_remote_code"],
        quantization_config=quantization_config,
        offload_folder=config["offload_folder"]
    )
    
    metrics.print_gpu_stats("After model loading")
    metrics.print_model_stats(model)
    
    return model, tokenizer

def demonstrate_system_metrics(model=None):
    """Demonstrate system metrics monitoring."""
    print("\n=== System Metrics Demo ===")
    
    metrics = SystemMetrics()
    metrics.print_system_info(model)

def demonstrate_automated_metrics():
    """Demonstrate automated metrics evaluation."""
    print("\n=== Automated Metrics Demo ===")
    
    metrics = AutomatedMetrics()
    
    # Example responses
    generated = "Python is a high-level programming language known for its simplicity and readability."
    reference = "Python is a popular high-level programming language that emphasizes code readability."
    
    # Evaluate single response
    scores = metrics.evaluate_response(generated, reference)
    print("\nSingle Response Evaluation:")
    for metric, score in scores.items():
        print(f"{metric}: {score:.3f}")
    
    # Evaluate conversation
    conversation = [
        {
            'generated': "Hello! How can I help you today?",
            'reference': "Hi! How may I assist you?"
        },
        {
            'generated': "Python is great for beginners because it's easy to learn.",
            'reference': "Python is an excellent choice for beginners due to its simple syntax."
        }
    ]
    
    conv_scores = metrics.evaluate_conversation(conversation)
    print("\nConversation Evaluation:")
    for metric, score in conv_scores.items():
        print(f"{metric}: {score:.3f}")

def demonstrate_human_evaluation():
    """Demonstrate human evaluation process."""
    print("\n=== Human Evaluation Demo ===")
    
    evaluator = HumanEvaluation(output_dir="evaluation_results")
    
    # Create evaluation form
    conversation = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity."}
    ]
    
    form = evaluator.create_evaluation_form(conversation)
    print("\nCreated evaluation form with ID:", form["conversation_id"])
    
    # Simulate multiple evaluators
    evaluators = [
        {
            "id": "eval1",
            "scores": {
                "Response Relevance (1-5)": 5,
                "Response Accuracy (1-5)": 4,
                "Language Quality (1-5)": 5,
                "Task Completion (1-5)": 4,
                "Overall Quality (1-5)": 4
            },
            "comments": "Good explanation, could include more examples"
        },
        {
            "id": "eval2",
            "scores": {
                "Response Relevance (1-5)": 4,
                "Response Accuracy (1-5)": 5,
                "Language Quality (1-5)": 5,
                "Task Completion (1-5)": 4,
                "Overall Quality (1-5)": 5
            },
            "comments": "Clear and concise explanation"
        }
    ]
    
    # Save evaluations
    for evaluator_data in evaluators:
        evaluator.save_evaluation(
            evaluator_id=evaluator_data["id"],
            form_data=form,
            scores=evaluator_data["scores"],
            comments=evaluator_data["comments"]
        )
    
    # Aggregate results
    aggregate_scores = evaluator.aggregate_evaluations(form["conversation_id"])
    print("\nAggregated Evaluation Scores:")
    for criterion, scores in aggregate_scores.items():
        print(f"{criterion}:")
        print(f"  Mean: {scores['mean']:.2f}")
        print(f"  Std: {scores['std']:.2f}")

def demonstrate_benchmark_suite():
    """Demonstrate benchmark suite functionality."""
    print("\n=== Benchmark Suite Demo ===")
    
    benchmark = BenchmarkSuite(benchmark_dir="benchmark_results")
    
    # Create a simple test set
    test_cases = [
        {
            "input": "What is machine learning?",
            "expected_output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "category": "technical_explanation"
        },
        {
            "input": "Write a Python hello world program",
            "expected_output": "Here's a simple Python hello world program:\n\nprint('Hello, World!')",
            "category": "coding"
        }
    ]
    
    # Create benchmark set
    benchmark.create_benchmark_set(
        test_cases=test_cases,
        name="basic_ml_benchmark",
        description="Basic machine learning concept explanations"
    )
    print("\nCreated benchmark set: basic_ml_benchmark")

def main():
    """Run all demonstrations."""
    args = parse_args()
    
    if args.list_models:
        list_available_models()
        return
    
    print("LLM Evaluation Framework Demo")
    print("=" * 30)
    
    # Load model with specified or default configuration
    model, tokenizer = load_model(args.model)
    
    # Show system metrics first
    demonstrate_system_metrics(model)
    
    # Run other demonstrations
    demonstrate_automated_metrics()
    demonstrate_human_evaluation()
    demonstrate_benchmark_suite()

if __name__ == "__main__":
    main() 