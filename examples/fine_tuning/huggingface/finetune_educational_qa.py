#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating HuggingFace fine-tuning for educational question answering.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parents[2]))

from src.huggingface.finetuner import HuggingFaceFineTuner
from src.utils import load_jsonl, save_jsonl

def main():
    """Run the HuggingFace fine-tuning example."""
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
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("Warning: Training on CPU will be slow. Consider using a GPU for faster training.")
    
    # Initialize the HuggingFace fine-tuner
    print("Initializing HuggingFace fine-tuner...")
    finetuner = HuggingFaceFineTuner(
        base_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Using a smaller model for example purposes
        output_dir=str(Path(__file__).parent / "output" / "finetuned-educational-qa"),
        device=device,
        lora_r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.05,  # LoRA dropout
        use_8bit=True if device == "cuda" else False,  # Use 8-bit quantization if on GPU
    )
    
    # Prepare the dataset for fine-tuning
    print("Preparing dataset for fine-tuning...")
    train_dataset = finetuner.prepare_data(
        dataset=dataset,
        system_prompt="You are an educational assistant that provides accurate, detailed, and helpful answers to student questions.",
        input_key="question",
        output_key="answer",
        train_test_split=0.8  # 80% for training, 20% for evaluation
    )
    
    # Start fine-tuning
    print("Starting fine-tuning...")
    finetuner.train(
        train_dataset=train_dataset,
        num_train_epochs=3,  # Reduced for example purposes
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
    
    # Example of how to use the fine-tuned model
    print("\nTesting the fine-tuned model...")
    test_question = "What is the theory of relativity?"
    
    response = finetuner.generate(
        prompt=test_question,
        system_prompt="You are an educational assistant that provides accurate, detailed, and helpful answers to student questions.",
        max_new_tokens=512,
        temperature=0.7,
    )
    
    print(f"\nQuestion: {test_question}")
    print(f"Answer: {response}")
    
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
    
    output_dir = Path(finetuner.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\nModel information saved to {output_dir / 'model_info.json'}")
    print(f"Fine-tuned model saved to {output_dir}")
    
    print("\nExample usage of the fine-tuned model:")
    print("```python")
    print("from src.huggingface.finetuner import HuggingFaceFineTuner")
    print("")
    print("# Load the fine-tuned model")
    print(f"finetuner = HuggingFaceFineTuner.from_pretrained(")
    print(f"    base_model=\"TinyLlama/TinyLlama-1.1B-Chat-v1.0\",")
    print(f"    model_path=\"{output_dir}\"")
    print(f")")
    print("")
    print("# Generate a response")
    print("response = finetuner.generate(")
    print("    prompt=\"What is quantum mechanics?\",")
    print("    system_prompt=\"You are an educational assistant that provides accurate, detailed, and helpful answers to student questions.\",")
    print("    max_new_tokens=512,")
    print("    temperature=0.7,")
    print(")")
    print("")
    print("print(response)")
    print("```")

if __name__ == "__main__":
    main() 