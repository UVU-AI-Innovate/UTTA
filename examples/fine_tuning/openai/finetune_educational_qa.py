#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating OpenAI fine-tuning for educational question answering.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parents[2]))

from src.openai.finetuner import OpenAIFineTuner
from src.utils import load_jsonl, save_jsonl

def main():
    """Run the OpenAI fine-tuning example."""
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
    
    # Initialize the OpenAI fine-tuner
    print("Initializing OpenAI fine-tuner...")
    finetuner = OpenAIFineTuner(
        base_model="gpt-3.5-turbo",
        suffix="educational-qa",
        n_epochs=3  # Reduced for example purposes
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
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    prepared_data_path = output_dir / "prepared_data.jsonl"
    
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
    
    # In a real application, you might want to wait for the job to complete
    # and then use the fine-tuned model, but for this example, we'll just
    # simulate using the model once it's ready
    
    print("\nSimulating using the fine-tuned model (once it's ready):")
    model_name = f"ft:{finetuner.base_model}:{finetuner.suffix}"
    print(f"Fine-tuned model name will be: {model_name}")
    
    # Example of how to use the fine-tuned model
    print("\nExample usage (after fine-tuning completes):")
    print("```python")
    print("from openai import OpenAI")
    print("client = OpenAI()")
    print(f"response = client.chat.completions.create(")
    print(f"    model=\"{model_name}\",")
    print(f"    messages=[")
    print(f"        {{\"role\": \"system\", \"content\": \"You are an educational assistant that provides accurate, detailed, and helpful answers to student questions.\"}},")
    print(f"        {{\"role\": \"user\", \"content\": \"What is the theory of relativity?\"")
    print(f"    ]")
    print(f")")
    print("print(response.choices[0].message.content)")
    print("```")
    
    # Save the fine-tuning configuration
    config = {
        "base_model": finetuner.base_model,
        "suffix": finetuner.suffix,
        "job_id": job_id,
        "model_name": model_name,
        "system_prompt": "You are an educational assistant that provides accurate, detailed, and helpful answers to student questions."
    }
    
    with open(output_dir / "finetuning_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nFine-tuning configuration saved to {output_dir / 'finetuning_config.json'}")

if __name__ == "__main__":
    main() 