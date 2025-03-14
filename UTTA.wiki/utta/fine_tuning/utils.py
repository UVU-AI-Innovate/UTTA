#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for the LLM fine-tuning project.
"""

import os
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for item in reader:
            data.append(item)
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file
    """
    with jsonlines.open(file_path, 'w') as writer:
        for item in data:
            writer.write(item)

def ensure_dir(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        Path object for the directory
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def format_openai_messages(
    system_prompt: str,
    user_input: str,
    assistant_output: str
) -> List[Dict[str, str]]:
    """
    Format messages for OpenAI API in the chat format.
    
    Args:
        system_prompt: System prompt
        user_input: User input
        assistant_output: Assistant output
        
    Returns:
        List of message dictionaries
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output}
    ]

def format_hf_prompt(
    system_prompt: str,
    user_input: str,
    assistant_output: Optional[str] = None,
    template: str = "chatml"
) -> str:
    """
    Format prompt for HuggingFace models using different templates.
    
    Args:
        system_prompt: System prompt
        user_input: User input
        assistant_output: Optional assistant output (for training examples)
        template: Template format to use (chatml, alpaca, etc.)
        
    Returns:
        Formatted prompt string
    """
    if template == "chatml":
        # ChatML format (used by many models)
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{user_input}<|im_end|>\n"
        if assistant_output is not None:
            prompt += f"<|im_start|>assistant\n{assistant_output}<|im_end|>"
        else:
            prompt += f"<|im_start|>assistant\n"
    
    elif template == "alpaca":
        # Alpaca instruction format
        prompt = f"### Instruction:\n{system_prompt}\n\n"
        prompt += f"### Input:\n{user_input}\n\n"
        prompt += f"### Response:\n"
        if assistant_output is not None:
            prompt += assistant_output
    
    elif template == "llama2":
        # Llama2 chat format
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        prompt += f"{user_input} [/INST] "
        if assistant_output is not None:
            prompt += assistant_output
    
    else:
        raise ValueError(f"Unknown template format: {template}")
    
    return prompt

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate simple evaluation metrics for generated text.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary of metrics
    """
    # This is a very simple implementation
    # In a real application, you would use more sophisticated metrics
    # like BLEU, ROUGE, or custom domain-specific metrics
    
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    # Simple length-based metrics
    avg_pred_len = sum(len(p.split()) for p in predictions) / len(predictions)
    avg_ref_len = sum(len(r.split()) for r in references) / len(references)
    
    # Simple word overlap metric
    total_overlap = 0
    for pred, ref in zip(predictions, references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        if len(ref_words) > 0:
            overlap = len(pred_words.intersection(ref_words)) / len(ref_words)
            total_overlap += overlap
    
    avg_overlap = total_overlap / len(predictions)
    
    return {
        "avg_prediction_length": avg_pred_len,
        "avg_reference_length": avg_ref_len,
        "word_overlap": avg_overlap
    }

def split_dataset(
    dataset: List[Dict[str, Any]], 
    train_ratio: float = 0.8,
    seed: int = 42
) -> tuple:
    """
    Split a dataset into training and testing sets.
    
    Args:
        dataset: List of data examples
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    import random
    random.seed(seed)
    
    # Create a shuffled copy of the dataset
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    # Split the dataset
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    
    return train_data, test_data 