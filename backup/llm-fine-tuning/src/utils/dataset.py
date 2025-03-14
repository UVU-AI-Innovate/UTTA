"""
Dataset utilities for loading, preprocessing, and splitting datasets.

This module provides common functions for working with datasets
across different fine-tuning approaches.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path

def load_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the loaded data
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

def save_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> bool:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return True
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")
        return False

def split_dataset(
    dataset: List[Dict[str, Any]],
    test_size: float = 0.2,
    seed: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split a dataset into training and testing sets.
    
    Args:
        dataset: List of data examples
        test_size: Proportion of the dataset to include in the test split (0.0-1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    if seed is not None:
        random.seed(seed)
    
    # Shuffle dataset
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    # Calculate split point
    test_samples = max(1, int(len(dataset) * test_size))
    
    # Split the dataset
    test_data = shuffled[:test_samples]
    train_data = shuffled[test_samples:]
    
    return train_data, test_data

def format_qa_pairs(
    dataset: List[Dict[str, Any]],
    input_key: str = "question",
    output_key: str = "answer",
    include_metadata: bool = False
) -> List[Dict[str, Any]]:
    """
    Format dataset into standardized question-answer pairs.
    
    Args:
        dataset: List of data examples
        input_key: Key for input text in dataset
        output_key: Key for output text in dataset
        include_metadata: Whether to include additional metadata fields
        
    Returns:
        List of formatted question-answer pairs
    """
    formatted_data = []
    
    for example in dataset:
        if input_key not in example or output_key not in example:
            continue
            
        formatted_example = {
            "question": example[input_key],
            "answer": example[output_key]
        }
        
        # Include additional metadata if requested
        if include_metadata:
            for key, value in example.items():
                if key not in [input_key, output_key, "question", "answer"]:
                    formatted_example[key] = value
        
        formatted_data.append(formatted_example)
    
    return formatted_data

def balance_categories(
    dataset: List[Dict[str, Any]],
    category_key: str = "category",
    max_per_category: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Balance a dataset to have equal representation from each category.
    
    Args:
        dataset: List of data examples
        category_key: Key for category field in dataset
        max_per_category: Maximum examples per category (if None, uses the smallest category size)
        
    Returns:
        Balanced dataset
    """
    # Count examples per category
    categories = {}
    for example in dataset:
        if category_key in example:
            category = example[category_key]
            if category not in categories:
                categories[category] = []
            categories[category].append(example)
    
    if not categories:
        return dataset  # No categories found
    
    # Determine target size per category
    if max_per_category is None:
        min_category_size = min(len(examples) for examples in categories.values())
        target_size = min_category_size
    else:
        target_size = max_per_category
    
    # Balance dataset
    balanced_dataset = []
    for category, examples in categories.items():
        if len(examples) > target_size:
            # Randomly sample examples
            sampled = random.sample(examples, target_size)
            balanced_dataset.extend(sampled)
        else:
            balanced_dataset.extend(examples)
    
    return balanced_dataset 