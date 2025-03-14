"""
OpenAI fine-tuning for educational question answering.

This module provides utilities for fine-tuning OpenAI models through their API.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from openai import OpenAI
import tiktoken

class OpenAIFineTuner:
    """
    Fine-tuner for OpenAI models using their API.
    
    This class provides utilities for preparing data, starting fine-tuning jobs,
    and using fine-tuned OpenAI models for educational question answering.
    """
    
    def __init__(
        self,
        base_model: str = "gpt-3.5-turbo",
        suffix: str = None,
        n_epochs: int = 3,
        api_key: str = None
    ):
        """
        Initialize the OpenAI fine-tuner.
        
        Args:
            base_model: The base model to fine-tune (default: "gpt-3.5-turbo")
            suffix: Suffix to add to the fine-tuned model name (default: None)
            n_epochs: Number of epochs for fine-tuning (default: 3)
            api_key: OpenAI API key (default: None, will use OPENAI_API_KEY env var)
        """
        self.base_model = base_model
        self.suffix = suffix or f"ft-{int(time.time())}"
        self.n_epochs = n_epochs
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
    
    def prepare_data(
        self,
        dataset: List[Dict[str, str]],
        system_prompt: str,
        input_key: str = "question",
        output_key: str = "answer"
    ) -> List[Dict[str, Any]]:
        """
        Prepare dataset for OpenAI fine-tuning.
        
        Args:
            dataset: List of examples with input and output keys
            system_prompt: System prompt to use for fine-tuning
            input_key: Key for input text in dataset (default: "question")
            output_key: Key for output text in dataset (default: "answer")
            
        Returns:
            List of examples formatted for OpenAI fine-tuning
        """
        prepared_data = []
        
        for example in dataset:
            # Format each example as a chat message
            formatted_example = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example[input_key]},
                    {"role": "assistant", "content": example[output_key]}
                ]
            }
            prepared_data.append(formatted_example)
        
        return prepared_data
    
    def start_finetuning(
        self,
        data_path: Union[str, Path],
        batch_size: int = None,
        learning_rate_multiplier: float = None
    ) -> str:
        """
        Start a fine-tuning job with OpenAI.
        
        Args:
            data_path: Path to the JSONL file with prepared data
            batch_size: Batch size for training (default: auto)
            learning_rate_multiplier: Learning rate multiplier (default: auto)
            
        Returns:
            The fine-tuning job ID
        """
        # Upload the training file
        print("Uploading training file...")
        with open(data_path, "rb") as f:
            response = self.client.files.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = response.id
        print(f"File uploaded with ID: {file_id}")
        
        # Create fine-tuning job parameters
        job_params = {
            "training_file": file_id,
            "model": self.base_model,
            "suffix": self.suffix,
            "hyperparameters": {
                "n_epochs": self.n_epochs,
            }
        }
        
        # Add optional parameters if provided
        if batch_size:
            job_params["hyperparameters"]["batch_size"] = batch_size
        
        if learning_rate_multiplier:
            job_params["hyperparameters"]["learning_rate_multiplier"] = learning_rate_multiplier
        
        # Create the fine-tuning job
        print("Creating fine-tuning job...")
        response = self.client.fine_tuning.jobs.create(**job_params)
        job_id = response.id
        print(f"Fine-tuning job created with ID: {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: The ID of the fine-tuning job
            
        Returns:
            Dictionary with job status information
        """
        response = self.client.fine_tuning.jobs.retrieve(job_id)
        return {
            "status": response.status,
            "fine_tuned_model": response.fine_tuned_model,
            "created_at": response.created_at,
            "finished_at": response.finished_at,
            "trained_tokens": response.trained_tokens,
            "error": response.error,
        }
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 60,
        max_wait_time: int = 3600
    ) -> Dict[str, Any]:
        """
        Wait for a fine-tuning job to complete.
        
        Args:
            job_id: The ID of the fine-tuning job
            poll_interval: Time in seconds between status checks (default: 60)
            max_wait_time: Maximum time to wait in seconds (default: 3600)
            
        Returns:
            Status dictionary with final status
        """
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status = self.get_job_status(job_id)
            
            if status["status"] in ["succeeded", "failed", "cancelled"]:
                return status
            
            print(f"Job status: {status['status']}. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)
        
        print(f"Maximum wait time ({max_wait_time}s) exceeded. Job still running.")
        return self.get_job_status(job_id)
    
    def generate(
        self,
        model_name: str,
        question: str,
        system_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer using a fine-tuned model.
        
        Args:
            model_name: Name of the fine-tuned model
            question: Question to answer
            system_prompt: System prompt to use
            max_tokens: Maximum tokens to generate (default: 1024)
            temperature: Temperature for generation (default: 0.7)
            
        Returns:
            Generated answer
        """
        response = self.client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
    
    def calculate_token_count(
        self,
        dataset: List[Dict[str, Any]],
        model: str = "gpt-3.5-turbo"
    ) -> Dict[str, int]:
        """
        Calculate the number of tokens in the dataset.
        
        Args:
            dataset: List of examples prepared for fine-tuning
            model: Model to use for token counting (default: "gpt-3.5-turbo")
            
        Returns:
            Dictionary with token count information
        """
        encoding = tiktoken.encoding_for_model(model)
        n_examples = len(dataset)
        total_tokens = 0
        
        for example in dataset:
            for message in example["messages"]:
                total_tokens += len(encoding.encode(message["content"])) + 4  # 4 tokens for role
            total_tokens += 2  # 2 tokens for message separator
        
        return {
            "n_examples": n_examples,
            "total_tokens": total_tokens,
            "tokens_per_example": total_tokens / n_examples if n_examples > 0 else 0,
            "estimated_cost": (total_tokens / 1000) * 0.008  # Approximate cost in USD
        } 