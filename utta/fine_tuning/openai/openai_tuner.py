"""
OpenAI Fine-Tuning Integration

This module provides utilities for fine-tuning OpenAI models,
allowing users to create custom models for specific educational 
domains and tasks.
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union
import datetime

# Import OpenAI
from openai import OpenAI

# Import DSPy handler for integration
from src.llm.dspy.handler import DSPyLLMHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='openai_finetuning.log'
)

class OpenAIFineTuner:
    """
    Handles the fine-tuning of OpenAI models with custom datasets.
    
    This class provides utilities to prepare training data, start
    fine-tuning jobs, monitor progress, and create handlers for
    fine-tuned models.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the OpenAI fine-tuner.
        
        Args:
            api_key: OpenAI API key (optional, will use env var if not provided)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logging.error("No OpenAI API key provided or found in environment")
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.fine_tuned_model_id = None
        
        # Create cache dir if it doesn't exist
        os.makedirs("cache/fine_tuning", exist_ok=True)
    
    def prepare_data(self, 
                    conversations: List[Dict[str, Any]], 
                    output_file: str = "cache/fine_tuning/training_data.jsonl") -> str:
        """
        Convert conversations to JSONL format for fine-tuning.
        
        Args:
            conversations: List of conversation dictionaries
            output_file: Path to save the JSONL file
            
        Returns:
            str: Path to the prepared data file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Validate conversations format
        valid_conversations = []
        for conv in conversations:
            if "messages" in conv and isinstance(conv["messages"], list):
                # Check for required roles
                has_system = any(msg.get("role") == "system" for msg in conv["messages"])
                has_user = any(msg.get("role") == "user" for msg in conv["messages"])
                has_assistant = any(msg.get("role") == "assistant" for msg in conv["messages"])
                
                if has_user and has_assistant:
                    valid_conversations.append(conv)
                else:
                    logging.warning(f"Skipping conversation missing required roles: {conv}")
            else:
                logging.warning(f"Skipping invalid conversation format: {conv}")
        
        logging.info(f"Found {len(valid_conversations)} valid conversations out of {len(conversations)}")
        
        # Write to JSONL file
        with open(output_file, 'w') as f:
            for conv in valid_conversations:
                f.write(json.dumps(conv) + "\n")
        
        logging.info(f"Prepared training data saved to {output_file}")
        return output_file
    
    def start_fine_tuning(self, 
                         training_file: str, 
                         model: str = "gpt-3.5-turbo", 
                         suffix: str = None) -> str:
        """
        Start a fine-tuning job with OpenAI.
        
        Args:
            training_file: Path to the JSONL training data file
            model: Base model to fine-tune
            suffix: Custom suffix for the fine-tuned model name
            
        Returns:
            str: Job ID of the fine-tuning job
        """
        # Generate a default suffix if none provided
        if not suffix:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"ft_{timestamp}"
        
        try:
            # Upload the training file
            with open(training_file, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            file_id = file_response.id
            logging.info(f"File uploaded with ID: {file_id}")
            
            # Wait for file to be processed (important for fine-tuning)
            self._wait_for_file_processing(file_id)
            
            # Create fine-tuning job
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
                suffix=suffix
            )
            
            job_id = response.id
            logging.info(f"Fine-tuning job created with ID: {job_id}")
            
            # Save job info for future reference
            self._save_job_info(job_id, model, suffix, training_file)
            
            return job_id
            
        except Exception as e:
            logging.error(f"Error starting fine-tuning job: {str(e)}")
            raise
    
    def _wait_for_file_processing(self, file_id: str, max_wait_seconds: int = 60):
        """
        Wait for a file to be processed by OpenAI.
        
        Args:
            file_id: ID of the uploaded file
            max_wait_seconds: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            try:
                file_info = self.client.files.retrieve(file_id)
                if file_info.status == "processed":
                    logging.info(f"File {file_id} has been processed")
                    return
                else:
                    logging.info(f"File {file_id} status: {file_info.status}. Waiting...")
                    time.sleep(5)
            except Exception as e:
                logging.warning(f"Error checking file status: {str(e)}")
                time.sleep(5)
        
        logging.warning(f"Exceeded wait time for file processing: {file_id}")
    
    def _save_job_info(self, job_id: str, model: str, suffix: str, training_file: str):
        """
        Save job information to a local file for tracking.
        
        Args:
            job_id: Fine-tuning job ID
            model: Base model
            suffix: Model suffix
            training_file: Path to training data
        """
        job_info = {
            "job_id": job_id,
            "base_model": model,
            "suffix": suffix,
            "training_file": training_file,
            "start_time": datetime.datetime.now().isoformat(),
            "status": "created"
        }
        
        # Save to JSON file
        os.makedirs("cache/fine_tuning/jobs", exist_ok=True)
        with open(f"cache/fine_tuning/jobs/{job_id}.json", "w") as f:
            json.dump(job_info, f, indent=2)
    
    def check_fine_tuning_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check the status of a fine-tuning job.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            Dict with job status information
        """
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            # Update job info file with status
            self._update_job_status(job_id, response.status)
            
            # If completed, save the fine-tuned model ID
            if response.status == "succeeded" and hasattr(response, "fine_tuned_model"):
                self.fine_tuned_model_id = response.fine_tuned_model
                logging.info(f"Fine-tuned model created: {self.fine_tuned_model_id}")
                self._update_job_status(job_id, response.status, self.fine_tuned_model_id)
            
            return {
                "status": response.status,
                "created_at": response.created_at,
                "finished_at": response.finished_at,
                "fine_tuned_model": getattr(response, "fine_tuned_model", None),
                "train_loss": getattr(response, "training_metrics", {}).get("train_loss", None),
                "train_accuracy": getattr(response, "training_metrics", {}).get("train_accuracy", None)
            }
            
        except Exception as e:
            logging.error(f"Error checking fine-tuning status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _update_job_status(self, job_id: str, status: str, model_id: str = None):
        """
        Update the job status in the tracking file.
        
        Args:
            job_id: Fine-tuning job ID
            status: Current job status
            model_id: Fine-tuned model ID (if available)
        """
        job_file = f"cache/fine_tuning/jobs/{job_id}.json"
        
        if not os.path.exists(job_file):
            logging.warning(f"Job file not found: {job_file}")
            return
        
        try:
            with open(job_file, "r") as f:
                job_info = json.load(f)
            
            job_info["status"] = status
            job_info["last_checked"] = datetime.datetime.now().isoformat()
            
            if model_id:
                job_info["fine_tuned_model_id"] = model_id
            
            with open(job_file, "w") as f:
                json.dump(job_info, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error updating job status: {str(e)}")
    
    def list_fine_tuned_models(self) -> List[Dict[str, Any]]:
        """
        List all available fine-tuned models.
        
        Returns:
            List of dictionaries with model information
        """
        try:
            models = self.client.models.list()
            
            # Filter for fine-tuned models
            fine_tuned_models = [
                {
                    "id": model.id,
                    "created_at": model.created,
                    "owned_by": model.owned_by
                }
                for model in models.data
                if "ft-" in model.id
            ]
            
            return fine_tuned_models
            
        except Exception as e:
            logging.error(f"Error listing fine-tuned models: {str(e)}")
            return []
    
    def list_fine_tuning_jobs(self) -> List[Dict[str, Any]]:
        """
        List all fine-tuning jobs.
        
        Returns:
            List of dictionaries with job information
        """
        try:
            # Try to get from OpenAI API
            jobs = self.client.fine_tuning.jobs.list(limit=10)
            
            return [
                {
                    "id": job.id,
                    "model": job.model,
                    "status": job.status,
                    "created_at": job.created_at,
                    "fine_tuned_model": getattr(job, "fine_tuned_model", None)
                }
                for job in jobs.data
            ]
            
        except Exception as e:
            logging.error(f"Error listing fine-tuning jobs: {str(e)}")
            
            # Fallback to local cache
            try:
                jobs_dir = "cache/fine_tuning/jobs"
                if not os.path.exists(jobs_dir):
                    return []
                
                jobs = []
                for filename in os.listdir(jobs_dir):
                    if filename.endswith(".json"):
                        with open(os.path.join(jobs_dir, filename), "r") as f:
                            job_info = json.load(f)
                            jobs.append(job_info)
                
                return jobs
                
            except Exception as inner_e:
                logging.error(f"Error reading local job cache: {str(inner_e)}")
                return []
    
    def create_handler_with_fine_tuned_model(self, fine_tuned_model_id: str) -> DSPyLLMHandler:
        """
        Create a DSPyLLMHandler with the specified fine-tuned model.
        
        Args:
            fine_tuned_model_id: ID of the fine-tuned model
            
        Returns:
            DSPyLLMHandler: Handler configured to use the fine-tuned model
        """
        try:
            # Validate that the model exists
            self.client.models.retrieve(fine_tuned_model_id)
            
            # Create handler with fine-tuned model
            handler = DSPyLLMHandler(model_name=fine_tuned_model_id)
            
            logging.info(f"Created handler with fine-tuned model: {fine_tuned_model_id}")
            return handler
            
        except Exception as e:
            logging.error(f"Error creating handler with fine-tuned model: {str(e)}")
            raise ValueError(f"Could not create handler with model {fine_tuned_model_id}: {str(e)}")
    
    def cancel_fine_tuning_job(self, job_id: str) -> bool:
        """
        Cancel a fine-tuning job that is in progress.
        
        Args:
            job_id: Fine-tuning job ID
            
        Returns:
            bool: Success or failure
        """
        try:
            self.client.fine_tuning.jobs.cancel(job_id)
            self._update_job_status(job_id, "cancelled")
            logging.info(f"Cancelled fine-tuning job: {job_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error cancelling fine-tuning job: {str(e)}")
            return False
    
    def delete_fine_tuned_model(self, model_id: str) -> bool:
        """
        Delete a fine-tuned model.
        
        Args:
            model_id: Fine-tuned model ID
            
        Returns:
            bool: Success or failure
        """
        try:
            self.client.models.delete(model_id)
            logging.info(f"Deleted fine-tuned model: {model_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting fine-tuned model: {str(e)}")
            return False 