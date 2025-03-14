"""
HuggingFace Fine-Tuning Integration

This module provides utilities for fine-tuning HuggingFace Transformer models,
enabling local fine-tuning on smaller models that can be deployed without API dependencies.
"""

import os
import json
import logging
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Union, Tuple
import datetime
import pandas as pd

# Import HuggingFace modules
try:
    import torch
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        Trainer, 
        TrainingArguments,
        DataCollatorForLanguageModeling
    )
    from datasets import Dataset
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logging.warning("HuggingFace Transformers not available. Please install with: pip install transformers datasets")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='huggingface_finetuning.log'
)

class HuggingFaceFineTuner:
    """
    Handles fine-tuning of HuggingFace models with custom datasets.
    
    This class provides utilities to prepare training data, fine-tune
    models using efficient methods like LoRA, and serve the fine-tuned
    models with the framework.
    """
    
    def __init__(self, model_name="microsoft/phi-2", output_dir="models/fine_tuned"):
        """
        Initialize the HuggingFace fine-tuner.
        
        Args:
            model_name: Base model name or path
            output_dir: Directory to save fine-tuned models
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "HuggingFace libraries not available. "
                "Please install with: pip install transformers datasets torch peft"
            )
        
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.info(f"Initializing HuggingFace fine-tuner with model {model_name} on {self.device}")
        
        # Will be loaded on-demand to save memory
        self.tokenizer = None
        self.model = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _load_model_and_tokenizer(self, use_8bit=False, use_4bit=False):
        """
        Load the model and tokenizer with optional quantization.
        
        Args:
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
        """
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with quantization if specified
            quantization_config = None
            if use_8bit:
                model_kwargs = {"load_in_8bit": True}
            elif use_4bit:
                model_kwargs = {"load_in_4bit": True}
            else:
                model_kwargs = {}
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                **model_kwargs
            )
            
            logging.info(f"Loaded model and tokenizer: {self.model_name}")
            
        except Exception as e:
            logging.error(f"Error loading model and tokenizer: {str(e)}")
            raise
    
    def prepare_data(self, 
                    conversations: List[Dict[str, Any]], 
                    max_seq_length: int = 512) -> Dataset:
        """
        Prepare conversations for fine-tuning by converting to HuggingFace dataset.
        
        Args:
            conversations: List of conversation dictionaries
            max_seq_length: Maximum sequence length for tokenization
            
        Returns:
            HuggingFace Dataset ready for training
        """
        if self.tokenizer is None:
            self._load_model_and_tokenizer()
        
        # Format conversations into text
        formatted_data = []
        for conv in conversations:
            if "messages" in conv and isinstance(conv["messages"], list):
                text = ""
                for message in conv["messages"]:
                    role = message.get("role", "user")
                    content = message.get("content", "")
                    
                    if role == "system":
                        text += f"<|system|>\n{content}\n"
                    elif role == "user":
                        text += f"<|user|>\n{content}\n"
                    elif role == "assistant":
                        text += f"<|assistant|>\n{content}\n"
                
                formatted_data.append({"text": text})
        
        logging.info(f"Formatted {len(formatted_data)} conversations")
        
        # Create dataset
        dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=max_seq_length
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Remove unnecessary columns for training
        tokenized_dataset = tokenized_dataset.remove_columns(
            [col for col in tokenized_dataset.column_names if col not in ["input_ids", "attention_mask"]]
        )
        
        return tokenized_dataset
    
    def fine_tune(self, 
                 dataset: Dataset, 
                 epochs: int = 3, 
                 batch_size: int = 8, 
                 learning_rate: float = 2e-5,
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 use_8bit: bool = False,
                 use_4bit: bool = False) -> str:
        """
        Fine-tune the model on the provided dataset.
        
        Args:
            dataset: HuggingFace dataset for training
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            use_8bit: Whether to use 8-bit quantization
            use_4bit: Whether to use 4-bit quantization
            
        Returns:
            str: Path to the fine-tuned model
        """
        if self.model is None or self.tokenizer is None:
            self._load_model_and_tokenizer(use_8bit=use_8bit, use_4bit=use_4bit)
        
        # Create a unique output directory for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_short = self.model_name.split("/")[-1]
        run_output_dir = os.path.join(self.output_dir, f"{model_name_short}_ft_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)
        
        try:
            # Configure LoRA if requested
            if use_lora:
                logging.info(f"Setting up LoRA with rank={lora_r}, alpha={lora_alpha}")
                
                # Prepare model for LoRA if using quantization
                if use_8bit or use_4bit:
                    self.model = prepare_model_for_kbit_training(self.model)
                
                # Configure LoRA
                lora_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                
                # Apply LoRA to model
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=run_output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                weight_decay=0.01,
                save_strategy="epoch",
                evaluation_strategy="no",
                logging_dir=os.path.join(run_output_dir, "logs"),
                logging_steps=50,
                save_total_limit=2,
                bf16=torch.cuda.is_available(),  # Use bfloat16 if available
                fp16=not torch.cuda.is_available() and not use_8bit and not use_4bit,  # Use fp16 as fallback
                remove_unused_columns=False  # Important for custom datasets
            )
            
            # Create data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator
            )
            
            # Train the model
            logging.info(f"Starting fine-tuning for {epochs} epochs")
            trainer.train()
            
            # Save the model and tokenizer
            if use_lora:
                # For LoRA, save adapter weights
                self.model.save_pretrained(run_output_dir)
                # Also save original model name for reference
                with open(os.path.join(run_output_dir, "base_model_name.json"), "w") as f:
                    json.dump({"base_model": self.model_name}, f)
            else:
                # For full fine-tuning, save the complete model
                trainer.save_model(run_output_dir)
                self.tokenizer.save_pretrained(run_output_dir)
            
            logging.info(f"Fine-tuning completed, model saved to: {run_output_dir}")
            
            # Save training config
            training_config = {
                "base_model": self.model_name,
                "timestamp": timestamp,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "use_lora": use_lora,
                "lora_r": lora_r if use_lora else None,
                "lora_alpha": lora_alpha if use_lora else None,
                "use_8bit": use_8bit,
                "use_4bit": use_4bit,
                "device": self.device,
                "output_dir": run_output_dir
            }
            
            with open(os.path.join(run_output_dir, "training_config.json"), "w") as f:
                json.dump(training_config, f, indent=2)
            
            return run_output_dir
            
        except Exception as e:
            logging.error(f"Error during fine-tuning: {str(e)}")
            raise
    
    def load_fine_tuned_model(self, model_path: str) -> Tuple[Any, Any]:
        """
        Load a fine-tuned model and tokenizer.
        
        Args:
            model_path: Path to the fine-tuned model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Check if this is a LoRA model
            is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            base_model_path = self.model_name
            
            # If LoRA, get base model name from saved file
            if is_lora and os.path.exists(os.path.join(model_path, "base_model_name.json")):
                with open(os.path.join(model_path, "base_model_name.json"), "r") as f:
                    base_model_info = json.load(f)
                    base_model_path = base_model_info.get("base_model", self.model_name)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            if is_lora:
                # For LoRA, load base model first
                from peft import PeftModel
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                
                # Then load adapter weights
                model = PeftModel.from_pretrained(base_model, model_path)
                logging.info(f"Loaded LoRA fine-tuned model from {model_path}")
                
            else:
                # For full fine-tuning, load directly
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                logging.info(f"Loaded fully fine-tuned model from {model_path}")
            
            return model, tokenizer
            
        except Exception as e:
            logging.error(f"Error loading fine-tuned model: {str(e)}")
            raise
    
    def list_fine_tuned_models(self) -> List[Dict[str, Any]]:
        """
        List all available fine-tuned models.
        
        Returns:
            List of dictionaries with model information
        """
        models = []
        
        try:
            if not os.path.exists(self.output_dir):
                return []
            
            for model_dir in os.listdir(self.output_dir):
                model_path = os.path.join(self.output_dir, model_dir)
                
                if not os.path.isdir(model_path):
                    continue
                
                # Look for training config
                config_path = os.path.join(model_path, "training_config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                        config["path"] = model_path
                        models.append(config)
                else:
                    # Basic info if config not found
                    models.append({
                        "path": model_path,
                        "name": model_dir,
                        "timestamp": os.path.getmtime(model_path)
                    })
            
            return models
            
        except Exception as e:
            logging.error(f"Error listing fine-tuned models: {str(e)}")
            return []
    
    def delete_fine_tuned_model(self, model_path: str) -> bool:
        """
        Delete a fine-tuned model.
        
        Args:
            model_path: Path to the model to delete
            
        Returns:
            bool: Success or failure
        """
        try:
            if not os.path.exists(model_path) or not os.path.isdir(model_path):
                logging.warning(f"Model path not found: {model_path}")
                return False
            
            # Check if this is in our output directory to prevent accidental deletion
            if not model_path.startswith(self.output_dir):
                logging.warning(f"Refusing to delete model outside output directory: {model_path}")
                return False
            
            # Delete the model directory
            shutil.rmtree(model_path)
            logging.info(f"Deleted fine-tuned model: {model_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error deleting fine-tuned model: {str(e)}")
            return False
    
    def export_model_for_integration(self, model_path: str, export_dir: str = None) -> str:
        """
        Export a fine-tuned model in a format suitable for integration with the framework.
        For LoRA models, this may merge weights with the base model.
        
        Args:
            model_path: Path to the fine-tuned model
            export_dir: Directory to export to (if None, creates a temp directory)
            
        Returns:
            str: Path to the exported model
        """
        try:
            # Load the model
            model, tokenizer = self.load_fine_tuned_model(model_path)
            
            # Check if it's a LoRA model
            is_lora = hasattr(model, "merge_and_unload")
            
            # Determine export path
            if export_dir is None:
                export_dir = os.path.join(
                    self.output_dir, 
                    f"export_{os.path.basename(model_path)}"
                )
            
            os.makedirs(export_dir, exist_ok=True)
            
            # Export process
            if is_lora:
                # For LoRA models, merge weights with base model
                logging.info("Merging LoRA adapter with base model")
                merged_model = model.merge_and_unload()
                
                # Save merged model
                merged_model.save_pretrained(export_dir)
                tokenizer.save_pretrained(export_dir)
                logging.info(f"Exported merged model to {export_dir}")
                
            else:
                # For full fine-tuning, just copy the model
                model.save_pretrained(export_dir)
                tokenizer.save_pretrained(export_dir)
                logging.info(f"Exported model to {export_dir}")
            
            return export_dir
            
        except Exception as e:
            logging.error(f"Error exporting model: {str(e)}")
            raise
    
    def create_adapter_for_dspy(self, model_path: str) -> Dict[str, Any]:
        """
        Create an adapter configuration for using the fine-tuned model with DSPy.
        
        Args:
            model_path: Path to the fine-tuned model
            
        Returns:
            dict: Configuration for DSPy integration
        """
        # This is a placeholder for future integration with DSPy
        # Current DSPy doesn't directly support local HF models
        
        return {
            "model_type": "huggingface", 
            "model_path": model_path,
            "integration_type": "custom",
            "integration_note": "Requires custom adapters for DSPy integration"
        } 