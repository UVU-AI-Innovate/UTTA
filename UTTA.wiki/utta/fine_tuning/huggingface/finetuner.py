"""
HuggingFace fine-tuning for educational question answering.

This module provides utilities for fine-tuning HuggingFace models with LoRA.
"""

import os
import json
import torch
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from datasets import Dataset

class HuggingFaceFineTuner:
    """
    Fine-tuner for HuggingFace models using LoRA.
    
    This class provides utilities for fine-tuning open source
    language models from HuggingFace with Low-Rank Adaptation (LoRA),
    which is memory-efficient and suitable for consumer GPUs.
    """
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-hf",
        output_dir: str = "lora-finetuned-model",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_4bit: bool = True,
        use_nested_quant: bool = False,
        gpu_memory_limit: Optional[int] = None
    ):
        """
        Initialize the HuggingFace fine-tuner.
        
        Args:
            base_model: HuggingFace model ID (default: "meta-llama/Llama-2-7b-hf")
            output_dir: Directory to save the fine-tuned model (default: "lora-finetuned-model")
            lora_r: LoRA attention dimension (default: 8)
            lora_alpha: LoRA alpha parameter (default: 16)
            lora_dropout: Dropout probability for LoRA layers (default: 0.05)
            use_4bit: Whether to use 4-bit quantization (default: True)
            use_nested_quant: Whether to use nested quantization for 4-bit (default: False)
            gpu_memory_limit: Optional GPU memory limit in GB (default: None)
        """
        self.base_model = base_model
        self.output_dir = output_dir
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.use_nested_quant = use_nested_quant
        
        # Set GPU memory limit if provided
        if gpu_memory_limit is not None and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(gpu_memory_limit / torch.cuda.get_device_properties(0).total_memory)
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and use_4bit:
            print("Warning: 4-bit quantization requested but CUDA is not available. Falling back to CPU.")
    
    def _prepare_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Prepare the model and tokenizer with quantization and LoRA config.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        # Configure quantization
        compute_dtype = torch.float16
        
        bnb_config = None
        if self.use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.use_nested_quant,
            )
        
        # Load base model and tokenizer
        print(f"Loading base model: {self.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=bnb_config,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Prepare model for k-bit training if using quantization
        if self.use_4bit and self.device == "cuda":
            model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self._get_target_modules(model)
        )
        
        # Apply LoRA to model
        model = get_peft_model(model, lora_config)
        
        # Set up gradient checkpointing
        if self.device == "cuda":
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        
        return model, tokenizer
    
    def _get_target_modules(self, model) -> List[str]:
        """
        Get target modules for LoRA based on model architecture.
        
        Args:
            model: The HuggingFace model
            
        Returns:
            List of module names for LoRA
        """
        # Default target modules for common architectures
        model_name_lower = self.base_model.lower()
        
        if "llama" in model_name_lower:
            return ["q_proj", "v_proj"]
        elif "mistral" in model_name_lower:
            return ["q_proj", "v_proj"]
        elif "gpt-neox" in model_name_lower:
            return ["query_key_value"]
        elif "gpt2" in model_name_lower:
            return ["c_attn"]
        elif "bloom" in model_name_lower:
            return ["query_key_value"]
        else:
            # Fallback to all linear layers
            return [name for name, module in model.named_modules() 
                   if isinstance(module, torch.nn.Linear) and "lm_head" not in name]
    
    def prepare_data(
        self,
        dataset: List[Dict[str, str]],
        system_prompt: str,
        input_key: str = "question",
        output_key: str = "answer",
        max_length: int = 1024
    ) -> Dataset:
        """
        Prepare dataset for HuggingFace fine-tuning.
        
        Args:
            dataset: List of examples with input and output keys
            system_prompt: System prompt to use for fine-tuning
            input_key: Key for input text in dataset (default: "question")
            output_key: Key for output text in dataset (default: "answer")
            max_length: Maximum sequence length (default: 1024)
            
        Returns:
            HuggingFace Dataset formatted for training
        """
        # Get tokenizer
        _, tokenizer = self._prepare_model_and_tokenizer()
        
        formatted_data = []
        for example in dataset:
            # Format each example for instruction tuning
            instruction = example[input_key]
            output = example[output_key]
            
            formatted_text = f"<s>[INST] {system_prompt}\n\n{instruction} [/INST] {output}</s>"
            formatted_data.append({"text": formatted_text})
        
        # Create HuggingFace dataset
        hf_dataset = Dataset.from_list(formatted_data)
        
        # Tokenize function
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        # Apply tokenization
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)
        tokenized_dataset = tokenized_dataset.remove_columns(["text"])
        
        # Set format for PyTorch
        tokenized_dataset.set_format("torch")
        
        return tokenized_dataset
    
    def fine_tune(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        num_train_epochs: float = 3,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        lr_scheduler_type: str = "cosine",
        warmup_ratio: float = 0.03,
        logging_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Fine-tune the model with LoRA.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (default: None)
            num_train_epochs: Number of training epochs (default: 3)
            learning_rate: Learning rate (default: 2e-4)
            batch_size: Batch size per device (default: 4)
            gradient_accumulation_steps: Gradient accumulation steps (default: 4)
            lr_scheduler_type: Learning rate scheduler type (default: "cosine")
            warmup_ratio: Warmup ratio (default: 0.03)
            logging_steps: Number of steps between logging (default: 10)
            
        Returns:
            Dictionary with training results and metrics
        """
        # Prepare model and tokenizer
        model, tokenizer = self._prepare_model_and_tokenizer()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            save_strategy="epoch",
            logging_steps=logging_steps,
            report_to="tensorboard",
            remove_unused_columns=False,
            optim="adamw_torch" if self.device == "cuda" else "adamw_hf",
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=transformers.DataCollatorForLanguageModeling(
                tokenizer=tokenizer, 
                mlm=False
            ),
        )
        
        # Train the model
        print("Starting training...")
        train_results = trainer.train()
        
        # Save the model
        trainer.save_model(self.output_dir)
        
        # Save training results
        metrics = train_results.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Evaluate if validation dataset is provided
        if val_dataset is not None:
            eval_metrics = trainer.evaluate(val_dataset)
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        
        # Compile results
        results = {
            "base_model": self.base_model,
            "lora_config": {
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout
            },
            "training": {
                "epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps
            },
            "metrics": metrics,
            "saved_path": self.output_dir
        }
        
        if val_dataset is not None:
            results["eval_metrics"] = eval_metrics
        
        return results
    
    def generate(
        self,
        question: str,
        system_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        model_path: Optional[str] = None
    ) -> str:
        """
        Generate an answer using the fine-tuned model.
        
        Args:
            question: Question to answer
            system_prompt: System prompt to use
            max_new_tokens: Maximum new tokens to generate (default: 256)
            temperature: Temperature for generation (default: 0.7)
            top_p: Top-p probability for nucleus sampling (default: 0.9)
            top_k: Top-k for sampling (default: 50)
            repetition_penalty: Penalty for repetition (default: 1.1)
            model_path: Path to the fine-tuned model (default: None, uses self.output_dir)
            
        Returns:
            Generated answer
        """
        model_path = model_path or self.output_dir
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Format the prompt for instruction tuning
        prompt = f"<s>[INST] {system_prompt}\n\n{question} [/INST]"
        
        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (after [/INST])
        if "[/INST]" in response:
            answer = response.split("[/INST]")[1].strip()
        else:
            answer = response
            
        return answer 