"""
DSPy optimization for educational question answering.

This module provides the optimizer for DSPy models, focusing on
optimizing educational question answering without changing model weights.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

class DSPyOptimizer:
    """
    Optimizer for DSPy educational question answering models.
    
    This class wraps the DSPy optimization process for educational QA,
    providing a simple interface for optimizing the model using the
    BootstrapFewShot technique.
    """
    
    def __init__(
        self,
        model,
        model_name: str = "gpt-3.5-turbo",
        max_tokens: int = 1024,
        temperature: float = 0.0
    ):
        """
        Initialize the DSPy optimizer.
        
        Args:
            model: The DSPy model to optimize
            model_name: The name of the LLM to use (default: gpt-3.5-turbo)
            max_tokens: Maximum number of tokens to generate (default: 1024)
            temperature: Temperature for generation (default: 0.0)
        """
        self.model = model
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize the LLM
        self.lm = dspy.OpenAI(
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        dspy.settings.configure(lm=self.lm)
        
        # Store the original and optimized modules
        self.original_module = model
        self.optimized_module = None
        
    def optimize(
        self,
        train_data: List[Dict[str, str]],
        num_iterations: int = 3,
        metric: str = "answer_quality"
    ):
        """
        Optimize the model using the provided training data.
        
        Args:
            train_data: List of training examples with 'question' and 'answer' keys
            num_iterations: Number of optimization iterations (default: 3)
            metric: Metric to use for optimization (default: "answer_quality")
        """
        # Convert data to DSPy format
        dspy_examples = []
        for example in train_data:
            dspy_examples.append(
                dspy.Example(
                    question=example["question"],
                    answer=example["answer"]
                ).with_inputs("question")
            )
        
        # Define the evaluation metric
        def evaluate_answer_quality(example, pred):
            """
            Evaluate the quality of the predicted answer.
            
            This function measures how well the predicted answer 
            captures the key information from the reference answer.
            
            Args:
                example: The reference example with the correct answer
                pred: The model's prediction
                
            Returns:
                A dictionary with the accuracy score
            """
            # Extract key concepts from the reference answer
            # This is a simple implementation - in practice, you might use
            # more sophisticated methods like semantic similarity
            key_concepts = example.answer.lower().split('.')[:3]
            concept_matches = 0
            
            # Count how many key concepts appear in the prediction
            for concept in key_concepts:
                if concept and concept.strip() and concept.strip() in pred.answer.lower():
                    concept_matches += 1
            
            # Calculate accuracy score
            if len(key_concepts) > 0:
                accuracy = concept_matches / len(key_concepts)
            else:
                accuracy = 0.0
                
            return {"accuracy": accuracy}
        
        # Create the optimizer
        teleprompter = BootstrapFewShot(metric=evaluate_answer_quality)
        
        # Run the optimization
        print(f"Starting optimization with {len(dspy_examples)} examples...")
        self.optimized_module = teleprompter.compile(
            self.model,
            trainset=dspy_examples,
            valset=dspy_examples[:min(len(dspy_examples), 3)],
            rounds=num_iterations
        )
        
        print("Optimization complete!")
        return self.optimized_module
    
    def predict(self, question: str) -> str:
        """
        Generate an answer using the optimized model.
        
        Args:
            question: The question to answer
            
        Returns:
            The generated answer
        """
        if self.optimized_module is None:
            raise ValueError("Model has not been optimized yet. Call optimize() first.")
        
        # Generate a prediction
        prediction = self.optimized_module(question=question)
        return prediction.answer
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: List of test examples with 'question' and 'answer' keys
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.optimized_module is None:
            raise ValueError("Model has not been optimized yet. Call optimize() first.")
        
        # Initialize metrics
        metrics = {
            "initial_accuracy": 0.0,
            "optimized_accuracy": 0.0,
            "improvement": 0.0
        }
        
        # Evaluate on test data
        initial_correct = 0
        optimized_correct = 0
        
        for example in test_data:
            question = example["question"]
            reference = example["answer"]
            
            # Get predictions from both models
            initial_pred = self.original_module(question=question)
            optimized_pred = self.optimized_module(question=question)
            
            # Simple evaluation based on keyword overlap
            initial_score = self._calculate_overlap(initial_pred.answer, reference)
            optimized_score = self._calculate_overlap(optimized_pred.answer, reference)
            
            if initial_score > 0.5:
                initial_correct += 1
            
            if optimized_score > 0.5:
                optimized_correct += 1
        
        # Calculate metrics
        metrics["initial_accuracy"] = initial_correct / len(test_data)
        metrics["optimized_accuracy"] = optimized_correct / len(test_data)
        
        if metrics["initial_accuracy"] > 0:
            metrics["improvement"] = (
                (metrics["optimized_accuracy"] - metrics["initial_accuracy"]) 
                / metrics["initial_accuracy"] * 100
            )
        
        return metrics
    
    def save(self, path: Union[str, Path]):
        """
        Save the optimized module to a file.
        
        Args:
            path: Path to save the module
        """
        if self.optimized_module is None:
            raise ValueError("Model has not been optimized yet. Call optimize() first.")
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save as a JSON file
        with open(path, "w") as f:
            # Get the optimized prompt
            if hasattr(self.optimized_module, "generate_answer"):
                prompt = self.optimized_module.generate_answer.teleprompter
                json.dump(
                    {
                        "model_name": self.model_name,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "optimized_prompt": prompt
                    }, 
                    f, 
                    indent=2
                )
            else:
                raise ValueError("Could not extract optimized prompt from module")
    
    def _calculate_overlap(self, prediction: str, reference: str) -> float:
        """
        Calculate the overlap between prediction and reference.
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            Overlap score between 0 and 1
        """
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        if len(ref_words) == 0:
            return 0.0
        
        overlap = len(pred_words.intersection(ref_words)) / len(ref_words)
        return overlap 