"""
DSPy Optimizer Module

This module provides optimization capabilities for DSPy-based LLM interfaces,
allowing for "soft fine-tuning" through prompt optimization rather than
actual model parameter updates.
"""

import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import dspy
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dspy_optimization.log'
)

class DSPyOptimizer:
    """
    DSPy Optimizer for prompt-based optimization of LLM performance.
    
    This class provides methods to optimize prompts for specific tasks
    using DSPy's teleprompt capabilities. It serves as a form of "soft fine-tuning"
    by optimizing the instruction and few-shot examples rather than model weights.
    """
    
    def __init__(self, llm_handler=None):
        """
        Initialize the DSPy Optimizer.
        
        Args:
            llm_handler: Optional DSPyLLMHandler instance
        """
        self.llm_handler = llm_handler
        self.optimized_modules = {}
    
    def create_training_examples(self, examples: List[Dict[str, Any]]) -> List[Tuple]:
        """
        Convert raw examples into the format expected by DSPy optimizers.
        
        Args:
            examples: List of dictionaries containing input and output pairs
            
        Returns:
            List of tuples in the format (input_dict, output_dict)
        """
        formatted_examples = []
        for example in examples:
            if 'input' in example and 'output' in example:
                formatted_examples.append((example['input'], example['output']))
        
        logging.info(f"Created {len(formatted_examples)} training examples")
        return formatted_examples
    
    def optimize_module(self, 
                        module_name: str, 
                        training_examples: List[Dict[str, Any]],
                        metric: str = "exact_match",
                        method: str = "bootstrap_few_shot",
                        max_demos: int = 5) -> Any:
        """
        Optimize a specific DSPy module using the provided training examples.
        
        Args:
            module_name: Name of the module to optimize (must exist in the LLM interface)
            training_examples: List of input/output examples for optimization
            metric: Metric to use for optimization (exact_match, rouge, etc.)
            method: Optimization method to use
            max_demos: Maximum number of demos to use
            
        Returns:
            The optimized module
        """
        if not self.llm_handler:
            raise ValueError("LLM Handler must be provided for optimization")
        
        # Get the module to optimize
        if not hasattr(self.llm_handler.interface, module_name):
            raise ValueError(f"Module {module_name} not found in LLM interface")
        
        target_module = getattr(self.llm_handler.interface, module_name)
        
        # Convert examples to DSPy format
        dspy_examples = self.create_training_examples(training_examples)
        
        if len(dspy_examples) == 0:
            raise ValueError("No valid training examples provided")
        
        # Select optimization method
        if method == "bootstrap_few_shot":
            optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=max_demos)
        elif method == "bootstrap_random_search":
            optimizer = BootstrapFewShotWithRandomSearch(metric=metric, max_bootstrapped_demos=max_demos)
        else:
            logging.warning(f"Unknown optimization method: {method}. Using BootstrapFewShot.")
            optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=max_demos)
        
        logging.info(f"Starting optimization of module {module_name} with {len(dspy_examples)} examples")
        
        # Perform optimization
        try:
            optimized_module = optimizer.optimize(
                module=target_module,
                trainset=dspy_examples
            )
            
            # Store optimized module
            self.optimized_modules[module_name] = optimized_module
            
            # Replace original module with optimized version
            setattr(self.llm_handler.interface, module_name, optimized_module)
            
            logging.info(f"Successfully optimized module {module_name}")
            return optimized_module
            
        except Exception as e:
            logging.error(f"Optimization failed: {str(e)}")
            raise
    
    def save_optimized_module(self, module_name: str, path: str) -> bool:
        """
        Save an optimized module to disk.
        
        Args:
            module_name: Name of the module to save
            path: Directory path to save to
            
        Returns:
            bool: Success or failure
        """
        if module_name not in self.optimized_modules:
            logging.error(f"Module {module_name} not found in optimized modules")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save the module with DSPy's save method
            self.optimized_modules[module_name].save(f"{path}/{module_name}")
            logging.info(f"Successfully saved optimized module {module_name} to {path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save optimized module: {str(e)}")
            return False
    
    def load_optimized_module(self, module_name: str, path: str) -> Any:
        """
        Load an optimized module from disk.
        
        Args:
            module_name: Name of the module to load
            path: Directory path to load from
            
        Returns:
            The loaded module
        """
        try:
            # Get the module class from the LLM interface
            if not hasattr(self.llm_handler.interface, module_name):
                logging.error(f"Module {module_name} not found in LLM interface")
                return None
            
            # Load the module with DSPy's load method
            module_class = getattr(self.llm_handler.interface, module_name).__class__
            loaded_module = module_class.load(f"{path}/{module_name}")
            
            # Store and replace the module
            self.optimized_modules[module_name] = loaded_module
            setattr(self.llm_handler.interface, module_name, loaded_module)
            
            logging.info(f"Successfully loaded optimized module {module_name} from {path}")
            return loaded_module
            
        except Exception as e:
            logging.error(f"Failed to load optimized module: {str(e)}")
            return None
            
    def evaluate_module(self, 
                       module_name: str, 
                       test_examples: List[Dict[str, Any]],
                       metric: str = "exact_match") -> Dict[str, Any]:
        """
        Evaluate a module's performance on test examples.
        
        Args:
            module_name: Name of the module to evaluate
            test_examples: List of test examples
            metric: Metric to use for evaluation
            
        Returns:
            Dict with evaluation results
        """
        if not hasattr(self.llm_handler.interface, module_name):
            raise ValueError(f"Module {module_name} not found in LLM interface")
        
        module = getattr(self.llm_handler.interface, module_name)
        dspy_examples = self.create_training_examples(test_examples)
        
        if len(dspy_examples) == 0:
            raise ValueError("No valid test examples provided")
        
        # Create evaluator
        evaluator = dspy.evaluate.Evaluate(
            metric=metric,
            task=module.__class__.__name__
        )
        
        # Perform evaluation
        try:
            results = evaluator.evaluate(
                module=module,
                testset=dspy_examples,
                num_threads=1,
                display_progress=True
            )
            
            logging.info(f"Evaluation results for {module_name}: {results}")
            return results
            
        except Exception as e:
            logging.error(f"Evaluation failed: {str(e)}")
            raise 