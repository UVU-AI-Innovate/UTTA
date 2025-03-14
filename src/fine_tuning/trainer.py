"""Fine-tuning trainer for pedagogical responses."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any
import dspy
from tqdm import tqdm

class PedagogicalTrainer:
    """Handles fine-tuning for pedagogical response generation."""
    
    def __init__(self):
        """Initialize the trainer."""
        self.model = dspy.LM("gpt-3.5-turbo")
        
        # Create signature for teaching responses
        class TeachingSignature(dspy.Signature):
            """Signature for teaching responses."""
            query: str
            context: str
            response: str
            metadata: dict
        
        self.generator = dspy.ChainOfThought(TeachingSignature)
        
        # Create storage directory
        self.storage_dir = Path("data/fine_tuning")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("Initialized PedagogicalTrainer")
    
    def prepare_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Prepare training data for fine-tuning.
        
        Args:
            data_path: Path to the training data file
            
        Returns:
            List of prepared training examples
        """
        try:
            # Load training data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            # Validate and format data
            prepared_data = []
            for example in data:
                if all(k in example for k in ['query', 'context', 'response', 'metadata']):
                    prepared_data.append({
                        'query': example['query'],
                        'context': example['context'],
                        'response': example['response'],
                        'metadata': example['metadata']
                    })
            
            logging.info(f"Prepared {len(prepared_data)} training examples")
            return prepared_data
            
        except Exception as e:
            logging.error(f"Error preparing training data: {e}")
            return []
    
    def train(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the model on the prepared data.
        
        Args:
            training_data: List of prepared training examples
            
        Returns:
            Dictionary containing training results
        """
        try:
            logging.info("Starting training process")
            
            # Create training signatures
            signatures = []
            for example in tqdm(training_data, desc="Creating signatures"):
                signature = self.generator(
                    query=example['query'],
                    context=example['context'],
                    response=example['response'],
                    metadata=example['metadata']
                )
                signatures.append(signature)
            
            # Optimize the model
            optimizer = dspy.ChainOfThoughtOptimizer(
                metric=self._evaluate_response,
                max_rounds=3
            )
            
            optimized_generator = optimizer.optimize(
                self.generator,
                signatures,
                num_threads=4
            )
            
            # Save the optimized model
            model_path = self.storage_dir / "optimized_model.dspy"
            dspy.save(optimized_generator, model_path)
            
            results = {
                "status": "completed",
                "examples_trained": len(training_data),
                "model_path": str(model_path)
            }
            
            logging.info("Training completed successfully")
            return results
            
        except Exception as e:
            logging.error(f"Error during training: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _evaluate_response(self,
                         predicted: dspy.Prediction,
                         target: dspy.Prediction) -> float:
        """Evaluate the quality of a predicted response.
        
        Args:
            predicted: Predicted response
            target: Target response
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            # Basic evaluation metrics
            metrics = {
                "length_ratio": min(
                    len(predicted.response) / len(target.response),
                    len(target.response) / len(predicted.response)
                ),
                "context_usage": self._evaluate_context_usage(
                    predicted.response,
                    predicted.context
                ),
                "pedagogical_elements": self._count_pedagogical_elements(
                    predicted.response
                )
            }
            
            # Combine metrics
            score = (
                0.3 * metrics["length_ratio"] +
                0.4 * metrics["context_usage"] +
                0.3 * metrics["pedagogical_elements"]
            )
            
            return min(max(score, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error in response evaluation: {e}")
            return 0.0
    
    def _evaluate_context_usage(self, response: str, context: str) -> float:
        """Evaluate how well the response uses the given context."""
        try:
            # Simple word overlap metric
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            overlap = len(response_words.intersection(context_words))
            total = len(context_words)
            
            return overlap / total if total > 0 else 0.0
            
        except Exception as e:
            logging.error(f"Error in context usage evaluation: {e}")
            return 0.0
    
    def _count_pedagogical_elements(self, response: str) -> float:
        """Count pedagogical elements in the response."""
        try:
            elements = {
                "question": ["?", "how", "what", "why", "can you"],
                "example": ["for example", "like", "such as"],
                "explanation": ["because", "therefore", "this means"],
                "engagement": ["let's", "try", "imagine"]
            }
            
            count = 0
            response_lower = response.lower()
            
            for element_type, markers in elements.items():
                for marker in markers:
                    if marker in response_lower:
                        count += 1
                        break
            
            return min(count / len(elements), 1.0)
            
        except Exception as e:
            logging.error(f"Error counting pedagogical elements: {e}")
            return 0.0 