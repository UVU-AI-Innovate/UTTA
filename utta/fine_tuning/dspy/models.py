"""
DSPy models for educational question answering.

This module defines the DSPy models used for educational question answering.
"""

import dspy

class EducationalQAModel(dspy.Module):
    """
    A DSPy module for educational question answering.
    
    This model uses Chain-of-Thought reasoning to generate detailed
    educational answers to questions. It's designed to be optimized
    using the DSPy framework without changing model weights.
    """
    
    def __init__(self):
        """Initialize the educational QA model."""
        super().__init__()
        # Define the signature for chain of thought reasoning
        self.generate_answer = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question: str) -> dspy.Prediction:
        """
        Generate an educational answer for the given question.
        
        Args:
            question: The educational question to answer
            
        Returns:
            A prediction containing the generated answer
        """
        # Generate a detailed educational answer with chain-of-thought
        prediction = self.generate_answer(question=question)
        return dspy.Prediction(answer=prediction.answer) 