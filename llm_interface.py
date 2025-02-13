"""
Language Model Interface Module

This module provides a standardized interface for interacting with Language
Models (LLMs) in the teaching simulation system. It handles all communication
with the LLM backend while providing error handling and context management.

Key Features:
    - Standardized LLM communication
    - Context-aware prompt building
    - Error handling and recovery
    - Configurable model parameters
    - Response formatting

Components:
    - LLMInterface: Main class for LLM interaction
    - Prompt Building: Context-aware prompt construction
    - Response Processing: Standardized response handling
    - Error Management: Graceful error handling

Dependencies:
    - requests: For API communication
    - os: For environment variables
    - typing: For type hints

Example:
    llm = LLMInterface()
    response = llm.generate(
        prompt="Explain addition to a second grader",
        context={"grade_level": "2nd", "subject": "math"}
    )
"""

import os
import requests
from typing import Dict, Any, Optional

class LLMInterface:
    """
    Interface for interacting with Language Models in educational contexts.
    
    This class provides a standardized way to communicate with LLM backends
    while maintaining educational context and handling errors gracefully.
    It supports context-aware prompt generation and response processing.
    
    Features:
        - Flexible model configuration
        - Context-aware prompt building
        - Error handling and recovery
        - Response standardization
        - Timeout management
    
    Attributes:
        base_url (str): Base URL for the LLM API
        model (str): Name of the LLM model to use
        max_tokens (int): Maximum response length
        temperature (float): Response randomness (0-1)
    
    Example:
        llm = LLMInterface()
        response = llm.generate(
            prompt="How would you explain fractions?",
            context={"student_level": "beginner"}
        )
    """
    
    def __init__(self):
        """
        Initialize the LLM interface with default settings.
        
        Sets up:
            - API endpoint configuration
            - Default model selection
            - Response parameters
            - Temperature settings
        
        The defaults can be overridden in individual generate() calls.
        """
        self.base_url = "http://localhost:11434/api"
        self.model = "mistral"  # Default model
        self.max_tokens = 2000
        self.temperature = 0.7
        
    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM with context awareness.
        
        This method handles the complete process of:
            1. Building the context-aware prompt
            2. Making the API request
            3. Processing the response
            4. Handling any errors
        
        Args:
            prompt (str): The main instruction or question for the LLM
            context (Optional[Dict[str, Any]]): Additional context including:
                - student_info: Student characteristics
                - subject_area: Current subject
                - grade_level: Student's grade
                - previous_context: Prior interactions
            **kwargs: Additional parameters including:
                - temperature: Response randomness (0-1)
                - max_tokens: Maximum response length
                - stream: Whether to stream the response
        
        Returns:
            Dict[str, Any]: A structured response containing:
                - text: The LLM's response
                - status: Success or error status
                - model: Name of the model used
                - error: Error message if applicable
        
        Example:
            response = llm.generate(
                prompt="Explain multiplication",
                context={
                    "grade_level": "2nd",
                    "learning_style": "visual"
                },
                temperature=0.8
            )
        """
        try:
            # Combine prompt with context if provided
            full_prompt = self._build_prompt(prompt, context) if context else prompt
            
            # Make API call to the LLM
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            
            return {
                "text": response.json()["response"],
                "status": "success",
                "model": self.model
            }
            
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            return {
                "text": "I apologize, but I'm having trouble processing that request.",
                "status": "error",
                "error": str(e)
            }
    
    def _build_prompt(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Build a complete prompt by combining the input with context.
        
        This method creates a structured prompt that includes both the
        main instruction and any relevant context information in a
        format optimized for LLM understanding.
        
        Args:
            prompt (str): The main instruction or question
            context (Dict[str, Any]): Contextual information including:
                - Student characteristics
                - Learning objectives
                - Previous interactions
                - Environmental factors
            
        Returns:
            str: A formatted prompt string that combines:
                - Context information
                - Main prompt
                - Any necessary formatting or structure
        
        Note:
            The context is formatted in a clear, hierarchical manner
            to ensure the LLM can effectively utilize the information.
        """
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        return f"Context:\n{context_str}\n\nPrompt:\n{prompt}" 