"""
Interface for interacting with Language Models (LLMs).
Provides a standardized way to communicate with different LLM backends.
"""

import os
import requests
from typing import Dict, Any, Optional

class LLMInterface:
    """Interface for interacting with Language Models."""
    
    def __init__(self):
        """Initialize the LLM interface with default settings."""
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
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt for the model
            context: Optional context information
            **kwargs: Additional parameters for the model
            
        Returns:
            Dictionary containing the model's response and any additional information
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
        
        Args:
            prompt: The base prompt
            context: Context information to include
            
        Returns:
            Complete prompt string with context
        """
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        return f"Context:\n{context_str}\n\nPrompt:\n{prompt}" 