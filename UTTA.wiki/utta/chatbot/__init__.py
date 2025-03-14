"""
UTTA Chatbot Module

This module provides functionality for interacting with LLMs.
"""

try:
    from .dspy.handler import DSPyLLMHandler
    from .llm_handler import LLMInterface
    
    __all__ = ["DSPyLLMHandler", "LLMInterface"]
except ImportError as e:
    print(f"Warning: Some chatbot components could not be imported: {e}")
    __all__ = []
