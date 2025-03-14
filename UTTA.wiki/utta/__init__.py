"""
UTTA - Utah Teacher Training Assistant

A comprehensive framework for developing and evaluating LLM-based teaching assistants.
"""

__version__ = "0.1.0"

# Import core components for easy access
from utta.core import DocumentProcessor, VectorDatabase
from utta.chatbot import DSPyLLMHandler
from utta.fine_tuning import OpenAIFineTuner, DSPyOptimizer

# Make commonly used classes available at the package level
__all__ = [
    "DocumentProcessor",
    "VectorDatabase",
    "DSPyLLMHandler",
    "OpenAIFineTuner",
    "DSPyOptimizer"
]

# Try to import HuggingFace tuner if it exists
try:
    from utta.fine_tuning import HuggingFaceFineTuner
    __all__.append("HuggingFaceFineTuner")
except ImportError:
    pass
