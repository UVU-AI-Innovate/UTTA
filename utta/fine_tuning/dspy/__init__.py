"""
DSPy Optimization module for fine-tuning through prompt engineering.

This module provides tools for optimizing prompts using the DSPy framework,
which allows for improving model performance without changing weights.
"""

from .models import EducationalQAModel
from .optimizer import DSPyOptimizer

__all__ = ["EducationalQAModel", "DSPyOptimizer"] 