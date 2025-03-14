"""
UTTA Core Module

This module provides the core functionality for document processing and vector storage.
"""

from .document_processor import DocumentProcessor
from .vector_database import VectorDatabase

__all__ = ["DocumentProcessor", "VectorDatabase"]
