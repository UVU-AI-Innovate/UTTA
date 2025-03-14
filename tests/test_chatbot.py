#!/usr/bin/env python3
"""
Test Script for UTTA

This script tests the basic functionality of the reorganized UTTA package.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_imports():
    """Test that we can import the core components."""
    print("Testing imports...")
    
    # Try importing core components
    try:
        from utta.core import VectorDatabase, DocumentProcessor
        print("✅ Successfully imported core components")
    except ImportError as e:
        print(f"❌ Failed to import core components: {e}")
        return False
    
    # Try importing chatbot components
    try:
        from utta.chatbot import DSPyLLMHandler
        print("✅ Successfully imported chatbot components")
    except ImportError as e:
        print(f"❌ Failed to import chatbot components: {e}")
        return False
    
    # Try importing fine-tuning components
    try:
        from utta.fine_tuning import OpenAIFineTuner, DSPyOptimizer
        print("✅ Successfully imported fine-tuning components")
    except ImportError as e:
        print(f"❌ Failed to import fine-tuning components: {e}")
        return False
    
    return True

def test_vector_db():
    """Test that the vector database works."""
    print("\nTesting VectorDatabase functionality...")
    
    try:
        from utta.core import VectorDatabase
        
        # Initialize vector DB
        vector_db = VectorDatabase()
        
        # Add sample data
        vector_db.add_chunks([
            {
                "text": "This is a test document for searching.",
                "metadata": {"source": "test"}
            }
        ])
        
        # Search for relevant content
        results = vector_db.search("test document", top_k=1)
        
        if results and len(results) > 0:
            print("✅ VectorDatabase search returned results")
            return True
        else:
            print("❌ VectorDatabase search failed to return results")
            return False
            
    except Exception as e:
        print(f"❌ VectorDatabase test failed with error: {e}")
        return False

if __name__ == "__main__":
    print("Running UTTA Package Tests\n" + "="*30)
    
    imports_ok = test_imports()
    
    if imports_ok:
        vector_db_ok = test_vector_db()
    
    print("\nTest Summary:")
    print(f"Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    
    if imports_ok:
        print(f"VectorDatabase: {'✅ PASS' if vector_db_ok else '❌ FAIL'}")
    
    print("="*30) 