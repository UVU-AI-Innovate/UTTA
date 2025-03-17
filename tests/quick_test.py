#!/usr/bin/env python3
"""
Quick test script for verifying UTTA functionality.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import components
try:
    from src.llm.dspy.handler import EnhancedDSPyLLMInterface
    from src.knowledge_base.document_indexer import DocumentIndexer
    from src.metrics.automated_metrics import AutomatedMetricsEvaluator
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    sys.exit(1)

def test_components():
    """Test initialization of main components."""
    
    # Test LLM Interface
    try:
        llm = EnhancedDSPyLLMInterface()
        logger.info("✅ LLM Interface initialized successfully")
    except Exception as e:
        logger.error(f"❌ LLM Interface initialization failed: {e}")
        return False
    
    # Test Document Indexer
    try:
        indexer = DocumentIndexer()
        logger.info("✅ Document Indexer initialized successfully")
    except Exception as e:
        logger.error(f"❌ Document Indexer initialization failed: {e}")
        return False
    
    # Test Metrics Evaluator
    try:
        evaluator = AutomatedMetricsEvaluator()
        logger.info("✅ Metrics Evaluator initialized successfully")
    except Exception as e:
        logger.error(f"❌ Metrics Evaluator initialization failed: {e}")
        return False
    
    return True

def main():
    """Run the component initialization test."""
    print("\n=== UTTA Components Test ===\n")
    
    # Create essential directories
    os.makedirs("data/cache", exist_ok=True)
    os.makedirs("data/index", exist_ok=True)
    os.makedirs("data/uploads", exist_ok=True)
    
    # Run only the components test
    try:
        result = test_components()
        print(f"\nResult: {'✅ PASS' if result else '❌ FAIL'}")
        return result
    except Exception as e:
        print(f"\nResult: ❌ FAIL - {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 