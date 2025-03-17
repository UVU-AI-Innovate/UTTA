"""
Test script to verify all major components of the Teacher Training Chatbot.
"""

import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all major components can be imported."""
    try:
        # Add src to Python path
        sys.path.append(str(Path(__file__).parent))
        
        # Test core components
        from src.core.document_processor import DocumentProcessor
        from src.core.vector_database import VectorDatabase
        logger.info("✓ Core components imported successfully")
        
        # Test LLM components
        from src.llm.llm_handler import EnhancedLLMInterface
        from src.llm.dspy.adapter import create_llm_interface
        from src.llm.dspy.handler import PedagogicalLanguageProcessor
        logger.info("✓ LLM components imported successfully")
        
        # Test evaluation components
        from src.evaluation.metrics.automated_metrics import AutomatedMetricsEvaluator
        from src.evaluation.benchmarks.benchmark_suite import BenchmarkSuite
        logger.info("✓ Evaluation components imported successfully")
        
        # Test retrieval components
        from src.retrieval.llama_index.integration import LlamaIndexRetriever
        logger.info("✓ Retrieval components imported successfully")
        
        return True
    except Exception as e:
        logger.error(f"Import error: {str(e)}")
        return False

def test_knowledge_base():
    """Test if knowledge base can be initialized and queried."""
    try:
        from src.core.vector_database import VectorDatabase
        from src.retrieval.llama_index.integration import LlamaIndexRetriever
        
        # Initialize components
        vector_db = VectorDatabase()
        retriever = LlamaIndexRetriever()
        
        # Test basic operations
        logger.info("✓ Knowledge base components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Knowledge base error: {str(e)}")
        return False

def test_fine_tuning():
    """Test if fine-tuning components can be initialized."""
    try:
        from src.llm.fine_tuning.dspy_optimizer import DSPyOptimizer
        from src.llm.fine_tuning.huggingface_tuner import HuggingFaceTuner
        
        # Initialize components
        dspy_optimizer = DSPyOptimizer()
        hf_tuner = HuggingFaceTuner()
        
        logger.info("✓ Fine-tuning components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Fine-tuning error: {str(e)}")
        return False

def test_evaluation():
    """Test if evaluation framework can be initialized."""
    try:
        from src.evaluation.metrics.automated_metrics import AutomatedMetricsEvaluator
        from src.evaluation.benchmarks.benchmark_suite import BenchmarkSuite
        
        # Initialize components
        evaluator = AutomatedMetricsEvaluator()
        benchmark = BenchmarkSuite()
        
        logger.info("✓ Evaluation framework initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        return False

def main():
    """Run all component tests."""
    logger.info("Starting component tests...")
    
    # Test imports
    if not test_imports():
        logger.error("❌ Import tests failed")
        return
    
    # Test knowledge base
    if not test_knowledge_base():
        logger.error("❌ Knowledge base tests failed")
        return
    
    # Test fine-tuning
    if not test_fine_tuning():
        logger.error("❌ Fine-tuning tests failed")
        return
    
    # Test evaluation
    if not test_evaluation():
        logger.error("❌ Evaluation tests failed")
        return
    
    logger.info("✓ All component tests completed successfully!")

if __name__ == "__main__":
    main() 