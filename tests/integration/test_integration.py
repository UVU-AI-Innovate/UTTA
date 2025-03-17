#!/usr/bin/env python3
"""Integration test for knowledge base, fine-tuning, and evaluation components."""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from src.llm.dspy.handler import EnhancedDSPyLLMInterface
from src.knowledge_base.document_indexer import DocumentIndexer
from src.knowledge_base.knowledge_graph import KnowledgeGraph
from src.fine_tuning.trainer import PedagogicalTrainer
from src.evaluation.metrics.automated_metrics import AutomatedMetricsEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('integration_test.log')
    ]
)

class IntegrationTester:
    """Tests the integration of various components."""
    
    def __init__(self):
        """Initialize components for testing."""
        self.llm = EnhancedDSPyLLMInterface(model_name="gpt-3.5-turbo")
        self.document_indexer = DocumentIndexer()
        self.knowledge_graph = KnowledgeGraph()
        self.trainer = PedagogicalTrainer()
        self.evaluator = AutomatedMetricsEvaluator()
        
        # Create test directories if they don't exist
        Path("data/test").mkdir(parents=True, exist_ok=True)
        Path("data/test/documents").mkdir(parents=True, exist_ok=True)
        Path("data/test/fine_tuning").mkdir(parents=True, exist_ok=True)
        Path("data/test/evaluation").mkdir(parents=True, exist_ok=True)
    
    def create_test_document(self) -> str:
        """Create a test document for knowledge base testing."""
        content = """
        # Teaching Strategies for Science Education
        
        ## Inquiry-Based Learning
        Inquiry-based learning is a teaching method that encourages students to:
        1. Ask questions
        2. Investigate phenomena
        3. Draw conclusions
        
        ## Scaffolding Techniques
        Effective scaffolding involves:
        - Breaking down complex concepts
        - Providing guided practice
        - Gradually removing support
        
        ## Assessment Methods
        Regular assessment helps track student progress through:
        - Formative assessments
        - Project-based evaluation
        - Peer review sessions
        """
        
        test_doc_path = "data/test/documents/teaching_strategies.md"
        with open(test_doc_path, "w") as f:
            f.write(content)
        
        logging.info(f"Created test document at {test_doc_path}")
        return test_doc_path
    
    def test_knowledge_integration(self) -> Dict[str, Any]:
        """Test knowledge base integration."""
        logging.info("\n=== Testing Knowledge Base Integration ===")
        
        # Create and index test document
        doc_path = self.create_test_document()
        indexed_doc = self.document_indexer.index_document(doc_path)
        logging.info(f"Indexed document with {len(indexed_doc['chunks'])} chunks")
        
        # Create knowledge graph nodes
        self.knowledge_graph.add_concept("inquiry_based_learning", {
            "name": "Inquiry-Based Learning",
            "type": "teaching_strategy",
            "description": "A method that encourages student questions and investigation"
        })
        logging.info("Added concepts to knowledge graph")
        
        # Test retrieval
        query = "What are effective scaffolding techniques?"
        retrieved_context = self.document_indexer.retrieve_relevant_chunks(query)
        logging.info(f"Retrieved {len(retrieved_context)} relevant chunks for query")
        
        # Generate response with retrieved context
        response = self.llm.generate_teaching_response(
            student_query=query,
            student_profile={"level": "beginner", "subject": "science"},
            context=retrieved_context
        )
        
        results = {
            "indexed_chunks": len(indexed_doc['chunks']),
            "retrieved_chunks": len(retrieved_context),
            "response": response
        }
        
        logging.info("Knowledge base integration test completed")
        return results
    
    def test_fine_tuning(self) -> Dict[str, Any]:
        """Test fine-tuning process."""
        logging.info("\n=== Testing Fine-Tuning Process ===")
        
        # Prepare training data
        training_data = [
            {
                "query": "How do I explain photosynthesis to 5th graders?",
                "context": "Plants use sunlight to make their own food through photosynthesis",
                "response": "Think of plants as tiny factories that make their own food using sunlight as power",
                "metadata": {"grade_level": "5th", "subject": "science"}
            },
            {
                "query": "What's the best way to teach fractions?",
                "context": "Students often struggle with understanding fractions as parts of a whole",
                "response": "Let's use pizza slices to understand how fractions work in real life",
                "metadata": {"grade_level": "4th", "subject": "math"}
            }
        ]
        
        # Save training data
        train_path = "data/test/fine_tuning/training_data.json"
        with open(train_path, "w") as f:
            json.dump(training_data, f, indent=2)
        
        # Prepare and train
        prepared_data = self.trainer.prepare_data(train_path)
        training_results = self.trainer.train(prepared_data)
        
        results = {
            "training_examples": len(training_data),
            "training_results": training_results
        }
        
        logging.info("Fine-tuning test completed")
        return results
    
    def test_evaluation(self) -> Dict[str, Any]:
        """Test evaluation metrics."""
        logging.info("\n=== Testing Evaluation Metrics ===")
        
        # Test responses to evaluate
        test_responses = [
            {
                "teaching_response": "Let's think about photosynthesis like a cooking recipe. Plants need three main ingredients: sunlight, water, and carbon dioxide. Just like you need specific ingredients to make cookies, plants need these ingredients to make their food. Would you like to draw a picture of this process?",
                "student_profile": {
                    "grade_level": "5th",
                    "learning_style": "visual"
                },
                "context": "Teaching photosynthesis to elementary students"
            },
            {
                "teaching_response": "Fractions are just parts of a whole. If we cut a pizza into 8 equal slices, each slice is 1/8 of the whole pizza. How many slices would you need to have 3/8 of the pizza?",
                "student_profile": {
                    "grade_level": "4th",
                    "learning_style": "hands-on"
                },
                "context": "Introduction to fractions"
            }
        ]
        
        # Evaluate responses
        evaluation_results = []
        for response in test_responses:
            metrics = self.evaluator.evaluate_response(
                teaching_response=response["teaching_response"],
                student_profile=response["student_profile"],
                context=response["context"]
            )
            evaluation_results.append(metrics)
        
        results = {
            "num_evaluations": len(evaluation_results),
            "evaluation_results": evaluation_results
        }
        
        logging.info("Evaluation test completed")
        return results
    
    def run_all_tests(self):
        """Run all integration tests and save results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Run all tests
            kb_results = self.test_knowledge_integration()
            ft_results = self.test_fine_tuning()
            eval_results = self.test_evaluation()
            
            # Combine results
            all_results = {
                "timestamp": timestamp,
                "knowledge_base": kb_results,
                "fine_tuning": ft_results,
                "evaluation": eval_results
            }
            
            # Save results
            results_path = f"data/test/integration_results_{timestamp}.json"
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2)
            
            logging.info(f"\n=== All integration tests completed ===")
            logging.info(f"Results saved to: {results_path}")
            
            # Print summary
            print("\n=== Integration Test Summary ===")
            print(f"Knowledge Base Integration:")
            print(f"- Indexed chunks: {kb_results['indexed_chunks']}")
            print(f"- Retrieved chunks: {kb_results['retrieved_chunks']}")
            print(f"\nFine-Tuning:")
            print(f"- Training examples: {ft_results['training_examples']}")
            print(f"- Training completion: {ft_results['training_results']['status']}")
            print(f"\nEvaluation:")
            print(f"- Responses evaluated: {eval_results['num_evaluations']}")
            print(f"- Average clarity score: {sum(r['clarity_score'] for r in eval_results['evaluation_results']) / len(eval_results['evaluation_results']):.2f}")
            
        except Exception as e:
            logging.error(f"Error during integration testing: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    tester = IntegrationTester()
    tester.run_all_tests() 