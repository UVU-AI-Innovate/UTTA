#!/usr/bin/env python3
"""
Run Enhanced Teacher Training Agent

This script demonstrates the enhanced teacher training agent with knowledge base integration.
The agent processes educational books, stores knowledge in a vector database, and uses this
knowledge to generate realistic classroom management scenarios.

Usage:
    python -m examples.teacher_training.run

Required directory structure:
    ./data/sample_documents/ - Directory containing educational textbooks/PDFs

Example:
    1. Place educational books in PDF, DOCX, or TXT format in ./data/sample_documents/
    2. Run this script
    3. Follow the interactive prompts to experience the enhanced training
"""

import os
import sys
import time
import json
import logging
import argparse
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import core components
from src.core.document_processor import DocumentProcessor
from src.core.vector_database import VectorDatabase

# Import scenario generator
from examples.teacher_training.scenario_generator import ClassroomScenarioGenerator
from examples.teacher_training.evaluator import ResponseEvaluator
from examples.teacher_training.student_simulation import StudentSimulator

# Import LLM components
from src.llm.dspy.handler import DSPyLLMHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("teacher_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
BOOKS_DIR = os.path.join(KNOWLEDGE_DIR, 'sample_documents')
VECTOR_DB_PATH = os.path.join(KNOWLEDGE_DIR, 'vector_db.sqlite')
STUDENT_PROFILES_PATH = os.path.join(KNOWLEDGE_DIR, 'student_profiles.json')

# Required packages - core functionality will work with just these
CORE_REQUIREMENTS = [
    "numpy",
    "faiss-gpu",  # Will check for either faiss-cpu or faiss-gpu
    "sentence-transformers"
]

# Optional packages for better performance
OPTIONAL_REQUIREMENTS = [
    "PyMuPDF",  # Better PDF extraction (alternative to PyPDF2)
    "tqdm"      # Progress bars
]

def check_dependencies():
    """Check if required dependencies are installed and install them if needed."""
    missing_core = []
    
    # Check core requirements
    for package in CORE_REQUIREMENTS:
        try:
            # Special handling for FAISS - check for either CPU or GPU version
            if package == "faiss-gpu":
                try:
                    # Try importing faiss (will work with either faiss-cpu or faiss-gpu)
                    import faiss
                    # Successfully imported, skip adding to missing packages
                    continue
                except ImportError:
                    # Neither version is installed, mark as missing
                    missing_core.append(package)
            else:
                # Normal check for other packages
                importlib.import_module(package.replace("-", "_").split(">=")[0])
        except ImportError:
            missing_core.append(package)
    
    # For core packages, offer to install them
    if missing_core:
        print(f"\nMissing required dependencies: {', '.join(missing_core)}")
        choice = input("Would you like to install them now? (y/n): ")
        if choice.lower() == 'y':
            for package in missing_core:
                try:
                    print(f"Installing {package}...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package]
                    )
                except subprocess.CalledProcessError as e:
                    print(f"Error installing {package}: {e}")
                    return False
            print("Required dependencies installed successfully!")
            return True
        else:
            print("Required packages must be installed to continue.")
            print("pip install faiss-gpu sentence-transformers")
            return False
    
    # Check optional packages (just inform user, don't exit)
    missing_optional = []
    for package in OPTIONAL_REQUIREMENTS:
        try:
            importlib.import_module(package.replace("-", "_").split(">=")[0])
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        print(f"\nSome optional dependencies are missing: {', '.join(missing_optional)}")
        print("For optimal performance, consider installing them:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True

def print_banner():
    """Print a welcome banner."""
    print("\n" + "="*80)
    print("ENHANCED TEACHER TRAINING AGENT".center(80))
    print("Scientific Book-Based Classroom Management Training".center(80))
    print("="*80)
    print("\nThis enhanced system uses scientific educational books to provide")
    print("evidence-based classroom management scenarios and teaching strategies.")
    print("\nThe system:")
    print("1. Processes scientific educational books from ./data/sample_documents/")
    print("2. Extracts and categorizes research-based knowledge")
    print("3. Generates realistic teaching scenarios based on educational research")
    print("4. Provides feedback and strategies supported by educational literature")
    print("="*80)

def ensure_knowledge_directory():
    """Ensure the knowledge base directory exists and contains scientific educational books."""
    if not os.path.exists(BOOKS_DIR) or not os.listdir(BOOKS_DIR):
        print("\n⚠️ No scientific educational books found in the knowledge base directory.")
        print("For optimal performance and evidence-based training, please add educational books.")
        print("\nRecommended book types:")
        print("- Elementary education textbooks")
        print("- Classroom management research")
        print("- Child development and psychology literature")
        print("- Teaching methodologies and pedagogical books")
        print(f"\nAdd these books to: {os.path.abspath(BOOKS_DIR)}")
        print(f"\nTemporarily using built-in strategies, but these are not as effective as")
        print(f"using scientific literature. The system works best with actual books.")
        return False
    
    print(f"\n✓ Found scientific educational books in {os.path.abspath(BOOKS_DIR)}")
    print("The system will use these books to provide evidence-based training.")
    return True

def knowledge_base_exists():
    """Check if a knowledge base already exists."""
    # Common locations for vector database storage
    kb_dir = os.path.join(KNOWLEDGE_DIR)
    common_indicators = ["vectors", "embeddings", "index", "faiss", "db"]
    
    if not os.path.exists(kb_dir):
        return False
    
    # Check for indicator files or directories
    for item in os.listdir(kb_dir):
        item_path = os.path.join(kb_dir, item)
        
        # Check for indicator directories
        if os.path.isdir(item_path) and any(ind in item.lower() for ind in common_indicators):
            if os.listdir(item_path):  # Not empty
                return True
        
        # Check for indicator files
        if os.path.isfile(item_path) and any(ind in item.lower() for ind in common_indicators):
            if os.path.getsize(item_path) > 100:  # Not empty
                return True
    
    return False

def generate_knowledge_base_report():
    """Generate a detailed report on the scientific knowledge base."""
    book_files = [f for f in os.listdir(BOOKS_DIR) if os.path.isfile(os.path.join(BOOKS_DIR, f))]
    
    print("\n" + "="*60)
    print(" SCIENTIFIC KNOWLEDGE BASE REPORT ".center(60, "="))
    print("="*60)
    
    print(f"\nSource documents: {len(book_files)} scientific educational books")
    
    # List the books
    if book_files:
        print("\nScientific literature in knowledge base:")
        for i, book in enumerate(book_files, 1):
            # Get file size in MB for reporting
            file_path = os.path.join(BOOKS_DIR, book)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            print(f"{i}. {book} ({file_size:.2f} MB)")
        
        print("\nKnowledge Extraction Process:")
        print("1. Books are processed using natural language processing")
        print("2. Text is divided into meaningful knowledge chunks")
        print("3. Chunks are categorized by educational topic")
        print("4. Knowledge is stored in a vector database for semantic retrieval")
        print("5. Scenarios and strategies are generated based on this knowledge")
    
    # List knowledge base components
    kb_components = []
    for item in os.listdir(KNOWLEDGE_DIR):
        item_path = os.path.join(KNOWLEDGE_DIR, item)
        if os.path.isdir(item_path) and item != "sample_documents":
            kb_components.append(f"{item} (directory)")
        elif os.path.isfile(item_path) and any(ext in item.lower() for ext in ['.db', '.index', '.bin', '.vec', '.npy', '.json']):
            kb_components.append(f"{item} (file)")
    
    if kb_components:
        print("\nKnowledge base technical components:")
        for component in kb_components:
            print(f"- {component}")
    
    print("\nThis system utilizes scientific literature to provide")
    print("evidence-based teaching scenarios and feedback.")
    print("="*60)

def get_huggingface_token():
    """
    Get the Hugging Face token from environment variable or user input.
    
    This function first checks if the token exists in the HF_TOKEN environment
    variable. If not found, it prompts the user to enter their token, with
    a secure password-style input that doesn't display the token on screen.
    
    Returns:
        str: The Hugging Face token, or None if not provided
    """
    # First try to get token from environment variable
    token = os.environ.get("HF_TOKEN")
    
    # If not in environment, prompt user (but make it optional)
    if not token:
        print("\n===== Hugging Face Authentication =====")
        print("To use gated models like Llama-3-8b, you need a Hugging Face token.")
        print("If you don't have one, you can create an account at https://huggingface.co/")
        print("and generate a token with read access.")
        print("You can skip this by pressing Enter to use open-access models instead.")
        
        token = getpass.getpass("Enter your Hugging Face token (hidden input): ")
        if token.strip() == "":
            print("No token provided. Will use open-access models only.")
            token = None
        else:
            print("Token received. Will use for accessing gated models.")
    else:
        print("Hugging Face token found in environment variables.")
    
    return token

def main():
    """Run the enhanced teacher training agent."""
    print_banner()
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install required dependencies and try again.")
        return
    
    # Ensure knowledge directory exists
    if not ensure_knowledge_directory():
        print("\nContinuing with integrated knowledge base from knowledge_base/default_strategies...")
    
    try:
        # Use existing components instead of importing the missing ai_agent module
        print("\nInitializing teacher training components...")
        
        # Initialize the vector database
        vector_db = VectorDatabase(db_path=VECTOR_DB_PATH)
        
        # Initialize the document processor
        doc_processor = DocumentProcessor()
        
        # Initialize the LLM handler
        llm_handler = DSPyLLMHandler(model_name="gpt-3.5-turbo")
        
        # Initialize the scenario generator
        scenario_generator = ClassroomScenarioGenerator(vector_db=vector_db, llm=llm_handler)
        
        # Initialize the response evaluator
        evaluator = ResponseEvaluator(vector_db=vector_db, llm_handler=llm_handler)
        
        # Initialize the student simulator
        student_simulator = StudentSimulator(vector_db=vector_db, llm_handler=llm_handler)
        
        # Check if knowledge base exists
        kb_exists = knowledge_base_exists()
        
        if kb_exists:
            print("\nExisting scientific knowledge base found.")
            generate_knowledge_base_report()
            print("\nUsing existing knowledge extracted from scientific literature.")
        else:
            print("\nNo existing scientific knowledge base found.")
            print("\nThe system requires scientific educational books for optimal performance.")
            print("To enhance the system with scientific knowledge, please add educational books to:")
            print(f"  {os.path.abspath(os.path.join(KNOWLEDGE_DIR, 'sample_documents'))}")
            print("\nTemporarily using basic strategies from knowledge_base/default_strategies")
            print("until scientific books are added.")
        
        # Create a simple interactive session
        print("\nStarting interactive teacher training session...")
        print("\nType 'exit' to quit the session.")
        
        # Generate a scenario
        scenario = scenario_generator.generate_scenario()
        print("\n" + "="*80)
        print("CLASSROOM SCENARIO".center(80))
        print("="*80)
        print(f"\n{scenario['scenario_content']}")
        
        while True:
            # Get teacher input
            teacher_input = input("\nYour response as a teacher: ")
            
            # Check if user wants to exit
            if teacher_input.lower() in ["exit", "quit", "bye"]:
                print("\nThank you for using the teacher training system!")
                break
            
            # Generate student response
            student_profile = scenario.get('student_profile', {})
            student_response = student_simulator.generate_student_response(
                teacher_input=teacher_input,
                student_profile=student_profile,
                scenario_context=scenario
            )
            print(f"\nStudent: {student_response}")
            
            # Evaluate teacher response
            evaluation = evaluator.evaluate_teacher_response(
                teacher_response=teacher_input,
                scenario=scenario,
                student_state=student_profile
            )
            
            print("\n" + "="*80)
            print("FEEDBACK".center(80))
            print("="*80)
            print(f"\nStrengths:")
            for strength in evaluation.get('strengths', []):
                print(f"- {strength}")
            
            print(f"\nAreas for improvement:")
            for area in evaluation.get('areas_for_improvement', []):
                print(f"- {area}")
            
            print(f"\nEffectiveness score: {evaluation.get('effectiveness_score', 0)}/10")
            print(f"\nRationale: {evaluation.get('rationale', '')}")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nError importing required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error running enhanced agent: {e}")
        print(f"\nAn error occurred: {e}")
        print("Please check teacher_training.log for details.")

if __name__ == "__main__":
    main() 