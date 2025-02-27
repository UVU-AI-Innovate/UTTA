#!/usr/bin/env python3
"""
Run Enhanced Teacher Training Agent

This script demonstrates the enhanced teacher training agent with knowledge base integration.
The agent processes educational books, stores knowledge in a vector database, and uses this
knowledge to generate realistic classroom management scenarios.

Usage:
    python run_enhanced.py

Required directory structure:
    ./knowledge_base/books/ - Directory containing educational textbooks/PDFs

Example:
    1. Place educational books in PDF, DOCX, or TXT format in ./knowledge_base/books/
    2. Run this script
    3. Follow the interactive prompts to experience the enhanced training
"""

import os
import logging
import sys
import importlib
import subprocess
import getpass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
    print("1. Processes scientific educational books from ./knowledge_base/books/")
    print("2. Extracts and categorizes research-based knowledge")
    print("3. Generates realistic teaching scenarios based on educational research")
    print("4. Provides feedback and strategies supported by educational literature")
    print("="*80)

def ensure_knowledge_directory():
    """Ensure the knowledge base directory exists and contains scientific educational books."""
    books_dir = os.path.join("knowledge_base", "books")
    os.makedirs(books_dir, exist_ok=True)
    
    # Check for default strategy files
    strategies_dir = os.path.join("knowledge_base", "default_strategies")
    has_strategy_files = os.path.exists(strategies_dir) and len(os.listdir(strategies_dir)) > 0
    
    if not os.path.exists(books_dir) or not os.listdir(books_dir):
        if has_strategy_files:
            print("\n⚠️ No scientific educational books found in the knowledge base directory.")
            print("For optimal performance and evidence-based training, please add educational books.")
            print("\nRecommended book types:")
            print("- Elementary education textbooks")
            print("- Classroom management research")
            print("- Child development and psychology literature")
            print("- Teaching methodologies and pedagogical books")
            print(f"\nAdd these books to: {os.path.abspath(books_dir)}")
            print(f"\nTemporarily using built-in strategies, but these are not as effective as")
            print(f"using scientific literature. The system works best with actual books.")
            return True
        else:
            print("\n⚠️ No scientific educational books or strategy files found.")
            print("\nFor this system to work effectively, please add educational books such as:")
            print("- Elementary education textbooks")
            print("- Classroom management research")
            print("- Child development and psychology literature")
            print("- Teaching methodologies and pedagogical books")
            print(f"\nAdd these books (PDF, DOCX, TXT) to: {os.path.abspath(books_dir)}")
            print("Then run this script again.")
            return False
    
    print(f"\n✓ Found scientific educational books in {os.path.abspath(books_dir)}")
    print("The system will use these books to provide evidence-based training.")
    return True

def knowledge_base_exists():
    """Check if a knowledge base already exists."""
    # Common locations for vector database storage
    kb_dir = os.path.join("knowledge_base")
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
    books_dir = os.path.join("knowledge_base", "books")
    kb_dir = os.path.join("knowledge_base")
    
    # Count source documents
    book_files = [f for f in os.listdir(books_dir) if os.path.isfile(os.path.join(books_dir, f))]
    
    print("\n" + "="*60)
    print(" SCIENTIFIC KNOWLEDGE BASE REPORT ".center(60, "="))
    print("="*60)
    
    print(f"\nSource documents: {len(book_files)} scientific educational books")
    
    # List the books
    if book_files:
        print("\nScientific literature in knowledge base:")
        for i, book in enumerate(book_files, 1):
            # Get file size in MB for reporting
            file_path = os.path.join(books_dir, book)
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
    for item in os.listdir(kb_dir):
        item_path = os.path.join(kb_dir, item)
        if os.path.isdir(item_path) and item != "books":
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
        # Import the enhanced agent
        from ai_agent import EnhancedTeacherTrainingAgent
        from llm_handler import EnhancedLLMInterface, PedagogicalLanguageProcessor
        
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
            print(f"  {os.path.abspath(os.path.join('knowledge_base', 'books'))}")
            print("\nTemporarily using basic strategies from knowledge_base/default_strategies")
            print("until scientific books are added.")
        
        # Create and run the enhanced agent
        print("\nInitializing scientific book-based teacher training agent...")
        
        # Try to initialize with use_existing parameter if knowledge base exists
        try:
            if kb_exists:
                agent = EnhancedTeacherTrainingAgent(use_existing=True)
            else:
                agent = EnhancedTeacherTrainingAgent()
                
            # Ensure LLM system is ready and display comprehensive information
            print("\nInitializing LLM system...")
            
            # Get Hugging Face token securely
            hf_token = get_huggingface_token()
            
            # Choose appropriate model based on token availability
            if hf_token:
                model_name = 'meta-llama/Llama-3-8b-hf'  # Gated model requiring authentication
                print(f"Using {model_name} with authentication")
            else:
                model_name = 'microsoft/phi-2'  # Open-access model
                print(f"Using {model_name} (open-access model)")
            
            processor = PedagogicalLanguageProcessor(
                model_name=model_name,
                quantization="8-bit",  # 8-bit quantization for good balance of quality and efficiency
                token=hf_token         # Token might be None, which is handled by the processor
            )
            
            processor.ensure_server_running()  # This now displays detailed LLM info
            
            # Start interactive session
            agent.start_enhanced_interactive_session()
            
        except TypeError:
            # If that doesn't work, try without the parameter
            agent = EnhancedTeacherTrainingAgent()
            
            # Ensure LLM system is ready and display comprehensive information
            print("\nInitializing LLM system...")
            
            # Get Hugging Face token securely
            hf_token = get_huggingface_token()
            
            # Choose appropriate model based on token availability
            if hf_token:
                model_name = 'meta-llama/Llama-3-8b-hf'  # Gated model requiring authentication
                print(f"Using {model_name} with authentication")
            else:
                model_name = 'microsoft/phi-2'  # Open-access model
                print(f"Using {model_name} (open-access model)")
            
            processor = PedagogicalLanguageProcessor(
                model_name=model_name,
                quantization="8-bit",  # 8-bit quantization for good balance of quality and efficiency
                token=hf_token         # Token might be None, which is handled by the processor
            )
            
            processor.ensure_server_running()  # This now displays detailed LLM info
            
            # Start interactive session
            agent.start_enhanced_interactive_session()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"\nError importing required modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        logger.error(f"Error running enhanced agent: {e}")
        print(f"\nAn error occurred: {e}")
        print("Please check knowledge_processing.log for details.")

if __name__ == "__main__":
    main() 