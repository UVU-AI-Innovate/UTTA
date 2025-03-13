#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Setup script for the LLM Fine-Tuning project.

This script helps users get started with the project by:
1. Checking if all required files are present
2. Installing dependencies
3. Setting up environment variables
4. Running a simple test to ensure everything is working
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Setup the LLM Fine-Tuning project")
    parser.add_argument(
        "--install-deps", 
        action="store_true",
        help="Install dependencies from requirements.txt"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true",
        help="Only check the project structure without installing dependencies"
    )
    parser.add_argument(
        "--openai-key", 
        type=str,
        help="OpenAI API key to use for examples"
    )
    return parser.parse_args()

def check_structure():
    """Check if the project structure is correct."""
    print("Checking project structure...")
    
    # Run the check_structure.py script
    try:
        result = subprocess.run(
            [sys.executable, "check_structure.py"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("Error checking project structure:")
        print(e.stdout)
        print(e.stderr)
        return False

def install_dependencies():
    """Install dependencies from requirements.txt."""
    print("Installing dependencies...")
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print("Error installing dependencies:")
        print(e)
        return False

def setup_environment(openai_key=None):
    """Set up environment variables."""
    print("Setting up environment variables...")
    
    # Check if OpenAI API key is provided or already set
    current_key = os.environ.get("OPENAI_API_KEY")
    
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
        print("OpenAI API key set from command line argument.")
    elif current_key:
        print("OpenAI API key already set in environment.")
    else:
        print("Warning: OpenAI API key not set. Some examples may not work.")
        print("You can set it with:")
        print("  export OPENAI_API_KEY=your-api-key-here")
    
    return True

def run_simple_test():
    """Run a simple test to ensure everything is working."""
    print("Running a simple test...")
    
    # Check if we can import the main modules
    try:
        # Try importing some key modules
        import numpy
        import pandas
        import tqdm
        import jsonlines
        
        print("Core dependencies imported successfully!")
        
        # Try importing project modules
        sys.path.append(str(Path(__file__).parent))
        from src.utils import load_jsonl, save_jsonl
        
        print("Project modules imported successfully!")
        
        # Check if the sample dataset exists
        data_path = Path(__file__).parent / "examples" / "data" / "educational_qa_sample.jsonl"
        if data_path.exists():
            data = load_jsonl(data_path)
            print(f"Sample dataset loaded successfully with {len(data)} examples!")
        else:
            print(f"Warning: Sample dataset not found at {data_path}")
        
        return True
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please make sure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    print("="*50)
    print("LLM Fine-Tuning Project Setup")
    print("="*50)
    
    # Check project structure
    if not check_structure():
        print("Project structure check failed. Please fix the issues before continuing.")
        return 1
    
    # If check-only flag is set, exit after checking structure
    if args.check_only:
        print("Check-only flag set. Exiting.")
        return 0
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            print("Failed to install dependencies. Please try installing them manually.")
            return 1
    
    # Set up environment variables
    if not setup_environment(args.openai_key):
        print("Failed to set up environment variables.")
        return 1
    
    # Run a simple test
    if not run_simple_test():
        print("Simple test failed. Please check the error messages.")
        return 1
    
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("="*50)
    print("\nYou can now run the examples:")
    print("  python examples/dspy/optimize_educational_qa.py")
    print("  python examples/openai/finetune_educational_qa.py")
    print("  python examples/huggingface/finetune_educational_qa.py")
    print("\nOr compare all approaches:")
    print("  python compare_approaches.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 