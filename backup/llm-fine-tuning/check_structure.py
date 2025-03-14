#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to check the project structure and ensure all necessary files are in place.
"""

import os
import sys
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama
init()

def print_status(path, exists):
    """Print the status of a file or directory."""
    status = f"{Fore.GREEN}✓{Style.RESET_ALL}" if exists else f"{Fore.RED}✗{Style.RESET_ALL}"
    print(f"{status} {path}")

def check_structure():
    """Check the project structure."""
    root_dir = Path(__file__).parent
    
    # Define expected structure
    expected_structure = {
        "src": {
            "dspy": {
                "__init__.py": True,
                "models.py": True,
                "optimizer.py": True,
            },
            "openai": {
                "__init__.py": True,
                "finetuner.py": True,
            },
            "huggingface": {
                "__init__.py": True,
                "finetuner.py": True,
            },
            "__init__.py": True,
            "utils.py": True,
        },
        "examples": {
            "data": {
                "educational_qa_sample.jsonl": True,
            },
            "dspy": {
                "optimize_educational_qa.py": True,
                "output": {
                    "": True,  # Just check if directory exists
                },
            },
            "openai": {
                "finetune_educational_qa.py": True,
                "output": {
                    "": True,  # Just check if directory exists
                },
            },
            "huggingface": {
                "finetune_educational_qa.py": True,
                "output": {
                    "": True,  # Just check if directory exists
                },
            },
        },
        "README.md": True,
        "requirements.txt": True,
        "compare_approaches.py": True,
    }
    
    missing_files = []
    
    # Check structure
    print(f"{Fore.CYAN}Checking project structure...{Style.RESET_ALL}")
    
    def check_recursive(structure, current_path=root_dir):
        for name, value in structure.items():
            path = current_path / name
            
            if isinstance(value, dict):
                # It's a directory
                exists = path.is_dir()
                print_status(path.relative_to(root_dir), exists)
                
                if exists:
                    check_recursive(value, path)
                else:
                    missing_files.append(path.relative_to(root_dir))
            else:
                # It's a file
                if name == "":  # Special case for empty directories
                    continue
                    
                exists = path.is_file()
                print_status(path.relative_to(root_dir), exists)
                
                if not exists:
                    missing_files.append(path.relative_to(root_dir))
    
    check_recursive(expected_structure)
    
    # Print summary
    print("\n" + "="*50)
    if missing_files:
        print(f"{Fore.RED}Missing {len(missing_files)} files/directories:{Style.RESET_ALL}")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease create these files to complete the project structure.")
        return False
    else:
        print(f"{Fore.GREEN}All files and directories are in place!{Style.RESET_ALL}")
        return True

def main():
    """Main function."""
    success = check_structure()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 