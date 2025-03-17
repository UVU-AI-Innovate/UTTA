#!/usr/bin/env python3
"""
Dependency Check Script for the UTTA System.
This script verifies if all required dependencies are installed and working correctly.
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_dependency(module_name, min_version=None, alt_module=None):
    """Check if a Python module is installed and meets the minimum version."""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, '__version__') and min_version:
            version = module.__version__
            if version < min_version:
                return False, f"{module_name} version {version} is less than required version {min_version}"
        return True, f"{module_name} is installed"
    except ImportError:
        if alt_module:
            try:
                importlib.import_module(alt_module)
                return True, f"{module_name} is not installed, but alternative {alt_module} is available"
            except ImportError:
                return False, f"Neither {module_name} nor alternative {alt_module} is installed"
        return False, f"{module_name} is not installed"

def check_environment():
    """Check the Python environment."""
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print("")

def check_dependencies():
    """Check if all required dependencies are installed."""
    dependencies = [
        ("openai", "0.28.0"),
        ("dspy", None, "dspy_ai"),
        ("torch", None),
        ("sentence_transformers", None),
        ("streamlit", None),
        ("spacy", None),
        ("textstat", None),
        ("numpy", None),
        ("pandas", None),
        ("python-dotenv", None, "dotenv"),
    ]
    
    print("Checking dependencies:")
    all_passed = True
    for dep in dependencies:
        module_name = dep[0]
        min_version = dep[1] if len(dep) > 1 else None
        alt_module = dep[2] if len(dep) > 2 else None
        
        passed, message = check_dependency(module_name, min_version, alt_module)
        status = "✅" if passed else "❌"
        print(f"  {status} {message}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_openai_dependency():
    """Check specifically for the OpenAI dependency which is critical for DSPy."""
    print("\nChecking OpenAI dependency:")
    try:
        import openai
        print(f"  ✅ OpenAI is installed (version: {openai.__version__})")
        
        # Check if the openai.error module exists (important for DSPy)
        try:
            from openai import error
            print("  ✅ openai.error module is available")
        except ImportError:
            print("  ❌ openai.error module is missing - This may cause issues with DSPy")
            print("     Run 'python fix_openai_dep.py' to fix this issue")
            return False
            
        return True
    except ImportError:
        print("  ❌ OpenAI is not installed")
        return False

def check_spacy_models():
    """Check if required spaCy models are installed."""
    print("\nChecking spaCy models:")
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            print("  ✅ en_core_web_sm model is installed")
            return True
        except IOError:
            print("  ❌ en_core_web_sm model is not installed")
            print("     Install it with: python -m spacy download en_core_web_sm")
            return False
    except ImportError:
        print("  ❌ spaCy is not installed")
        return False

def main():
    """Main function to run all checks."""
    print("=== UTTA Dependency Check ===\n")
    
    check_environment()
    deps_ok = check_dependencies()
    openai_ok = check_openai_dependency()
    spacy_ok = check_spacy_models()
    
    print("\n=== Summary ===")
    print(f"  Dependencies: {'✅ OK' if deps_ok else '❌ Issues found'}")
    print(f"  OpenAI: {'✅ OK' if openai_ok else '❌ Issues found'}")
    print(f"  SpaCy Models: {'✅ OK' if spacy_ok else '❌ Issues found'}")
    
    if not (deps_ok and openai_ok and spacy_ok):
        print("\nRecommended action:")
        print("  1. Run 'python fix_openai_dep.py' to fix OpenAI dependency issues")
        print("  2. Run 'python -m spacy download en_core_web_sm' to install missing spaCy model")
        print("  3. Run 'pip install -r requirements.txt' to install missing dependencies")
        return 1
    else:
        print("\n✅ All dependencies are properly installed.")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 