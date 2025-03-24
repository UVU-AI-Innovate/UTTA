#!/usr/bin/env python
# Fix DSPy and OpenAI compatibility issues

import os
import sys
import subprocess
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed."""
    try:
        importlib.util.find_spec(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name, version=None):
    """Install a package with pip."""
    cmd = [sys.executable, "-m", "pip", "install"]
    if version:
        cmd.append(f"{package_name}=={version}")
    else:
        cmd.append(package_name)
    
    print(f"Installing {package_name}{'==' + version if version else ''}...")
    subprocess.check_call(cmd)

def main():
    print("Checking and fixing OpenAI and DSPy dependencies...")
    
    # Fix OpenAI version if needed
    if check_package_installed("openai"):
        import openai
        print(f"Current OpenAI version: {openai.__version__}")
        if openai.__version__ != "0.28.0":
            print("Reinstalling OpenAI to version 0.28.0 (required for DSPy)...")
            install_package("openai", "0.28.0")
    else:
        install_package("openai", "0.28.0")
    
    # Install or fix DSPy
    if check_package_installed("dspy"):
        print("Reinstalling DSPy to ensure compatibility...")
        install_package("dspy-ai", "2.0.4")
    else:
        install_package("dspy-ai", "2.0.4")
    
    # Fix sentence-transformers
    if not check_package_installed("sentence_transformers"):
        install_package("sentence-transformers", "2.2.2")
    
    print("Dependency fixes completed.")
    print("Please restart the application to apply changes.")

if __name__ == "__main__":
    main() 