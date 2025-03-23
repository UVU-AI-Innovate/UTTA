#!/usr/bin/env python
"""
Utility functions for the UTTA examples.
Contains helper functions for environment setup, API key loading, etc.
"""
import os
import sys
from pathlib import Path

def load_api_key():
    """
    Load the OpenAI API key from the .env file or environment variables.
    Returns:
        bool: True if API key was successfully loaded, False otherwise
    """
    # First check if it's already in environment
    if os.getenv("OPENAI_API_KEY"):
        print("API key already loaded in environment.")
        return True
        
    # Try loading from dotenv package
    try:
        from dotenv import load_dotenv
        
        # Try parent directory first (examples/ -> UTTA/)
        root_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(root_env_path):
            load_dotenv(root_env_path)
            print(f"Loaded environment from {root_env_path}")
            if os.getenv("OPENAI_API_KEY"):
                masked_key = os.getenv("OPENAI_API_KEY")[:8] + "..." + os.getenv("OPENAI_API_KEY")[-4:]
                print(f"Loaded API key: {masked_key}")
                return True
        
        # Try current directory
        local_env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(local_env_path):
            load_dotenv(local_env_path)
            print(f"Loaded environment from {local_env_path}")
            if os.getenv("OPENAI_API_KEY"):
                return True
    except ImportError:
        print("Warning: python-dotenv package not installed. Trying direct file reading.")
    
    # If dotenv failed or not available, try direct file reading
    try:
        env_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'),  # root dir
            os.path.join(os.path.dirname(__file__), '.env'),                   # examples dir
        ]
        
        for env_path in env_paths:
            if os.path.exists(env_path):
                print(f"Reading .env file directly from {env_path}")
                with open(env_path, 'r') as env_file:
                    for line in env_file:
                        if line.strip().startswith('OPENAI_API_KEY='):
                            api_key = line.strip().split('=', 1)[1]
                            # Remove quotes if present
                            if (api_key.startswith('"') and api_key.endswith('"')) or \
                               (api_key.startswith("'") and api_key.endswith("'")):
                                api_key = api_key[1:-1]
                            os.environ["OPENAI_API_KEY"] = api_key
                            masked_key = api_key[:8] + "..." + api_key[-4:]
                            print(f"Directly loaded API key: {masked_key}")
                            return True
    except Exception as e:
        print(f"Error reading .env file: {e}")
    
    # If we got here, no API key was found
    print("Warning: OPENAI_API_KEY not found in environment or .env files.")
    return False

def check_dspy_installation():
    """
    Check if DSPy is properly installed and can be used.
    Returns:
        bool: True if DSPy is installed and can be configured, False otherwise
    """
    try:
        import dspy
        # Try to create a simple signature to verify dspy functionality
        class SimpleTest(dspy.Signature):
            input_text = dspy.InputField()
            output_text = dspy.OutputField()
        
        # Try to create a module
        test_module = dspy.ChainOfThought(SimpleTest)
        print("DSPy successfully imported and functional")
        return True
    except ImportError:
        print("Warning: DSPy not installed. Run 'pip install dspy-ai==2.0.4' to install.")
        return False
    except Exception as e:
        print(f"Warning: DSPy installed but not functioning correctly: {e}")
        return False

def check_cuda_availability():
    """
    Check if CUDA is available for PyTorch.
    Returns:
        bool: True if CUDA is available, False otherwise
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device_name = torch.cuda.get_device_name(0)
            device_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"CUDA available: {device_name} ({device_mem:.1f} GB)")
        else:
            print("CUDA not available. Using CPU.")
        return has_cuda
    except ImportError:
        print("Warning: PyTorch not installed. Run 'pip install torch' to install.")
        return False 