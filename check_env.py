#!/usr/bin/env python

import os
import sys
from dotenv import load_dotenv

def main():
    """Check if the OpenAI API key is properly loaded from the environment."""
    print("Checking environment variables...")
    
    # Try loading from .env file
    load_dotenv(override=True)
    
    # Check if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ OpenAI API key found in environment: {api_key[:5]}...{api_key[-4:]}")
    else:
        print("❌ OpenAI API key not found in environment")
        print("Make sure your .env file contains a valid OPENAI_API_KEY")
    
    print("\nChecking other required environment variables:")
    env_vars = [
        "DEFAULT_MODEL", "TEMPERATURE", "MAX_TOKENS",
        "CACHE_DIR", "INDEX_DIR", "MODEL_DIR",
        "LOG_LEVEL", "LOG_FILE",
        "STREAMLIT_PORT", "STREAMLIT_THEME"
    ]
    
    all_found = True
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var} not found")
            all_found = False
    
    if all_found:
        print("\nAll required environment variables are set.")
    else:
        print("\nSome environment variables are missing. Check your .env file.")

if __name__ == "__main__":
    main() 