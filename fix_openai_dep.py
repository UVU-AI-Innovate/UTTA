#!/usr/bin/env python3
"""
Fix script for OpenAI dependency issues with DSPy.
This script will install the correct version of OpenAI that's compatible with DSPy 2.0.4.
"""

import sys
import subprocess
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_openai_compatibility():
    """Check if OpenAI package is compatible with DSPy."""
    try:
        # Try to import the specific openai.error module needed by DSPy
        import openai.error
        logger.info("✅ OpenAI package is compatible with DSPy")
        return True
    except ImportError:
        logger.error("❌ OpenAI package is incompatible with DSPy (missing openai.error module)")
        return False

def fix_openai_installation():
    """Install the correct version of OpenAI that works with DSPy 2.0.4."""
    logger.info("Installing OpenAI v0.28.0 (compatible with DSPy 2.0.4)...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "openai==0.28.0"])
        logger.info("✅ Successfully installed OpenAI v0.28.0")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install OpenAI: {e}")
        return False

def check_dspy_installation():
    """Check if DSPy is installed."""
    try:
        import dspy
        logger.info(f"✅ DSPy found (version: {getattr(dspy, '__version__', 'unknown')})")
        return True
    except ImportError:
        logger.warning("❌ DSPy not found, installing...")
        return False

def fix_dspy_installation():
    """Install DSPy."""
    logger.info("Installing DSPy 2.0.4...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "dspy-ai==2.0.4"])
        logger.info("✅ Successfully installed DSPy 2.0.4")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install DSPy: {e}")
        return False

def main():
    """Run the fix script."""
    print("=" * 60)
    print("DSPy OpenAI Dependency Fix Script")
    print("=" * 60)
    
    # Check if OpenAI is already compatible
    if check_openai_compatibility():
        print("\nYour OpenAI package is already compatible with DSPy.")
        return
    
    # Fix OpenAI installation
    if fix_openai_installation():
        print("\nOpenAI dependency fixed successfully.")
    else:
        print("\nFailed to fix OpenAI dependency. Try manually running:")
        print("pip install openai==0.28.0")
    
    # Check DSPy installation
    if not check_dspy_installation():
        fix_dspy_installation()
    
    # Verify fix worked
    if check_openai_compatibility():
        print("\nFix succeeded! The application should now work correctly.")
        print("Run the web application with: streamlit run src/web_app.py")
    else:
        print("\nFix did not resolve the issue. Try restarting your environment.")

if __name__ == "__main__":
    main() 