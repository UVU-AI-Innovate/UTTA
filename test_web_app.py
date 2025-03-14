"""
Test script for the web application.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to Python path
sys.path.append(str(Path(__file__).parent))

def test_web_app():
    """Test if the web application can be initialized."""
    try:
        from src.web.app import WebInterface
        
        # Initialize web interface
        interface = WebInterface()
        logger.info("✓ Web interface initialized successfully")
        
        # Test basic functionality
        assert hasattr(interface, 'llm_interface'), "LLM interface not initialized"
        assert hasattr(interface, 'setup_page'), "Setup page method not found"
        
        logger.info("✓ Web interface basic functionality verified")
        return True
    except Exception as e:
        logger.error(f"Web app test failed: {str(e)}")
        return False

def main():
    """Run web app tests."""
    logger.info("Starting web app tests...")
    
    if test_web_app():
        logger.info("✓ All web app tests passed!")
    else:
        logger.error("❌ Web app tests failed")

if __name__ == "__main__":
    main() 