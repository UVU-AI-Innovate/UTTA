#!/usr/bin/env python3
"""
UTTA Test Runner

This script runs all tests for the UTTA framework.
"""

import os
import sys
import unittest

if __name__ == "__main__":
    print("Running UTTA Tests...")
    print("=====================")
    
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Run the standalone test script as well
    print("\nRunning standalone test script...")
    os.system('python tests/test_chatbot.py')
    
    print("\nTest run complete.") 