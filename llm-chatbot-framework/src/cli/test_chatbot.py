#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.llm.dspy.handler import PedagogicalLanguageProcessor
from src.llm.dspy.adapter import create_llm_interface

def test_basic_conversation():
    """Test basic conversation capabilities of the chatbot."""
    print("\nTesting Basic Conversation Capabilities...")
    
    # Initialize the LLM interface
    llm = create_llm_interface()
    processor = PedagogicalLanguageProcessor(llm)
    
    # Test cases
    test_inputs = [
        "Can you explain what photosynthesis is?",
        "How would you explain addition to a 6-year-old?",
        "What's the best way to teach fractions?"
    ]
    
    for test_input in test_inputs:
        print(f"\nTeacher: {test_input}")
        response = processor.get_teaching_response(test_input)
        print(f"Chatbot: {response}")
        print("-" * 80)

def test_pedagogical_features():
    """Test pedagogical features of the chatbot."""
    print("\nTesting Pedagogical Features...")
    
    llm = create_llm_interface()
    processor = PedagogicalLanguageProcessor(llm)
    
    # Test scaffolding and engagement
    test_case = {
        "student_level": "beginner",
        "topic": "multiplication",
        "context": "The student is struggling with basic multiplication concepts"
    }
    
    print("\nTesting adaptive teaching response...")
    print(f"Context: {test_case['context']}")
    response = processor.get_adaptive_teaching_response(
        test_case["topic"],
        student_level=test_case["student_level"],
        context=test_case["context"]
    )
    print(f"Chatbot: {response}")
    print("-" * 80)

def main():
    print("Starting Teacher Training Chatbot Tests...")
    
    try:
        test_basic_conversation()
        test_pedagogical_features()
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 