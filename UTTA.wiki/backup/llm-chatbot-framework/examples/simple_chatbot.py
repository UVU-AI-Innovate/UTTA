#!/usr/bin/env python3
"""
Simple Chatbot Example

This example demonstrates how to create a basic chatbot using the LLM Chatbot Framework.
It shows how to initialize the core components and create a simple interactive chat loop.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core components
from src.core.vector_database import VectorDatabase
from src.llm.dspy.handler import DSPyLLMHandler
from src.core.document_processor import DocumentProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def main():
    """Run the simple chatbot example."""
    print("Initializing Simple Chatbot...")
    
    # Initialize components
    vector_db = VectorDatabase()
    llm_handler = DSPyLLMHandler(model_name="gpt-3.5-turbo")
    document_processor = DocumentProcessor()
    
    # Add some sample knowledge to the vector database
    print("Adding sample knowledge to the database...")
    vector_db.add_chunks([
        {
            "text": "Effective classroom management involves clear expectations and consistent routines.",
            "metadata": {"topic": "classroom_management"}
        },
        {
            "text": "Differentiated instruction adapts teaching methods to meet diverse student needs.",
            "metadata": {"topic": "instruction"}
        },
        {
            "text": "Formative assessment provides ongoing feedback to improve learning.",
            "metadata": {"topic": "assessment"}
        }
    ])
    
    # Simple chat loop
    print("\nSimple Chatbot is ready! Type 'exit' to quit.")
    print("You can ask questions about classroom management, instruction, or assessment.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        # Search for relevant knowledge
        search_results = vector_db.search(user_input, top_k=2)
        
        # Extract relevant context
        context = ""
        if search_results:
            context = "Based on the following information:\n\n"
            for i, result in enumerate(search_results):
                context += f"{i+1}. {result['text']}\n"
        
        # Generate response using LLM
        prompt = f"""
        You are an educational assistant helping teachers.
        
        {context}
        
        Please answer the following question in a helpful, concise way:
        {user_input}
        """
        
        response = llm_handler.generate(prompt)
        
        # Print response
        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main() 