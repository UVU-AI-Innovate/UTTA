#!/usr/bin/env python
"""
Fine-Tuning Example

This example demonstrates how to use the fine-tuning capabilities
of the framework to improve model performance on specific tasks.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any
import time

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from src.llm.fine_tuning.dspy_optimizer import DSPyOptimizer
from src.llm.fine_tuning.openai_tuner import OpenAIFineTuner
from src.llm.dspy.handler import DSPyLLMHandler
from src.core.vector_database import VectorDatabase

# Define the required DSPy module for optimization
import dspy
from dspy.signatures import Signature

# Define a simple DSPy module for our example
class TeacherResponder(dspy.Module):
    """Simple module for educational QA."""
    def __init__(self):
        super().__init__()
        self.generate = dspy.Predict("question, context -> answer")
    
    def forward(self, question, context):
        return self.generate(question=question, context=context)

# Create a mock DSPyLLMHandler for testing without API keys
class MockDSPyLLMHandler:
    """Mock handler for testing without API keys"""
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.model_name = model_name
        self.interface = type("Interface", (), {})()
        self.interface.teacher_responder = TeacherResponder()
        print(f"Initialized mock handler with model: {model_name}")
    
    def get_llm_response(self, messages, output_format=None):
        """Mock response generator"""
        # Extract the query from the messages
        query = messages[0]["content"] if messages else ""
        
        # Simple rule-based responses for testing
        if "classroom management" in query.lower():
            return "Classroom management refers to the skills and techniques teachers use to keep students organized, focused, and academically productive during class. It involves creating a positive learning environment, establishing clear expectations, and managing student behavior effectively."
        elif "formative assessment" in query.lower():
            return "Formative assessment refers to methods that teachers use to evaluate student comprehension, learning needs, and academic progress during a lesson or course. It provides feedback to both students and teachers to improve the learning process."
        elif "differentiated instruction" in query.lower():
            return "Differentiated instruction is the practice of using multiple teaching approaches to address varied learning needs of students. It allows teachers to tailor their teaching methods to reach all students effectively."
        else:
            return "This is a mock response for testing purposes. In a real implementation, this would be generated by the LLM."

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fine_tuning_example.log"),
        logging.StreamHandler()
    ]
)

def demonstrate_dspy_optimization():
    """
    Demonstrate DSPy's optimization capabilities (soft fine-tuning).
    
    This example shows how to:
    1. Create training examples
    2. Optimize a DSPy module
    3. Save and load optimized modules
    4. Use the optimized module for better performance
    """
    print("\n=== DSPy Optimization Example (Mock Mode) ===\n")
    print("Note: This is running in mock mode without actual LLM API calls")
    
    # Create a mock DSPy LLM handler
    handler = MockDSPyLLMHandler(model_name="gpt-3.5-turbo")
    
    # Initialize the optimizer
    optimizer = DSPyOptimizer(llm_handler=handler)
    
    # Create some educational context for testing
    contexts = [
        "Classroom management refers to the wide variety of skills and techniques that teachers use to keep students organized, orderly, focused, attentive, on task, and academically productive during a class.",
        "Formative assessment refers to a wide variety of methods that teachers use to conduct in-process evaluations of student comprehension, learning needs, and academic progress during a lesson, unit, or course.",
        "Differentiated instruction is the practice of developing and using multiple teaching approaches to present learning content in a way that addresses the varied learning needs of students."
    ]
    
    # Add the contexts to a vector database
    print("Setting up vector database with sample educational content...")
    db = VectorDatabase()
    db.add_chunks([
        {"text": context, "metadata": {"source": "education_basics"}}
        for context in contexts
    ])
    
    # Create training examples for the optimization
    # Each example should have an input and an expected output
    print("Creating training examples for optimization...")
    training_examples = [
        {
            "input": {
                "question": "What is classroom management?",
                "context": contexts[0]
            },
            "output": {
                "answer": "Classroom management refers to the skills and techniques teachers use to keep students organized, focused, and academically productive during class."
            }
        },
        {
            "input": {
                "question": "What is formative assessment?",
                "context": contexts[1]
            },
            "output": {
                "answer": "Formative assessment refers to methods that teachers use to evaluate student comprehension, learning needs, and academic progress during a lesson or course."
            }
        },
        {
            "input": {
                "question": "What is differentiated instruction?",
                "context": contexts[2]
            },
            "output": {
                "answer": "Differentiated instruction is the practice of using multiple teaching approaches to address varied learning needs of students."
            }
        }
    ]
    
    # Test the unoptimized module performance
    print("\nUnoptimized response:")
    unoptimized_result = handler.get_llm_response(
        [{"role": "user", "content": f"Question: What is classroom management?\nContext: {contexts[0]}\nAnswer:"}]
    )
    print(unoptimized_result)
    
    # Mock optimization process
    print("\nMock optimizing the teacher_responder module...")
    print("(In a real scenario, this would call the DSPy optimization methods)")
    
    # Simulate optimization by modifying the mock handler
    import time
    time.sleep(2)  # Simulate processing time
    
    # Create mock optimized response
    optimized_result = "Classroom management refers to the skills and techniques teachers use to keep students organized, focused, and academically productive during class. It's a fundamental aspect of effective teaching that creates an environment conducive to learning."
    
    # Show comparison
    print("\nComparison:")
    print(f"Unoptimized: {unoptimized_result[:100]}...")
    print(f"Optimized (simulated): {optimized_result[:100]}...")
    
    print("\nIn a real scenario with an API key set, the optimize_module method would:")
    print("1. Find optimal few-shot examples")
    print("2. Tune the prompt structure")
    print("3. Save the optimized module for reuse")

def mock_openai_fine_tuning(use_mock=True):
    """
    Demonstrate OpenAI fine-tuning process.
    
    When use_mock=True, no actual API calls are made.
    """
    print("\n=== OpenAI Fine-Tuning Example ===\n")
    print("Note: This is a demonstration of the workflow. No actual API calls are made.")
    
    # Create a set of example conversations for fine-tuning
    print("Creating example conversations for fine-tuning...")
    conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are an educational assistant focused on teaching strategies."},
                {"role": "user", "content": "What are some strategies for managing a disruptive classroom?"},
                {"role": "assistant", "content": "For managing a disruptive classroom, consider these strategies:\n1. Establish clear expectations and rules on day one\n2. Be consistent with consequences\n3. Use positive reinforcement for good behavior\n4. Implement a quiet signal to get attention\n5. Address disruptive behavior privately when possible\n6. Create engaging lessons to minimize off-task behavior\n7. Use proximity control by moving around the classroom\n8. Build positive relationships with students"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an educational assistant focused on teaching strategies."},
                {"role": "user", "content": "How can I differentiate instruction for varied ability levels?"},
                {"role": "assistant", "content": "To differentiate instruction for varied ability levels:\n1. Pre-assess students to identify knowledge gaps and strengths\n2. Provide tiered assignments with varying complexity\n3. Use flexible grouping based on skills or interests\n4. Offer choice boards that allow students to select activities\n5. Implement learning stations with different tasks\n6. Adjust questioning techniques to match cognitive levels\n7. Vary instruction formats (visual, auditory, kinesthetic)\n8. Allow multiple ways to demonstrate understanding"}
            ]
        }
    ]
    
    # Save the conversations to a JSONL file
    os.makedirs("cache/fine_tuning", exist_ok=True)
    training_file = "cache/fine_tuning/education_examples.jsonl"
    
    with open(training_file, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")
    
    print(f"Saved training data to {training_file}")
    
    if use_mock:
        # Mock the fine-tuning process for demonstration purposes
        print("\nMock fine-tuning process (no actual API calls):")
        
        # Simulate starting a fine-tuning job
        print("1. Starting fine-tuning job...")
        job_id = f"ft-job-{int(time.time())}"
        print(f"   Fine-tuning job started with ID: {job_id}")
        
        # Simulate checking status
        print("2. Checking job status...")
        for status in ["validating_files", "queued", "running", "succeeded"]:
            print(f"   Job status: {status}")
            time.sleep(1)
        
        # Simulate job completion
        model_id = f"ft:gpt-3.5-turbo:{int(time.time())}"
        print(f"3. Fine-tuning complete. Model ID: {model_id}")
        
        # Simulate using the fine-tuned model
        print("4. Using the fine-tuned model:")
        prompt = "How can I engage reluctant learners?"
        response = "To engage reluctant learners:\n1. Connect content to their interests and real-world applications\n2. Build a supportive relationship and show genuine interest\n3. Provide choices and autonomy in learning activities\n4. Use interactive and hands-on learning approaches\n5. Break tasks into manageable chunks with clear success criteria\n6. Celebrate small wins and progress\n7. Implement cooperative learning opportunities\n8. Use technology and multimedia tools strategically"
        
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response[:100]}...")
        
    else:
        # Actual fine-tuning process (requires OpenAI API key)
        try:
            # Initialize the OpenAI fine-tuner
            tuner = OpenAIFineTuner()
            
            # Start fine-tuning job
            print("\nStarting actual fine-tuning job...")
            job_id = tuner.start_fine_tuning(
                training_file=training_file,
                model="gpt-3.5-turbo",
                suffix="edu-assistant"
            )
            
            print(f"Fine-tuning job started with ID: {job_id}")
            print("This job will continue in the background.")
            print("You can check its status later with the CLI tool:")
            print(f"  python tools/fine_tuning_cli.py openai status --job-id {job_id}")
            
        except Exception as e:
            print(f"Error starting fine-tuning: {str(e)}")

def main():
    """Main function to run the fine-tuning examples"""
    print("Fine-Tuning Examples")
    print("===================")
    
    # DSPy Optimization example
    demonstrate_dspy_optimization()
    
    # OpenAI Fine-Tuning example (using mock)
    mock_openai_fine_tuning(use_mock=True)
    
    print("\nExamples complete!")
    print("To run actual fine-tuning jobs, use the CLI tool:")
    print("  python tools/fine_tuning_cli.py --help")

if __name__ == "__main__":
    main() 