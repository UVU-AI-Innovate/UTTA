#!/usr/bin/env python
"""
DSPy Optimization Example for Educational QA

This example demonstrates prompt optimization using DSPy rather than traditional fine-tuning.
Key characteristics of this approach:
- No model weight updates (only prompt optimization)
- Very little training data needed (as few as 10-20 examples)
- No special infrastructure required
- Quick implementation and testing
- Limited to capabilities already in the base model
"""
import dspy
import json
import random
import os
from pathlib import Path

###########################################
# STEP 1: ENVIRONMENT & DATA PREPARATION #
###########################################

print("Step 1: Environment & Data Preparation")

# Configure DSPy to use OpenAI
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-3.5-turbo", temperature=0.7))
print("- LLM: OpenAI GPT-3.5-Turbo")

# Define a simple dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), "small_edu_qa.jsonl")

# Sample data for educational QA - same across all examples for comparison
SAMPLE_DATA = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "How does gravity work?", "answer": "Gravity is a force that attracts objects toward each other, keeping planets in orbit around the Sun."},
    {"question": "Who discovered electricity?", "answer": "Electricity was studied by multiple scientists, but Benjamin Franklin is famous for his kite experiment in 1752."},
    {"question": "What is photosynthesis?", "answer": "Photosynthesis is a process by which plants convert sunlight into energy, producing oxygen as a byproduct."},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare wrote Hamlet in the early 17th century."}
]

# Save the dataset if it does not exist
if not Path(DATASET_PATH).exists():
    with open(DATASET_PATH, "w") as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item) + "\n")
    print(f"- Sample dataset saved as {DATASET_PATH}")
else:
    print(f"- Using existing dataset from {DATASET_PATH}")

print(f"- Working with {len(SAMPLE_DATA)} examples (DSPy needs fewer than other methods)")

####################################
# STEP 2: DATA FORMAT PREPARATION #
####################################

print("\nStep 2: Data Preprocessing")

# Function to load dataset
def load_jsonl(file_path):
    """Load JSON Lines file into a list of dictionaries"""
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

# Function to split dataset
def split_dataset(dataset, test_size=0.3):
    """Split dataset into training and testing sets"""
    random.shuffle(dataset)
    split_index = int(len(dataset) * (1 - test_size))
    return dataset[:split_index], dataset[split_index:]

# Load dataset
dataset = load_jsonl(DATASET_PATH)
train_data, test_data = split_dataset(dataset, test_size=0.3)
print(f"- Split into {len(train_data)} training and {len(test_data)} test examples")

#######################################
# STEP 3: MODEL CONFIGURATION        #
#######################################

print("\nStep 3: Model Configuration")
print("- Approach: DSPy prompt optimization (no model weight updates)")
print("- Method: Chain-of-Thought reasoning")
print("- Base model remains unchanged")

# Initialize DSPy Signature and Module
class EducationalQA(dspy.Signature):
    """Answer educational questions with accurate information."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Using ChainOfThought for better reasoning
class EducationalQAModel(dspy.Module):
    def __init__(self, model_name="gpt-3.5-turbo"):
        super().__init__()
        print("- Using ChainOfThought to improve reasoning")
        self.predictor = dspy.ChainOfThought(EducationalQA)

    def forward(self, question):
        """Process a question and return an answer using chain-of-thought reasoning"""
        response = self.predictor(question=question)
        return response.answer

#######################################
# STEP 4: OPTIMIZATION (ALTERNATIVE  #
#         TO TRADITIONAL FINE-TUNING)#
#######################################

print("\nStep 4: Optimization Setup (not actual fine-tuning)")
print("- DSPy uses the existing model's capabilities through better prompting")
print("- No model weight updates occur (unlike OpenAI or HuggingFace fine-tuning)")
print("- In a full implementation, DSPy would optimize prompts using techniques like:")
print("  * Few-shot example selection")
print("  * Prompt augmentation")
print("  * Chain-of-thought guidance")

# Create model instance
model = EducationalQAModel()
print("- Created EducationalQA model instance")

# Note about DSPy optimizers that would be used in a more complex implementation
print("- Note: In a complete implementation, we would use DSPy optimizers like:")
print("  * dspy.BootstrapFewShot")
print("  * dspy.MultiPromptOptimizer")
print("  * These would select optimal examples and prompt formats")

########################################
# STEP 5: EVALUATION AND DEMONSTRATION #
########################################

print("\nStep 5: Evaluation and Demonstration")

# Evaluate the model
print("- Evaluating model on test data...")
correct_answers = 0
for example in test_data:
    question, expected_answer = example["question"], example["answer"]
    predicted_answer = model.forward(question)
    print(f"\nQ: {question}\nExpected: {expected_answer}\nPredicted: {predicted_answer}")

    # Simple evaluation (checks if key terms are in the response)
    if any(word in predicted_answer.lower() for word in expected_answer.lower().split()):
        correct_answers += 1

accuracy = correct_answers / len(test_data)
print(f"\nEvaluation Accuracy: {accuracy:.2f}")

# Predict an answer for a new question
new_question = "What causes the seasons on Earth?"
print("\n--- Demonstrating inference on new question ---")
print(f"Q: {new_question}")
new_answer = model.forward(new_question)
print(f"A: {new_answer}")

##########################
# STEP 6: COMPARISON    #
##########################

print("\nStep 6: Comparison with other methods")
print("- DSPy vs. OpenAI fine-tuning:")
print("  * DSPy: Optimizes prompts only, OpenAI: Updates model weights")
print("  * DSPy: Works with few examples, OpenAI: Needs more data")
print("  * DSPy: Lower cost (only API calls), OpenAI: Higher cost (training + API)")
print("  * DSPy: Limited to base model capabilities, OpenAI: Can enhance model capabilities")
print("\n- DSPy vs. HuggingFace LoRA:")
print("  * DSPy: No special hardware needed, HuggingFace: Requires GPU")
print("  * DSPy: Works through API, HuggingFace: Runs locally")
print("  * DSPy: Data shared with API provider, HuggingFace: Data stays local")
print("  * DSPy: Quick to implement, HuggingFace: More complex setup")
