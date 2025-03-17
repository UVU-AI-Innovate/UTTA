#!/usr/bin/env python
"""
DSPy Optimization Example for Teacher-Student Dialogue

This example demonstrates prompt optimization using DSPy for creating a teacher-student dialogue chatbot.
Key characteristics of this approach:
- No model weight updates (only prompt optimization)
- Very little training data needed (as few as 10-20 dialogue examples)
- No special infrastructure required
- Models multi-turn educational conversations rather than simple Q&A
- Incorporates pedagogical techniques like Socratic questioning and scaffolding
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
DATASET_PATH = os.path.join(os.path.dirname(__file__), "teacher_student_dialogues.jsonl")

# Sample data for teacher-student dialogues
SAMPLE_DATA = [
    {
        "dialogue": [
            {"role": "student", "content": "I'm confused about photosynthesis."},
            {"role": "teacher", "content": "What specific part is confusing you?"},
            {"role": "student", "content": "How do plants convert sunlight to energy?"},
            {"role": "teacher", "content": "Plants capture sunlight with chlorophyll in their chloroplasts. This energy is used to convert CO2 and water into glucose and oxygen through a series of chemical reactions."}
        ]
    },
    {
        "dialogue": [
            {"role": "student", "content": "I don't understand Newton's third law."},
            {"role": "teacher", "content": "Newton's third law states that for every action, there's an equal and opposite reaction. Can you think of any examples where you might have experienced this in everyday life?"},
            {"role": "student", "content": "Maybe when I push against a wall?"},
            {"role": "teacher", "content": "Exactly! When you push against a wall, you're applying a force to the wall, but at the same time, the wall is pushing back against your hand with equal force. That's why your hand doesn't go through the wall. Can you think of another example?"}
        ]
    },
    {
        "dialogue": [
            {"role": "student", "content": "I'm struggling with calculating derivatives."},
            {"role": "teacher", "content": "I understand derivatives can be challenging. Could you tell me which specific aspect you're finding difficult?"},
            {"role": "student", "content": "I don't understand the chain rule."},
            {"role": "teacher", "content": "The chain rule helps us find the derivative of composite functions. If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x). Let's work through an example: if y = sin(x²), what would be its derivative?"},
            {"role": "student", "content": "I think it would be cos(x²) × 2x?"},
            {"role": "teacher", "content": "Excellent! That's correct. You've applied the chain rule perfectly. The derivative of sin(x²) is indeed cos(x²) × 2x."}
        ]
    },
    {
        "dialogue": [
            {"role": "student", "content": "What's the difference between weather and climate?"},
            {"role": "teacher", "content": "That's a great question. Before I explain, what do you think the difference might be?"},
            {"role": "student", "content": "Weather is like what's happening outside right now, and climate is maybe more long-term?"},
            {"role": "teacher", "content": "You're on the right track! Weather refers to short-term atmospheric conditions like temperature, humidity, and precipitation on a given day. Climate, on the other hand, describes the average weather patterns in a region over a longer period, typically 30+ years. Does that distinction make sense?"}
        ]
    },
    {
        "dialogue": [
            {"role": "student", "content": "I need help with the Pythagorean theorem."},
            {"role": "teacher", "content": "I'd be happy to help with the Pythagorean theorem. What do you already know about it?"},
            {"role": "student", "content": "It's something about triangles, but I don't remember the formula."},
            {"role": "teacher", "content": "The Pythagorean theorem relates to right triangles specifically. It states that in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) equals the sum of squares of the other two sides. So if we call the sides a, b, and c, with c being the hypotenuse, the formula is: a² + b² = c². Would you like to try an example problem?"}
        ]
    }
]

# Save the dataset if it does not exist
if not Path(DATASET_PATH).exists():
    with open(DATASET_PATH, "w") as f:
        for item in SAMPLE_DATA:
            f.write(json.dumps(item) + "\n")
    print(f"- Sample dialogue dataset saved as {DATASET_PATH}")
else:
    print(f"- Using existing dialogue dataset from {DATASET_PATH}")

print(f"- Working with {len(SAMPLE_DATA)} dialogue examples")

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

# Function to format dialogue for DSPy input
def format_dialogue_history(dialogue, up_to_index=-1):
    """Format dialogue history up to specified turn for model input"""
    if up_to_index == -1:
        up_to_index = len(dialogue) - 1
    
    history = ""
    for i, turn in enumerate(dialogue[:up_to_index + 1]):
        role = turn["role"].capitalize()
        history += f"{role}: {turn['content']}\n"
    
    return history.strip()

# Load dataset
dataset = load_jsonl(DATASET_PATH)
train_data, test_data = split_dataset(dataset, test_size=0.3)
print(f"- Split into {len(train_data)} training and {len(test_data)} test dialogue examples")

#######################################
# STEP 3: MODEL CONFIGURATION        #
#######################################

print("\nStep 3: Model Configuration")
print("- Approach: DSPy prompt optimization for teacher-student dialogue")
print("- Method: Conversation history with pedagogical techniques")
print("- Base model remains unchanged")

# Initialize DSPy Signature and Module
class TeacherResponse(dspy.Signature):
    """Generate a teacher's response in an educational dialogue with a student."""
    conversation_history = dspy.InputField(desc="The conversation history between teacher and student so far")
    student_query = dspy.InputField(desc="The student's most recent question or statement")
    teacher_response = dspy.OutputField(desc="A helpful, pedagogically-sound response that guides the student toward understanding")

# Using ChainOfThought for better reasoning
class TeacherStudentDialogueModel(dspy.Module):
    def __init__(self, model_name="gpt-3.5-turbo"):
        super().__init__()
        print("- Using ChainOfThought for pedagogical reasoning")
        self.predictor = dspy.ChainOfThought(TeacherResponse)
        self.conversation_history = []

    def add_to_history(self, role, content):
        """Add a turn to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def get_formatted_history(self):
        """Get the formatted conversation history for model input"""
        if not self.conversation_history:
            return ""
        
        history = ""
        for turn in self.conversation_history:
            role = turn["role"].capitalize()
            history += f"{role}: {turn['content']}\n"
        
        return history.strip()
    
    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

    def respond_to_student(self, student_query):
        """Process a student query and return a teacher response"""
        # Add student query to history
        self.add_to_history("student", student_query)
        
        # Get current conversation history
        conversation_history = self.get_formatted_history()
        
        # Generate teacher response
        response = self.predictor(
            conversation_history=conversation_history,
            student_query=student_query
        )
        
        # Add teacher response to history
        self.add_to_history("teacher", response.teacher_response)
        
        return response.teacher_response

#######################################
# STEP 4: OPTIMIZATION (ALTERNATIVE  #
#         TO TRADITIONAL FINE-TUNING)#
#######################################

print("\nStep 4: Optimization Setup (not actual fine-tuning)")
print("- DSPy uses the existing model's capabilities through better prompting")
print("- No model weight updates occur (unlike OpenAI or HuggingFace fine-tuning)")
print("- The system optimizes prompts for effective teacher-student interactions")

# Create model instance
dialogue_model = TeacherStudentDialogueModel()
print("- Created TeacherStudentDialogue model instance")

# In a full implementation, we would use DSPy optimizers here
print("- Note: In a complete implementation, DSPy optimizers would be used to:")
print("  * Learn optimal prompting for different teaching scenarios")
print("  * Improve pedagogical strategies based on student responses")
print("  * Balance explanation with questioning to promote understanding")

########################################
# STEP 5: EVALUATION AND DEMONSTRATION #
########################################

print("\nStep 5: Evaluation and Demonstration")

# Function to evaluate a full dialogue example
def evaluate_dialogue(model, dialogue):
    """Evaluate model on a complete dialogue example"""
    print("\n--- Dialogue Evaluation ---")
    
    # Reset conversation history
    model.reset_conversation()
    
    # Process each turn in the dialogue
    for i in range(0, len(dialogue) - 1, 2):
        # Get student query and expected teacher response
        student_query = dialogue[i]["content"]
        expected_response = dialogue[i+1]["content"]
        
        print(f"\nStudent: {student_query}")
        
        # Get model's response
        predicted_response = model.respond_to_student(student_query)
        
        print(f"Expected Teacher: {expected_response}")
        print(f"Predicted Teacher: {predicted_response}")
    
    return True  # In a real implementation, return a score

# Evaluate the model on one test dialogue
if test_data:
    test_dialogue = test_data[0]["dialogue"]
    evaluate_dialogue(dialogue_model, test_dialogue)

# Interactive demo
print("\n--- Interactive Teacher-Student Dialogue Demo ---")
print("Type 'exit' to end the conversation.\n")

dialogue_model.reset_conversation()
while True:
    user_input = input("Student: ")
    if user_input.lower() == 'exit':
        break
    
    response = dialogue_model.respond_to_student(user_input)
    print(f"Teacher: {response}")

##########################
# STEP 6: COMPARISON    #
##########################

print("\nStep 6: Comparison with other methods")
print("- DSPy dialogue vs. OpenAI fine-tuning:")
print("  * DSPy: Optimizes prompts only, OpenAI: Updates model weights")
print("  * DSPy: Multi-turn context through explicit history tracking")
print("  * DSPy: Lower cost (only API calls), OpenAI: Higher cost (training + API)")
print("  * DSPy: Limited to base model capabilities, OpenAI: Can enhance model capabilities")
print("\n- DSPy dialogue vs. HuggingFace LoRA:")
print("  * DSPy: No special hardware needed, HuggingFace: Requires GPU")
print("  * DSPy: Works through API, HuggingFace: Runs locally")
print("  * DSPy: Data shared with API provider, HuggingFace: Data stays local")
print("  * DSPy: Quick to implement, HuggingFace: More complex dialogue management")
