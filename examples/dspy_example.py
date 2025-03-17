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
import sys
from pathlib import Path
from typing import List, Dict, Any

###########################################
# STEP 1: ENVIRONMENT & DATA PREPARATION #
###########################################

print("Step 1: Environment & Data Preparation")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

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

# Additional data for testing and evaluation
TEST_SCENARIOS = [
    {
        "subject": "Biology",
        "student_queries": [
            "I'm confused about how DNA works.",
            "So DNA has two strands, but how does it copy itself?",
            "What happens if there's a mistake in the copying?"
        ],
        "misconceptions": ["DNA copies itself by splitting completely into two separate strands"]
    },
    {
        "subject": "Physics",
        "student_queries": [
            "Why do objects float in water?",
            "So it's about density? What about boats made of metal?",
            "Does that mean anything can float if shaped right?"
        ],
        "misconceptions": ["Metal always sinks", "Only light things can float"]
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

# Prepare dataset for DSPy optimization
def prepare_dspy_examples(dialogues):
    """Convert dialogues to DSPy example format for optimization"""
    examples = []
    
    for dialogue_data in dialogues:
        dialogue = dialogue_data["dialogue"]
        
        for i in range(1, len(dialogue), 2):
            if i >= len(dialogue) - 1:
                break  # Skip if there's no teacher response
                
            student_query = dialogue[i-1]["content"]
            teacher_response = dialogue[i]["content"]
            
            # Get conversation history up to this point
            history = format_dialogue_history(dialogue, i-2) if i > 1 else ""
            
            example = dspy.Example(
                conversation_history=history,
                student_query=student_query,
                teacher_response=teacher_response
            )
            examples.append(example)
    
    return examples

# Load dataset
dataset = load_jsonl(DATASET_PATH)
train_data, test_data = split_dataset(dataset, test_size=0.3)
print(f"- Split into {len(train_data)} training and {len(test_data)} test dialogue examples")

# Create DSPy examples for optimization
train_examples = prepare_dspy_examples(train_data)
test_examples = prepare_dspy_examples(test_data)
print(f"- Created {len(train_examples)} training examples and {len(test_examples)} test examples for DSPy")

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
# STEP 4: OPTIMIZATION (FINE-TUNING) #
#######################################

print("\nStep 4: DSPy Optimization")
print("- Using actual DSPy optimizers for prompt optimization")
print("- No model weight updates occur (unlike OpenAI or HuggingFace fine-tuning)")
print("- Training optimizes prompts for effective teacher-student interactions")

# Create model instances - one for pre-optimization, one for post-optimization
unoptimized_model = TeacherStudentDialogueModel()
print("- Created unoptimized TeacherStudentDialogue model")

# Define metric for optimization
class PedagogicalMetric(dspy.Metric):
    """Evaluates teacher responses based on pedagogical qualities."""
    
    def __init__(self):
        self.techniques = [
            "what do you think", 
            "can you explain", 
            "why do you think",
            "try to", 
            "let's break", 
            "for example",
            "does that make sense",
            "can you think of"
        ]
    
    def score(self, example, pred, trace=None):
        response = pred.teacher_response.lower()
        
        # Check for pedagogical techniques
        uses_technique = any(technique in response for technique in self.techniques)
        
        # Check for follow-up questions
        has_question = "?" in response
        
        # Check if response is just giving away the answer or guiding
        is_guiding = uses_technique or has_question
        
        # Check for explanations that build understanding
        has_explanation = len(response) > 100  # Simple heuristic for now
        
        # Calculate final score (0-1)
        score = 0.0
        if uses_technique:
            score += 0.4
        if has_question:
            score += 0.3
        if has_explanation:
            score += 0.3
        
        return min(1.0, score)  # Cap at 1.0

# Setup optimization
metric = PedagogicalMetric()
teleprompter = dspy.teleprompt.Teleprompter(teacher_model=unoptimized_model)

# Optimize the model
optimized_model = teleprompter.compile(
    student=unoptimized_model,
    trainset=train_examples[:min(len(train_examples), 10)],  # Limit for example purposes
    metric=metric,
    max_bootstrapped_demos=3,
    verbose=True
)

print("- Completed DSPy optimization")
print("- Created optimized TeacherStudentDialogue model")

###########################################
# STEP 5: EVALUATION OF FINE-TUNING      #
###########################################

print("\nStep 5: Comprehensive Evaluation")

def evaluate_teacher_responses(responses: List[str]) -> Dict[str, float]:
    """Evaluate the quality of teacher responses using multiple metrics"""
    metrics = {
        "pedagogical_techniques": 0,  # Socratic, scaffolding, etc.
        "follow_up_questions": 0,     # Teacher asking questions to check understanding
        "response_length": 0,         # Length of responses
        "adaptability": 0             # Adjusting to student level
    }
    
    # List of pedagogical techniques to check for
    techniques = [
        "what do you think", "can you explain", "why do you think", 
        "try to", "let's break", "for example", "you're right", 
        "that's correct", "good question", "excellent"
    ]
    
    total_length = 0
    adaptive_phrases = ["you mentioned", "as you said", "building on your", "you're right"]
    
    for response in responses:
        response_lower = response.lower()
        total_length += len(response)
        
        # Check for pedagogical techniques
        for technique in techniques:
            if technique in response_lower:
                metrics["pedagogical_techniques"] += 1
                break
                
        # Check for follow-up questions
        if "?" in response:
            metrics["follow_up_questions"] += 1
            
        # Check for adaptability
        for phrase in adaptive_phrases:
            if phrase in response_lower:
                metrics["adaptability"] += 1
                break
    
    # Calculate percentages and averages
    total = len(responses)
    if total > 0:
        metrics["pedagogical_techniques"] = (metrics["pedagogical_techniques"] / total) * 100
        metrics["follow_up_questions"] = (metrics["follow_up_questions"] / total) * 100
        metrics["response_length"] = total_length / total
        metrics["adaptability"] = (metrics["adaptability"] / total) * 100
    
    return metrics

def run_scenario_evaluation(model, scenarios: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Run a comprehensive evaluation across multiple teaching scenarios"""
    all_responses = []
    results_by_scenario = {}
    
    for i, scenario in enumerate(scenarios):
        # Reset conversation for fresh evaluation
        model.reset_conversation()
        
        # Run the complete dialogue scenario
        scenario_responses = []
        for student_query in scenario["student_queries"]:
            response = model.respond_to_student(student_query)
            scenario_responses.append(response)
            
            # Print the interaction
            print(f"\nScenario {i+1}, Subject: {scenario['subject']}")
            print(f"Student: {student_query}")
            print(f"Teacher: {response}")
        
        # Store responses
        all_responses.extend(scenario_responses)
        
        # Evaluate this scenario
        results_by_scenario[scenario['subject']] = evaluate_teacher_responses(scenario_responses)
    
    # Calculate overall metrics
    overall_metrics = evaluate_teacher_responses(all_responses)
    results_by_scenario["OVERALL"] = overall_metrics
    
    return results_by_scenario

print("- Evaluating unoptimized model...")
unoptimized_results = run_scenario_evaluation(unoptimized_model, TEST_SCENARIOS)

print("\n- Evaluating optimized model...")
optimized_results = run_scenario_evaluation(optimized_model, TEST_SCENARIOS)

# Compare results
print("\n--- Evaluation Results Comparison ---")
print(f"{'Metric':<25} {'Unoptimized':<15} {'Optimized':<15} {'Improvement':<15}")
print("-" * 70)

for metric in unoptimized_results["OVERALL"]:
    unopt_value = unoptimized_results["OVERALL"][metric]
    opt_value = optimized_results["OVERALL"][metric]
    improvement = opt_value - unopt_value
    improvement_str = f"{improvement:+.2f}"
    
    print(f"{metric:<25} {unopt_value:<15.2f} {opt_value:<15.2f} {improvement_str:<15}")

###########################################
# STEP 6: INTERACTIVE DEMONSTRATION      #
###########################################

print("\n\n--- Interactive Teacher-Student Dialogue Demo with Optimized Model ---")
print("Type 'exit' to end the conversation.\n")

optimized_model.reset_conversation()
while True:
    user_input = input("Student: ")
    if user_input.lower() == 'exit':
        break
    
    response = optimized_model.respond_to_student(user_input)
    print(f"Teacher: {response}")

##########################
# STEP 7: COMPARISON    #
##########################

print("\nStep 7: Comparison with other methods")
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
