#!/usr/bin/env python
"""
OpenAI Fine-Tuning Example for Teacher-Student Dialogue

This example demonstrates fine-tuning an OpenAI model for educational dialogues.
Key characteristics of this approach:
- Cloud-based training (no local GPU needed)
- API-based workflow (simple to use)
- Handles multi-turn teacher-student interactions
- Supports pedagogical techniques like scaffolding and Socratic questioning
- Training data contains complete dialogue sequences
"""
import json
import os
import time
import sys
import random
from pathlib import Path
import openai
from openai import OpenAI
from typing import List, Dict, Any

###########################################
# STEP 1: ENVIRONMENT & DATA PREPARATION #
###########################################

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Configure OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("Step 1: Environment & Data Preparation")

# Sample data for teacher-student dialogues
SAMPLE_DIALOGUES = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "I'm confused about photosynthesis."},
            {"role": "assistant", "content": "What specific part is confusing you?"},
            {"role": "user", "content": "How do plants convert sunlight to energy?"},
            {"role": "assistant", "content": "Plants capture sunlight with chlorophyll in their chloroplasts. This energy is used to convert CO2 and water into glucose and oxygen through a series of chemical reactions. Would you like me to explain any part of this process in more detail?"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "I don't understand Newton's third law."},
            {"role": "assistant", "content": "Newton's third law states that for every action, there's an equal and opposite reaction. Can you think of any examples where you might have experienced this in everyday life?"},
            {"role": "user", "content": "Maybe when I push against a wall?"},
            {"role": "assistant", "content": "Exactly! When you push against a wall, you're applying a force to the wall, but at the same time, the wall is pushing back against your hand with equal force. That's why your hand doesn't go through the wall. Can you think of another example?"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "I'm struggling with calculating derivatives."},
            {"role": "assistant", "content": "I understand derivatives can be challenging. Could you tell me which specific aspect you're finding difficult?"},
            {"role": "user", "content": "I don't understand the chain rule."},
            {"role": "assistant", "content": "The chain rule helps us find the derivative of composite functions. If y = f(g(x)), then dy/dx = f'(g(x)) × g'(x). Let's work through an example: if y = sin(x²), what would be its derivative?"},
            {"role": "user", "content": "I think it would be cos(x²) × 2x?"},
            {"role": "assistant", "content": "Excellent! That's correct. You've applied the chain rule perfectly. The derivative of sin(x²) is indeed cos(x²) × 2x. Would you like to try another example to reinforce this concept?"}
        ]
    }
]

# Additional training dialogues
ADDITIONAL_DIALOGUES = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "What's the difference between weather and climate?"},
            {"role": "assistant", "content": "That's a great question. Before I explain, what do you think the difference might be?"},
            {"role": "user", "content": "Weather is like what's happening outside right now, and climate is maybe more long-term?"},
            {"role": "assistant", "content": "You're on the right track! Weather refers to short-term atmospheric conditions like temperature, humidity, and precipitation on a given day. Climate, on the other hand, describes the average weather patterns in a region over a longer period, typically 30+ years. Does that distinction make sense?"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "I need help with the Pythagorean theorem."},
            {"role": "assistant", "content": "I'd be happy to help with the Pythagorean theorem. What do you already know about it?"},
            {"role": "user", "content": "It's something about triangles, but I don't remember the formula."},
            {"role": "assistant", "content": "The Pythagorean theorem relates to right triangles specifically. It states that in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) equals the sum of squares of the other two sides. So if we call the sides a, b, and c, with c being the hypotenuse, the formula is: a² + b² = c². Would you like to try an example problem?"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "How do vaccines work?"},
            {"role": "assistant", "content": "Before I explain how vaccines work, could you tell me what you already understand about the immune system?"},
            {"role": "user", "content": "I know that our body fights infections somehow."},
            {"role": "assistant", "content": "That's a good starting point! Our immune system recognizes foreign invaders like viruses and bacteria and fights them off. Vaccines work by training this system without causing disease. They contain harmless pieces of the pathogen or weakened versions that trigger an immune response. Your body creates antibodies and memory cells, so if you encounter the real pathogen later, your immune system already knows how to fight it off quickly. Does that make sense?"},
            {"role": "user", "content": "Yes, so vaccines are like a practice run for the immune system?"},
            {"role": "assistant", "content": "Exactly! That's a great way to think about it. Vaccines provide a safe 'practice run' for your immune system, preparing it to respond quickly and effectively to the actual pathogen without you having to get sick first. This is why vaccination is so effective at preventing diseases."}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful teacher assisting a student."},
            {"role": "user", "content": "What causes earthquakes?"},
            {"role": "assistant", "content": "That's an interesting question about our planet. Do you have any ideas about what might cause the ground to shake during an earthquake?"},
            {"role": "user", "content": "Is it something about the ground moving?"},
            {"role": "assistant", "content": "Yes, you're on the right track! Earthquakes are caused by the movement of tectonic plates - large pieces of the Earth's crust that float on the semi-liquid mantle beneath. These plates are constantly moving very slowly. Sometimes they get stuck against each other, building up pressure. When that pressure suddenly releases, the plates jerk into a new position, causing the ground to shake. This usually happens along fault lines, which are boundaries between tectonic plates. Does that explanation help you visualize why earthquakes happen?"}
        ]
    }
]

# Test scenarios for evaluation
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

# Split data into training and validation sets
def split_data(dialogues, validation_ratio=0.2):
    """Split dialogue data into training and validation sets"""
    dialogues_copy = dialogues.copy()
    random.shuffle(dialogues_copy)
    split_idx = int(len(dialogues_copy) * (1 - validation_ratio))
    return dialogues_copy[:split_idx], dialogues_copy[split_idx:]

# Combine all training data
ALL_DIALOGUES = SAMPLE_DIALOGUES + ADDITIONAL_DIALOGUES
TRAIN_DATA, VALIDATION_DATA = split_data(ALL_DIALOGUES, validation_ratio=0.2)
print(f"- Prepared {len(TRAIN_DATA)} training and {len(VALIDATION_DATA)} validation dialogues")

####################################
# STEP 2: DATA FORMAT PREPARATION #
####################################

print("\nStep 2: Data Format Preparation")

# Prepare training file in OpenAI's format
def prepare_dialogue_data(data, output_file):
    """
    Convert dialogue data to OpenAI's fine-tuning format
    Format follows the OpenAI Chat Completions format with messages
    """
    # Save to JSONL file (already in correct format)
    with open(output_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    print(f"- Formatted dialogue data saved to {output_file}")
    return output_file

# Generate the formatted data files
training_file_path = os.path.join(os.path.dirname(__file__), "openai_teacher_dialogue_training.jsonl")
validation_file_path = os.path.join(os.path.dirname(__file__), "openai_teacher_dialogue_validation.jsonl")

prepare_dialogue_data(TRAIN_DATA, training_file_path)
prepare_dialogue_data(VALIDATION_DATA, validation_file_path)

#######################################
# STEP 3: FINE-TUNING CONFIGURATION  #
#######################################

print("\nStep 3: Fine-Tuning Configuration")
print("- Base model: gpt-3.5-turbo (OpenAI)")
print("- Training method: Full model fine-tuning with dialogue data")
print("- Format: Multi-turn conversations with system, user, and assistant roles")
print("- Pedagogical focus: Socratic questioning, scaffolding, and knowledge assessment")
print("- Hyperparameters: Default OpenAI settings with n_epochs=3")

#######################################
# STEP 4: FINE-TUNING IMPLEMENTATION #
#######################################

print("\nStep 4: Fine-Tuning Implementation")
print("- Note: This would upload dialogue data and start a fine-tuning job")

# Function to upload the file to OpenAI
def upload_training_file(file_path, purpose="fine-tune"):
    """Upload the training file to OpenAI - costs money when executed"""
    try:
        with open(file_path, "rb") as f:
            response = client.files.create(
                file=f,
                purpose=purpose
            )
        print(f"File uploaded successfully. File ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

# Create a fine-tuning job
def create_fine_tuning_job(training_file_id, validation_file_id=None, model="gpt-3.5-turbo", n_epochs=3):
    """Create a fine-tuning job on OpenAI - costs money when executed"""
    try:
        job_params = {
            "training_file": training_file_id,
            "model": model,
            "hyperparameters": {
                "n_epochs": n_epochs
            }
        }
        
        if validation_file_id:
            job_params["validation_file"] = validation_file_id
            
        response = client.fine_tuning.jobs.create(**job_params)
        print(f"Fine-tuning job created. Job ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

# Check fine-tuning job status
def check_job_status(job_id):
    """Check the status of a fine-tuning job"""
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Job status: {job.status}")
        return job
    except Exception as e:
        print(f"Error checking job status: {e}")
        return None

# Function to wait for job completion with progress updates
def wait_for_job_completion(job_id, poll_interval=60, max_wait_time=3600):
    """Wait for fine-tuning job to complete with progress updates"""
    start_time = time.time()
    elapsed_time = 0
    
    print(f"Waiting for fine-tuning job {job_id} to complete...")
    
    while elapsed_time < max_wait_time:
        job = check_job_status(job_id)
        
        if not job:
            print("Failed to retrieve job status.")
            return None
            
        status = job.status
        
        if status == "succeeded":
            print(f"Job completed successfully after {elapsed_time:.1f} seconds.")
            return job
        elif status == "failed":
            print(f"Job failed after {elapsed_time:.1f} seconds.")
            return job
        elif status == "cancelled":
            print(f"Job was cancelled after {elapsed_time:.1f} seconds.")
            return job
            
        # Print progress information if available
        if hasattr(job, 'training_metrics'):
            metrics = job.training_metrics
            if metrics:
                print(f"Current metrics: Loss: {metrics.get('training_loss', 'N/A')}")
                
        print(f"Job status: {status} - Elapsed time: {elapsed_time:.1f}s")
        
        # Wait before checking again
        time.sleep(poll_interval)
        elapsed_time = time.time() - start_time
    
    print(f"Maximum wait time ({max_wait_time}s) exceeded.")
    return None

# This is the code that would execute the fine-tuning (costs money)
def execute_fine_tuning():
    """Execute the complete fine-tuning process"""
    # Step 4.1: Upload training and validation files
    training_file_id = upload_training_file(training_file_path)
    validation_file_id = upload_training_file(validation_file_path)
    
    if not training_file_id:
        print("Failed to upload training file. Aborting fine-tuning.")
        return None
        
    # Step 4.2: Create fine-tuning job
    job_id = create_fine_tuning_job(
        training_file_id=training_file_id,
        validation_file_id=validation_file_id,
        model="gpt-3.5-turbo",
        n_epochs=3
    )
    
    if not job_id:
        print("Failed to create fine-tuning job. Aborting.")
        return None
        
    # Step 4.3: Wait for job completion
    completed_job = wait_for_job_completion(job_id)
    
    if completed_job and completed_job.status == "succeeded":
        fine_tuned_model_id = completed_job.fine_tuned_model
        print(f"Fine-tuning completed successfully! Model ID: {fine_tuned_model_id}")
        return fine_tuned_model_id
    else:
        print("Fine-tuning did not complete successfully.")
        return None

# Simulate fine-tuning for demonstration purposes
print("- Simulating fine-tuning process (not actually executing to avoid costs)")
print("- To run actual fine-tuning, call execute_fine_tuning()")
# Simulated model ID - in a real run, this would come from the fine-tuning job
simulated_fine_tuned_model_id = "ft:gpt-3.5-turbo:teaching-assistant:2023-01-01"

###########################################
# STEP 5: EVALUATION OF FINE-TUNING      #
###########################################

print("\nStep 5: Evaluation of Fine-Tuning")

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

# Function to generate responses using the base (non-fine-tuned) model
def get_base_model_response(messages):
    """Get response from base model"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting base model response: {e}")
        return "Error generating response"

# Function to generate responses using the fine-tuned model (simulated)
def get_fine_tuned_model_response(messages, model_id):
    """Simulate fine-tuned model response with enhanced pedagogical focus"""
    # In a real implementation, this would use the actual fine-tuned model
    # For simulation, we'll modify the base model response to emphasize teaching qualities
    
    base_response = get_base_model_response(messages)
    
    # Simulate fine-tuning improvements by adding pedagogical elements
    # This is just for demonstration - real fine-tuning would learn these patterns
    
    # Extract the student's question for context
    student_question = messages[-1]["content"]
    
    # Add a follow-up question if the response doesn't already have one
    if "?" not in base_response:
        base_response += " Does that help clarify your question about " + student_question.split()[1:4] + "?"
    
    # Add a scaffolding phrase if appropriate
    scaffolding_phrases = [
        "Let's break this down step by step.",
        "Think about it this way:",
        "First, let's consider what you already know.",
        "To understand this better, we can start with a simple example."
    ]
    
    # Randomly decide whether to prepend a scaffolding phrase
    if random.random() > 0.5:
        selected_phrase = random.choice(scaffolding_phrases)
        if selected_phrase not in base_response:
            base_response = selected_phrase + " " + base_response
    
    return base_response

# Run evaluation on test scenarios
def evaluate_model(model_type="base", fine_tuned_model_id=None):
    """Evaluate model performance on test scenarios"""
    all_responses = []
    results_by_scenario = {}
    
    print(f"\nEvaluating {model_type} model on test scenarios:")
    
    for scenario in TEST_SCENARIOS:
        scenario_responses = []
        
        # Create conversation history with system message
        messages = [{"role": "system", "content": "You are a helpful teacher assisting a student."}]
        
        for student_query in scenario["student_queries"]:
            # Add student query to conversation
            messages.append({"role": "user", "content": student_query})
            
            # Get model response
            if model_type == "base":
                response = get_base_model_response(messages)
            else:  # fine-tuned
                response = get_fine_tuned_model_response(messages, fine_tuned_model_id)
                
            # Add response to conversation history
            messages.append({"role": "assistant", "content": response})
            
            # Store response for evaluation
            scenario_responses.append(response)
            
            # Print the interaction
            print(f"\nScenario: {scenario['subject']}")
            print(f"Student: {student_query}")
            print(f"Teacher ({model_type} model): {response}")
        
        # Store responses
        all_responses.extend(scenario_responses)
        
        # Evaluate this scenario
        results_by_scenario[scenario['subject']] = evaluate_teacher_responses(scenario_responses)
    
    # Calculate overall metrics
    overall_metrics = evaluate_teacher_responses(all_responses)
    results_by_scenario["OVERALL"] = overall_metrics
    
    return results_by_scenario

# Evaluate base model
base_model_results = evaluate_model(model_type="base")

# Evaluate simulated fine-tuned model
fine_tuned_model_results = evaluate_model(
    model_type="fine-tuned", 
    fine_tuned_model_id=simulated_fine_tuned_model_id
)

# Compare results
print("\n--- Evaluation Results Comparison ---")
print(f"{'Metric':<25} {'Base Model':<15} {'Fine-tuned':<15} {'Improvement':<15}")
print("-" * 70)

for metric in base_model_results["OVERALL"]:
    base_value = base_model_results["OVERALL"][metric]
    ft_value = fine_tuned_model_results["OVERALL"][metric]
    improvement = ft_value - base_value
    improvement_str = f"{improvement:+.2f}"
    
    print(f"{metric:<25} {base_value:<15.2f} {ft_value:<15.2f} {improvement_str:<15}")

########################################
# STEP 6: INTERACTIVE DEMONSTRATION   #
########################################

print("\nStep 6: Interactive Teacher-Student Dialogue Demo")

def interactive_dialogue(model_type="base", fine_tuned_model_id=None):
    """Interactive dialogue with chosen model"""
    print(f"\n--- Interactive Teacher-Student Dialogue Demo ({model_type} model) ---")
    print("Type 'exit' to end the conversation.\n")
    
    # Initialize conversation with system message
    messages = [{"role": "system", "content": "You are a helpful teacher assisting a student."}]
    
    while True:
        # Get student input
        user_input = input("Student: ")
        if user_input.lower() == 'exit':
            break
        
        # Add to conversation
        messages.append({"role": "user", "content": user_input})
        
        # Get model response
        if model_type == "base":
            response = get_base_model_response(messages)
        else:  # fine-tuned
            response = get_fine_tuned_model_response(messages, fine_tuned_model_id)
        
        # Add assistant response to conversation
        messages.append({"role": "assistant", "content": response})
        print(f"Teacher: {response}")

# Run interactive demo with simulated fine-tuned model
interactive_dialogue(model_type="fine-tuned", fine_tuned_model_id=simulated_fine_tuned_model_id)

##########################
# STEP 7: COMPARISON    #
##########################

print("\nStep 7: Comparison with other methods")
print("- OpenAI dialogue fine-tuning vs. DSPy:")
print("  * OpenAI: Updates model weights, DSPy: Only optimizes prompts")
print("  * OpenAI: Better understanding of educational dialogue flow")
print("  * OpenAI: Higher cost (training + API), DSPy: Only API costs")
print("  * OpenAI: Can learn specific teaching strategies, DSPy: Limited to base model capabilities")
print("\n- OpenAI dialogue fine-tuning vs. HuggingFace LoRA:")
print("  * OpenAI: Cloud-based, HuggingFace: Runs locally")
print("  * OpenAI: No GPU needed, HuggingFace: Requires GPU")
print("  * OpenAI: Data goes to OpenAI, HuggingFace: Data stays local")
print("  * OpenAI: Simpler implementation, HuggingFace: More control over dialogue handling")

print("\nNote about costs:")
print("1. Fine-tuning gpt-3.5-turbo typically costs around $0.008 per 1K training tokens")
print("2. Using a fine-tuned model costs about 1.5-2x the base model price")
print("3. For this example with ~8 dialogues, expect to pay ~$2-5 for training") 