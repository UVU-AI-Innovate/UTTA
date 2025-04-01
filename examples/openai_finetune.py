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
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from parent directory if this is in examples/
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded environment variables from {env_path}")
    else:
        load_dotenv()  # Try default locations
        print("Loaded environment variables from default locations")
except ImportError:
    print("Warning: dotenv package not found. Environment variables must be set manually.")

###########################################
# STEP 1: DATA MANAGEMENT CLASSES        #
###########################################

@dataclass
class Message:
    """Represents a single message in a dialogue"""
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class Dialogue:
    """Represents a complete dialogue sequence"""
    messages: List[Message]
    subject: str
    difficulty: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": {
                "subject": self.subject,
                "difficulty": self.difficulty,
                "tags": self.tags
            }
        }

class DatasetManager:
    """Manages dialogue datasets for fine-tuning"""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        self.dialogues: List[Dialogue] = []
        self.ensure_data_directory()

    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        os.makedirs(self.data_dir, exist_ok=True)

    def add_dialogue(self, dialogue: Dialogue):
        """Add a single dialogue to the dataset"""
        self.dialogues.append(dialogue)

    def add_dialogues_from_file(self, filepath: str):
        """Load dialogues from a JSON or JSONL file"""
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
                for item in data:
                    self.add_dialogue_from_dict(item)
            else:  # JSONL
                for line in f:
                    item = json.loads(line)
                    self.add_dialogue_from_dict(item)

    def add_dialogue_from_dict(self, data: Dict[str, Any]):
        """Convert dictionary to Dialogue object and add it"""
        messages = [Message(**msg) for msg in data["messages"]]
        metadata = data.get("metadata", {})
        dialogue = Dialogue(
            messages=messages,
            subject=metadata.get("subject", "general"),
            difficulty=metadata.get("difficulty", "intermediate"),
            tags=metadata.get("tags", [])
        )
        self.add_dialogue(dialogue)

    def save_to_file(self, filename: str, subset: Optional[List[str]] = None):
        """Save dialogues to a JSONL file, optionally filtering by subjects"""
        filepath = os.path.join(self.data_dir, filename)
        dialogues_to_save = self.dialogues
        if subset:
            dialogues_to_save = [d for d in self.dialogues if d.subject in subset]
            
        with open(filepath, 'w') as f:
            for dialogue in dialogues_to_save:
                f.write(json.dumps(dialogue.to_dict()) + '\n')
        print(f"Saved {len(dialogues_to_save)} dialogues to {filepath}")

    def split_train_val(self, val_ratio: float = 0.2) -> tuple[List[Dialogue], List[Dialogue]]:
        """Split dialogues into training and validation sets"""
        random.shuffle(self.dialogues)
        split_idx = int(len(self.dialogues) * (1 - val_ratio))
        return self.dialogues[:split_idx], self.dialogues[split_idx:]

    def get_dialogues_by_subject(self, subject: str) -> List[Dialogue]:
        """Get all dialogues for a specific subject"""
        return [d for d in self.dialogues if d.subject == subject]

    def get_dialogues_by_difficulty(self, difficulty: str) -> List[Dialogue]:
        """Get all dialogues of a specific difficulty level"""
        return [d for d in self.dialogues if d.difficulty == difficulty]

    def get_dialogues_by_tags(self, tags: List[str]) -> List[Dialogue]:
        """Get all dialogues that have any of the specified tags"""
        return [d for d in self.dialogues if any(tag in d.tags for tag in tags)]

###########################################
# STEP 2: SAMPLE DIALOGUE CREATION       #
###########################################

def create_sample_dialogues() -> List[Dialogue]:
    """Create sample dialogues for different subjects"""
    return [
        Dialogue(
            messages=[
                Message("system", "You are a helpful teacher assisting a student."),
                Message("user", "I'm confused about photosynthesis."),
                Message("assistant", "What specific part is confusing you?"),
                Message("user", "How do plants convert sunlight to energy?"),
                Message("assistant", "Plants capture sunlight with chlorophyll in their chloroplasts. This energy is used to convert CO2 and water into glucose and oxygen through a series of chemical reactions. Would you like me to explain any part of this process in more detail?")
            ],
            subject="biology",
            difficulty="beginner",
            tags=["photosynthesis", "plants", "energy"]
        ),
        # Add more sample dialogues here
    ]

###########################################
# STEP 3: DATASET INITIALIZATION         #
###########################################

def initialize_dataset():
    """Initialize the dataset with sample dialogues"""
    dataset = DatasetManager()
    
    # Add sample dialogues
    for dialogue in create_sample_dialogues():
        dataset.add_dialogue(dialogue)
    
    # Load additional dialogues from files
    for filename in os.listdir(dataset.data_dir):
        if filename.endswith(('.json', '.jsonl')):
            dataset.add_dialogues_from_file(os.path.join(dataset.data_dir, filename))
    
    return dataset

###########################################
# STEP 4: DATA PREPARATION              #
###########################################

print("\nStep 4: Data Preparation")

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Set it with: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Configure OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Step 4: Data Preparation")

####################################
# STEP 5: FINE-TUNING CONFIGURATION #
####################################

print("\nStep 5: Fine-Tuning Configuration")
print("- Base model: gpt-3.5-turbo (OpenAI)")
print("- Training method: Full model fine-tuning with dialogue data")
print("- Format: Multi-turn conversations with system, user, and assistant roles")
print("- Pedagogical focus: Socratic questioning, scaffolding, and knowledge assessment")
print("- Hyperparameters: Default OpenAI settings with n_epochs=3")

#######################################
# STEP 6: FINE-TUNING IMPLEMENTATION #
#######################################

print("\nStep 6: Fine-Tuning Implementation")
print("- Note: This would upload dialogue data and start a fine-tuning job")

# Function to upload the file to OpenAI
def upload_training_file(file_path, purpose="fine-tune"):
    """Upload the training file to OpenAI - costs money when executed"""
    try:
        with open(file_path, "rb") as f:
            response = openai.File.create(
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
            
        response = openai.FineTuningJob.create(**job_params)
        print(f"Fine-tuning job created. Job ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

# Check fine-tuning job status
def check_job_status(job_id):
    """Check the status of a fine-tuning job"""
    try:
        job = openai.FineTuningJob.retrieve(job_id)
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
# STEP 7: EVALUATION OF FINE-TUNING      #
###########################################

print("\nStep 7: Evaluation of Fine-Tuning")

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
        response = openai.ChatCompletion.create(
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
        # Get the first few words of the question as context
        question_words = student_question.split()
        question_context = " ".join(question_words[1:4]) if len(question_words) > 3 else student_question
        base_response += f" Does that help clarify your question about {question_context}?"
    
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
# STEP 8: INTERACTIVE DEMONSTRATION   #
########################################

print("\nStep 8: Interactive Teacher-Student Dialogue Demo")

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
# STEP 9: COMPARISON    #
##########################

print("\nStep 9: Comparison with other methods")
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

if __name__ == "__main__":
    # Initialize dataset
    dataset = initialize_dataset()
    
    # Split into train/val sets
    train_dialogues, val_dialogues = dataset.split_train_val()
    
    # Save to files
    dataset.save_to_file("train.jsonl")
    dataset.save_to_file("val.jsonl")
    
    print(f"Prepared {len(train_dialogues)} training and {len(val_dialogues)} validation dialogues")
    
    # Continue with fine-tuning process...
    # (rest of the main execution code) 