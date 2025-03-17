#!/usr/bin/env python
"""
OpenAI Fine-Tuning Example for Educational QA

This example demonstrates fine-tuning an OpenAI model using their API.
Key characteristics of this approach:
- Cloud-based training (no local GPU needed)
- API-based workflow (simple to use)
- Medium amount of training data needed
- Limited control over training process
- Data goes to OpenAI servers
"""
import json
import os
import time
from pathlib import Path
import openai

###########################################
# STEP 1: ENVIRONMENT & DATA PREPARATION #
###########################################

# Configure OpenAI API key (would need to be set in environment)
openai.api_key = os.getenv("OPENAI_API_KEY")
print("Step 1: Environment & Data Preparation")

# Sample data for educational QA - same across all examples for comparison
SAMPLE_DATA = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "How does gravity work?", "answer": "Gravity is a force that attracts objects toward each other, keeping planets in orbit around the Sun."},
    {"question": "Who discovered electricity?", "answer": "Electricity was studied by multiple scientists, but Benjamin Franklin is famous for his kite experiment in 1752."},
    {"question": "What is photosynthesis?", "answer": "Photosynthesis is a process by which plants convert sunlight into energy, producing oxygen as a byproduct."},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare wrote Hamlet in the early 17th century."}
]

# Additional training examples - OpenAI requires more examples than DSPy
ADDITIONAL_DATA = [
    {"question": "What are the three states of matter?", "answer": "The three common states of matter are solid, liquid, and gas. Some sources also include plasma as a fourth state."},
    {"question": "How do vaccines work?", "answer": "Vaccines work by training the immune system to recognize and combat pathogens by introducing a harmless form of the pathogen to trigger an immune response."},
    {"question": "What is the water cycle?", "answer": "The water cycle is the continuous movement of water within Earth and its atmosphere, including processes like evaporation, condensation, precipitation, and collection."},
    {"question": "Who was Albert Einstein?", "answer": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to quantum mechanics."},
    {"question": "What causes earthquakes?", "answer": "Earthquakes are caused by sudden releases of energy in the Earth's crust, typically due to movement along fault lines."}
]

# Combine all training data
TRAIN_DATA = SAMPLE_DATA + ADDITIONAL_DATA
print(f"- Prepared {len(TRAIN_DATA)} examples for training")

####################################
# STEP 2: DATA FORMAT PREPARATION #
####################################

print("\nStep 2: Data Format Preparation")

# Prepare training file in OpenAI's format
def prepare_training_data(data, output_file):
    """
    Convert training data to OpenAI's fine-tuning format for 0.28.0
    
    IMPORTANT: The format required depends on OpenAI's API version:
    - For v0.28.0: prompt/completion format
    - For v1.x.x: messages format with role/content
    """
    formatted_data = []
    
    # Format each example according to OpenAI's requirements
    for item in data:
        formatted_item = {
            "prompt": item["question"],
            "completion": item["answer"]
        }
        formatted_data.append(formatted_item)
    
    # Save to JSONL file
    with open(output_file, "w") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"- Formatted data saved to {output_file}")
    return output_file

# Generate the formatted data file
training_file_path = os.path.join(os.path.dirname(__file__), "openai_edu_qa_training.jsonl")
prepare_training_data(TRAIN_DATA, training_file_path)

#######################################
# STEP 3: FINE-TUNING CONFIGURATION  #
#######################################

print("\nStep 3: Fine-Tuning Configuration")
print("- Base model: davinci (OpenAI)")
print("- Training method: Full model fine-tuning (server-side)")
print("- Infrastructure: OpenAI's cloud servers")
print("- Control: Limited to API parameters")

#######################################
# STEP 4: FINE-TUNING IMPLEMENTATION #
#######################################

print("\nStep 4: Fine-Tuning Implementation (commented out to prevent charges)")
print("- Note: This would normally upload data and start a fine-tuning job")
print("- Process runs asynchronously on OpenAI's servers")
print("- Training could take hours depending on dataset size")

# Function to upload the file to OpenAI
def upload_training_file(file_path):
    """Upload the training file to OpenAI - costs money when executed"""
    try:
        with open(file_path, "rb") as f:
            response = openai.File.create(
                file=f,
                purpose="fine-tune"
            )
        print(f"File uploaded successfully. File ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

# Create a fine-tuning job
def create_fine_tuning_job(file_id, model="davinci"):
    """Create a fine-tuning job on OpenAI - costs money when executed"""
    try:
        response = openai.FineTune.create(
            training_file=file_id,
            model=model
        )
        print(f"Fine-tuning job created. Job ID: {response.id}")
        return response.id
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

# Check fine-tuning job status
def check_job_status(job_id):
    """Check the status of a fine-tuning job"""
    try:
        job = openai.FineTune.retrieve(id=job_id)
        print(f"Job status: {job.status}")
        return job
    except Exception as e:
        print(f"Error checking job status: {e}")
        return None

# This is the code you would uncomment to actually run fine-tuning (costs money)
"""
# Step 4.1: Upload training file
file_id = upload_training_file(training_file_path)

if file_id:
    # Step 4.2: Create fine-tuning job
    job_id = create_fine_tuning_job(file_id)
    
    if job_id:
        # Step 4.3: Check job status (this would normally be done with polling)
        job = check_job_status(job_id)
        
        # In a real implementation, you would wait for the job to complete
        # This could take several hours depending on the dataset size
        
        # After job completion, the fine-tuned model ID would be available as:
        # fine_tuned_model = job.fine_tuned_model
"""

########################################
# STEP 5: INFERENCE AND DEMONSTRATION #
########################################

print("\nStep 5: Inference and Demonstration")

# Test the fine-tuned model function
def test_fine_tuned_model(model, question):
    """Test the fine-tuned model with a question - uses OpenAI credits"""
    try:
        response = openai.Completion.create(
            model=model,
            prompt=question,
            temperature=0.7,
            max_tokens=150
        )
        answer = response.choices[0].text.strip()
        print(f"\nQ: {question}\nA: {answer}")
        return answer
    except Exception as e:
        print(f"Error testing model: {e}")
        return None

# For demonstration, just simulate without API calls due to model deprecation issues
print("\n--- Demonstration (simulation) ---")
print("Q: What causes the seasons on Earth?")
print("A: The seasons on Earth are caused by the tilt of the Earth's axis as it orbits around the Sun. This tilt changes the angle at which sunlight hits different parts of the Earth throughout the year, creating seasonal variations in temperature and daylight hours.")

##########################
# STEP 6: COMPARISON    #
##########################

print("\nStep 6: Comparison with other methods")
print("- OpenAI fine-tuning vs. DSPy:")
print("  * OpenAI: Updates model weights, DSPy: Only optimizes prompts")
print("  * OpenAI: Needs more data, DSPy: Works with fewer examples")
print("  * OpenAI: Higher cost (training + API), DSPy: Only API costs")
print("  * OpenAI: More powerful adaptation, DSPy: Limited to base model capabilities")
print("\n- OpenAI fine-tuning vs. HuggingFace LoRA:")
print("  * OpenAI: Cloud-based, HuggingFace: Runs locally")
print("  * OpenAI: No GPU needed, HuggingFace: Requires GPU")
print("  * OpenAI: Data goes to OpenAI, HuggingFace: Data stays local")
print("  * OpenAI: Limited control, HuggingFace: Full control over training")

print("\nNote: To actually fine-tune, uncomment the steps above and ensure you have:")
print("1. Set your OpenAI API key as an environment variable (OPENAI_API_KEY)")
print("2. Have billing set up on your OpenAI account")
print("3. Understand that fine-tuning will incur charges on your OpenAI account") 