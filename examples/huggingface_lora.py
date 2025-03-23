#!/usr/bin/env python
"""
HuggingFace LoRA Fine-Tuning Example for Educational QA

This example demonstrates fine-tuning an open-source LLM using Low-Rank Adaptation (LoRA).
Key characteristics of this approach:
- Local model training (requires GPU)
- Full control over the training process
- Data stays on your infrastructure (privacy)
- One-time compute cost (no ongoing API fees)
- Can deploy models anywhere after training
- Technical expertise needed
"""
import json
import os
import sys
from pathlib import Path

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

# Check if we should run in simulation mode (no GPU/compatibility issues)
SIMULATION_MODE = False
try:
    import torch
    # Check if GPU is available or if we have compatibility issues
    if not torch.cuda.is_available():
        print("WARNING: GPU not available. Running in simulation mode.")
        SIMULATION_MODE = True
except (ImportError, Exception) as e:
    print(f"WARNING: Error importing PyTorch or checking GPU: {e}")
    print("Running in simulation mode.")
    SIMULATION_MODE = True

if not SIMULATION_MODE:
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            BitsAndBytesConfig, 
            TrainingArguments, 
            Trainer,
            DataCollatorForLanguageModeling
        )
        from peft import (
            LoraConfig, 
            get_peft_model, 
            prepare_model_for_kbit_training,
            TaskType
        )
    except ImportError as e:
        print(f"WARNING: Error importing required libraries: {e}")
        print("Running in simulation mode.")
        SIMULATION_MODE = True

###########################################
# STEP 1: ENVIRONMENT & DATA PREPARATION #
###########################################

print("Step 1: Environment & Data Preparation")

# Check for GPU availability
if not SIMULATION_MODE:
    if torch.cuda.is_available():
        print(f"- GPU available: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"- VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("- No GPU detected. This script requires a GPU to run efficiently.")
        print("- Training would be extremely slow on CPU and may run out of memory.")
        print("- Switching to simulation mode.")
        SIMULATION_MODE = True
else:
    print("- Running in simulation mode (no GPU required)")
    print("- This will demonstrate the workflow without actual model training")

# Sample data for educational QA - same across all examples for comparison
SAMPLE_DATA = [
    {"question": "What is the capital of France?", "answer": "The capital of France is Paris."},
    {"question": "How does gravity work?", "answer": "Gravity is a force that attracts objects toward each other, keeping planets in orbit around the Sun."},
    {"question": "Who discovered electricity?", "answer": "Electricity was studied by multiple scientists, but Benjamin Franklin is famous for his kite experiment in 1752."},
    {"question": "What is photosynthesis?", "answer": "Photosynthesis is a process by which plants convert sunlight into energy, producing oxygen as a byproduct."},
    {"question": "Who wrote Hamlet?", "answer": "William Shakespeare wrote Hamlet in the early 17th century."}
]

# Additional training examples for LoRA fine-tuning
ADDITIONAL_DATA = [
    {"question": "What are the three states of matter?", "answer": "The three common states of matter are solid, liquid, and gas. Some sources also include plasma as a fourth state."},
    {"question": "How do vaccines work?", "answer": "Vaccines work by training the immune system to recognize and combat pathogens by introducing a harmless form of the pathogen to trigger an immune response."},
    {"question": "What is the water cycle?", "answer": "The water cycle is the continuous movement of water within Earth and its atmosphere, including processes like evaporation, condensation, precipitation, and collection."},
    {"question": "Who was Albert Einstein?", "answer": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to quantum mechanics."},
    {"question": "What causes earthquakes?", "answer": "Earthquakes are caused by sudden releases of energy in the Earth's crust, typically due to movement along fault lines."}
]

# Add more examples to reach a minimum training dataset size
# HuggingFace fine-tuning typically needs more examples than other methods
EXTRA_DATA = [
    {"question": "What is the process of digestion?", "answer": "Digestion is the process of breaking down food into nutrients that can be absorbed by the body, involving mechanical and chemical processes."},
    {"question": "How do airplanes fly?", "answer": "Airplanes fly due to lift created by the shape of the wings and forward motion, following Bernoulli's principle and Newton's laws of motion."},
    {"question": "What is the periodic table?", "answer": "The periodic table is a systematic arrangement of chemical elements organized by atomic number, electron configuration, and recurring chemical properties."},
    {"question": "How does a computer process information?", "answer": "Computers process information using a central processing unit (CPU) that performs arithmetic, logical, control, and input/output operations based on instructions."},
    {"question": "What is the theory of evolution?", "answer": "The theory of evolution explains how species change over time through natural selection, where beneficial traits are passed on to offspring."},
    {"question": "How do magnets work?", "answer": "Magnets work due to the alignment of electron spins in their atoms, creating a magnetic field that can attract or repel other magnetic materials."},
    {"question": "What is the greenhouse effect?", "answer": "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat, warming the planet's surface to a habitable temperature."},
    {"question": "How do plants reproduce?", "answer": "Plants reproduce through various methods including seeds, spores, or vegetative reproduction, often involving pollination and fertilization processes."},
    {"question": "What is DNA?", "answer": "DNA (deoxyribonucleic acid) is a molecule that contains the genetic information and instructions for development, functioning, and reproduction of all living organisms."},
    {"question": "How does the immune system work?", "answer": "The immune system protects the body from harmful substances by recognizing and responding to antigens, using various cells, tissues, and organs to fight disease."}
]

# Combine all training data
TRAIN_DATA = SAMPLE_DATA + ADDITIONAL_DATA + EXTRA_DATA
print(f"- Prepared {len(TRAIN_DATA)} examples for training (LoRA fine-tuning benefits from more examples)")

####################################
# STEP 2: DATA FORMAT PREPARATION #
####################################

print("\nStep 2: Data Format Preparation")

# Configuration for the training
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Using a smaller instruct model
print(f"- Base model: {MODEL_NAME}")
print("- This model would be downloaded to your local machine")

# Prepare dataset formatting
def format_prompt(example):
    """
    Format a single example as a prompt for instruction tuning
    
    Unlike API-based approaches, we have full control over prompt formatting
    and can customize it specifically for our model.
    """
    return f"""<s>[INST] You are an educational assistant. Answer the following question with accurate information.

Question: {example['question']} [/INST]

{example['answer']}</s>"""

def prepare_dataset():
    """Prepare the dataset for training with the appropriate formatting"""
    if SIMULATION_MODE:
        print("- Simulation: Dataset preparation step skipped")
        return None
        
    # Create a HuggingFace dataset object
    ds = Dataset.from_list(TRAIN_DATA)
    
    # Add formatted text column using our prompt template
    ds = ds.map(lambda x: {"formatted_text": format_prompt(x)})
    
    # Store dataset for inspection (optional)
    ds.to_json(os.path.join(os.path.dirname(__file__), "edu_qa_formatted.jsonl"))
    print("- Formatted dataset saved to edu_qa_formatted.jsonl")
    
    return ds

print("- Preparing dataset with instruction format...")
# This would create the dataset in a real implementation
# dataset = prepare_dataset()
print("- Dataset preparation would convert QA pairs into instruction format")

# Tokenization explanation (we're not running this code to save time)
print("- Tokenization would convert text to token IDs")
print("- For LoRA fine-tuning, training works directly on token IDs")

#######################################
# STEP 3: FINE-TUNING CONFIGURATION  #
#######################################

print("\nStep 3: Fine-Tuning Configuration")

# LoRA hyperparameters (Low-Rank Adaptation)
CUTOFF_LEN = 512  # Maximum sequence length
LORA_R = 8        # LoRA attention dimension
LORA_ALPHA = 16   # Alpha parameter for LoRA
LORA_DROPOUT = 0.05
BATCH_SIZE = 4    # Smaller batch size for limited GPU memory
MICRO_BATCH_SIZE = 1  # Gradient accumulation steps
EPOCHS = 1        # Number of training epochs
LEARNING_RATE = 3e-4
TRAIN_ON_INPUTS = True  # Train on inputs as well as outputs

print(f"- Training approach: Low-Rank Adaptation (LoRA)")
print(f"- LoRA rank: {LORA_R} (smaller = less parameters, larger = more expressive)")
print(f"- LoRA alpha: {LORA_ALPHA} (scaling factor)")
print(f"- Batch size: {BATCH_SIZE} (limited by GPU memory)")
print(f"- Learning rate: {LEARNING_RATE}")
print(f"- Training epochs: {EPOCHS}")
print(f"- Sequence length: {CUTOFF_LEN} tokens")

print("\n- Key advantage of LoRA:")
print("  * Only trains a small number of parameters (~0.1-1% of full model)")
print("  * Enables fine-tuning even on consumer GPUs")
print("  * Resulting adapter is small (typically <100 MB)")

#######################################
# STEP 4: FINE-TUNING IMPLEMENTATION #
#######################################

print("\nStep 4: Fine-Tuning Implementation (simulation only)")
print("- Note: Real training would take hours and require a GPU")

def setup_model_and_tokenizer():
    """
    Demonstration of how to set up the model and tokenizer with LoRA
    
    This function shows the key steps in LoRA integration:
    1. Load base model with quantization (to reduce VRAM usage)
    2. Prepare model for LoRA training 
    3. Apply LoRA configuration
    """
    print("- Loading model and tokenizer (would download ~4GB)...")
    
    if SIMULATION_MODE:
        print("- Simulation: Model and tokenizer loading skipped")
        return None, None
    
    # Configure 4-bit quantization to reduce memory requirements
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # In real implementation, this would load the actual model
    print("- Model loading would use quantization to reduce VRAM requirements")
    print("- QLoRA (Quantized LoRA) enables training on consumer GPUs")
    
    # Configure LoRA targeting specific layers
    lora_config = LoraConfig(
        r=LORA_R,                       # LoRA dimension
        lora_alpha=LORA_ALPHA,          # LoRA scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
        lora_dropout=LORA_DROPOUT,      # Dropout for regularization
        bias="none",                    # No bias fine-tuning
        task_type=TaskType.CAUSAL_LM    # Causal language modeling task
    )
    
    print("- LoRA adapters applied to attention layers (most parameter-efficient)")
    
    return None, None  # Would return model and tokenizer in real implementation

def train():
    """
    Main training function - simulation only
    
    In a real implementation, this would:
    1. Set up the model with LoRA
    2. Configure training arguments 
    3. Create a Trainer
    4. Run training
    5. Save the resulting adapter
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"- Created output directory: {OUTPUT_DIR}")
    
    print("- Full training would involve:")
    print("  * Preparing dataset with instruction format")
    print("  * Loading quantized base model (~4GB download)")
    print("  * Applying LoRA configuration")
    print("  * Training for 1 epoch (~1-2 hours on a good GPU)")
    print("  * Saving LoRA adapter (~15MB)")
    
    # Simulate adapter saving in simulation mode
    if SIMULATION_MODE:
        with open(os.path.join(OUTPUT_DIR, "adapter_config.json"), "w") as f:
            json.dump({
                "r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": LORA_DROPOUT,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }, f, indent=2)
        print(f"- Created simulated adapter configuration file")
    
    print("\n- Training complete! (simulation)")

########################################
# STEP 5: INFERENCE AND DEMONSTRATION #
########################################

print("\nStep 5: Inference and Demonstration")

def generate_answer(question, model_path):
    """Generate an answer using the fine-tuned model - simulation only"""
    print(f"- Real inference would:")
    print(f"  * Load the base model")
    print(f"  * Apply the trained LoRA adapter")
    print(f"  * Run inference locally (no API calls needed)")
    
    # Format prompt the same way as during training
    prompt = f"""<s>[INST] You are an educational assistant. Answer the following question with accurate information.

Question: {question} [/INST]

"""
    
    # In real implementation, we'd generate text with the model
    # But for this simulation, we'll return a pre-defined answer
    
    return "The seasons on Earth are caused by the tilt of Earth's axis relative to its orbit around the Sun. As Earth orbits the Sun, different parts of the planet receive sunlight at different angles, creating seasonal variations in temperature and daylight hours."

# Demonstrate inference with a pre-defined answer
print("\n--- Demonstration (simulation of inference) ---")
test_question = "What causes the seasons on Earth?"
print(f"Q: {test_question}")
answer = generate_answer(test_question, "results/final_model")
print(f"A: {answer}")

##########################
# STEP 6: COMPARISON    #
##########################

print("\nStep 6: Comparison with other methods")
print("- HuggingFace LoRA vs. OpenAI fine-tuning:")
print("  * HuggingFace: Runs locally, OpenAI: Cloud-based")
print("  * HuggingFace: Requires GPU, OpenAI: No special hardware")
print("  * HuggingFace: Data stays local, OpenAI: Data goes to OpenAI")
print("  * HuggingFace: One-time compute cost, OpenAI: Pay per API call + training")
print("  * HuggingFace: Full control, OpenAI: Limited to API parameters")
print("\n- HuggingFace LoRA vs. DSPy:")
print("  * HuggingFace: Updates model weights, DSPy: Only optimizes prompts")
print("  * HuggingFace: Requires GPU, DSPy: No special hardware")
print("  * HuggingFace: Complex setup, DSPy: Simple implementation")
print("  * HuggingFace: Needs more data, DSPy: Works with fewer examples")
print("  * HuggingFace: Can improve model capabilities, DSPy: Limited to base model")

print("\nNote: To run actual training, you would need:")
print("1. A GPU with at least 16GB of VRAM")
print("2. Required libraries installed (transformers, peft, bitsandbytes, etc.)")
print("3. Sufficient disk space for model checkpoints (~10GB)")
print("4. Uncomment the training code in this file") 