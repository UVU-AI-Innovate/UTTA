"""
Configuration settings for the LLM evaluation framework.
"""

import os
from pathlib import Path
import torch

# Base paths
ROOT_DIR = Path(__file__).parent
CONFIG_DIR = ROOT_DIR / "config"
CONFIG_DIR.mkdir(exist_ok=True)

# Token file path
TOKEN_FILE = CONFIG_DIR / ".env"

def load_token():
    """Load Hugging Face token from various possible locations."""
    # Try loading from environment variable first
    token = os.getenv("HUGGING_FACE_TOKEN")
    if token:
        return token
    
    # Try loading from .env file
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, "r") as f:
            for line in f:
                if line.startswith("HUGGING_FACE_TOKEN="):
                    return line.split("=")[1].strip()
    
    # Try loading from ~/.huggingface/token
    hf_token_path = os.path.expanduser("~/.huggingface/token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            token = f.read().strip()
            if token:
                return token
    
    return None

def save_token(token):
    """Save Hugging Face token to both .env and ~/.huggingface/token."""
    # Save to .env file
    with open(TOKEN_FILE, "w") as f:
        f.write(f"HUGGING_FACE_TOKEN={token}\n")
    
    # Save to ~/.huggingface/token
    hf_token_path = os.path.expanduser("~/.huggingface/token")
    os.makedirs(os.path.dirname(hf_token_path), exist_ok=True)
    with open(hf_token_path, "w") as f:
        f.write(token)

# Available models configuration
AVAILABLE_MODELS = {
    "llama2-13b": {
        "name": "meta-llama/Llama-2-13b-hf",
        "description": "Best balance of size and performance, fast inference",
        "parameters": "13B",
        "vram_required": "24GB",
        "quantization": "4-bit",
        "recommended_gpu": "RTX 3090/4090 or better",
        "max_memory": {0: "11GB", 1: "11GB"}  # Split across GPUs
    },
    "mistral-7b": {
        "name": "HuggingFaceH4/mistral-7b-sft-beta",  # Using H4's fine-tuned version which is openly available
        "description": "Highly optimized, great reasoning, faster inference",
        "parameters": "7B",
        "vram_required": "12GB",
        "quantization": "4-bit",
        "recommended_gpu": "RTX 3060 12GB or better",
        "max_memory": {0: "11GB"}  # Can run on single GPU
    },
    "mixtral-8x7b": {
        "name": "mistralai/Mixtral-8x7B-v0.1",
        "description": "Top-tier performance, uses only 2 experts at a time",
        "parameters": "12B active",
        "vram_required": "24GB",
        "quantization": "4-bit",
        "recommended_gpu": "RTX 3090/4090 or better",
        "max_memory": {0: "11GB", 1: "11GB"}  # Split across GPUs
    },
    "gpt4all-falcon": {
        "name": "nomic-ai/gpt4all-falcon-7b",
        "description": "Fine-tuned Falcon model, fast and efficient",
        "parameters": "7B",
        "vram_required": "16GB",
        "quantization": "4-bit",
        "recommended_gpu": "RTX 3080 or better",
        "max_memory": {0: "14GB"}  # Can run on single GPU
    },
    "phi-2": {
        "name": "microsoft/phi-2",
        "description": "Very lightweight, strong on reasoning despite small size",
        "parameters": "2.7B",
        "vram_required": "4GB",
        "quantization": "4-bit",
        "recommended_gpu": "RTX 3050 or better",
        "max_memory": {0: "4GB"}  # Can run on small GPU
    },
    "llama2-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "description": "Solid performance, runs easily on 16GB GPUs",
        "parameters": "7B",
        "vram_required": "12GB",
        "quantization": "4-bit",
        "recommended_gpu": "RTX 3060 12GB or better",
        "max_memory": {0: "11GB"}  # Can run on single GPU
    }
}

# Default model and device settings
DEFAULT_MODEL_KEY = "phi-2"  # Default to most lightweight model
DEFAULT_MODEL_NAME = AVAILABLE_MODELS[DEFAULT_MODEL_KEY]["name"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_MAP = "auto"
OFFLOAD_FOLDER = "offload"

# Quantization settings
LOAD_IN_4BIT = True
COMPUTE_DTYPE = "float16"
QUANT_TYPE = "nf4"
DOUBLE_QUANT = True
TRUST_REMOTE_CODE = True

# Load token on import
HUGGING_FACE_TOKEN = load_token()

def get_available_vram():
    """Get available VRAM on each GPU."""
    if not torch.cuda.is_available():
        return None
    
    vram_info = {}
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
        reserved_memory = torch.cuda.memory_reserved(i) / 1024**3
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**3
        available_memory = total_memory - allocated_memory
        vram_info[i] = {
            "total": f"{total_memory:.1f}GB",
            "available": f"{available_memory:.1f}GB",
            "used": f"{allocated_memory:.1f}GB"
        }
    return vram_info

def list_available_models():
    """Print available models and their details."""
    vram_info = get_available_vram()
    
    print("\nAvailable Models:")
    print("=" * 80)
    
    if vram_info:
        print("\nDetected GPUs:")
        for gpu_id, memory in vram_info.items():
            print(f"GPU {gpu_id}:")
            print(f"  Total VRAM: {memory['total']}")
            print(f"  Available: {memory['available']}")
            print(f"  Used: {memory['used']}")
        print("\n" + "=" * 80)
    
    for key, model in AVAILABLE_MODELS.items():
        print(f"\nModel Key: {key}")
        print(f"Name: {model['name']}")
        print(f"Parameters: {model['parameters']}")
        print(f"VRAM Required: {model['vram_required']}")
        print(f"Description: {model['description']}")
        print(f"Recommended GPU: {model['recommended_gpu']}")
        print(f"Quantization: {model['quantization']}")
    
    print(f"\nCurrent Default Model: {DEFAULT_MODEL_KEY}")
    print("=" * 80)

def get_model_config(model_key=None):
    """Get configuration for specified model or default model."""
    key = model_key if model_key in AVAILABLE_MODELS else DEFAULT_MODEL_KEY
    model = AVAILABLE_MODELS[key]
    
    return {
        "model_name": model["name"],
        "device": DEVICE,
        "device_map": DEVICE_MAP,
        "max_memory": model["max_memory"],
        "offload_folder": OFFLOAD_FOLDER,
        "load_in_4bit": LOAD_IN_4BIT,
        "compute_dtype": COMPUTE_DTYPE,
        "quant_type": QUANT_TYPE,
        "double_quant": DOUBLE_QUANT,
        "trust_remote_code": TRUST_REMOTE_CODE
    } 