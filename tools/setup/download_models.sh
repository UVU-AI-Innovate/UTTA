#!/bin/bash

# Script to download Llama 3 model files from Hugging Face
# This script requires huggingface-cli to be installed
# Install with: pip install huggingface_hub

echo "Llama 3 Model Downloader"
echo "========================"
echo ""

# Create models directory if it doesn't exist
mkdir -p models
echo "Created models directory."

# Function to download model
download_model() {
    local MODEL_SIZE=$1
    local QUANT_LEVEL=$2
    local OUTPUT_NAME=$3
    
    local REPO="TheBloke/Llama-3-${MODEL_SIZE}B-GGUF"
    local FILENAME="llama-3-${MODEL_SIZE}b-${QUANT_LEVEL}.gguf"
    local OUTPUT_PATH="models/${OUTPUT_NAME}"
    
    echo ""
    echo "Downloading ${FILENAME} from ${REPO}..."
    echo "This may take a while depending on your internet connection."
    echo ""
    
    # Download the model file
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download ${REPO} ${FILENAME} --local-dir models/ --local-dir-use-symlinks False
        # Rename if needed
        if [ "$FILENAME" != "$OUTPUT_NAME" ]; then
            echo "Renaming to $OUTPUT_NAME..."
            mv "models/${FILENAME}" "$OUTPUT_PATH"
        fi
    else
        echo "huggingface-cli not found. Installing now..."
        pip install huggingface_hub
        huggingface-cli download ${REPO} ${FILENAME} --local-dir models/ --local-dir-use-symlinks False
        # Rename if needed
        if [ "$FILENAME" != "$OUTPUT_NAME" ]; then
            echo "Renaming to $OUTPUT_NAME..."
            mv "models/${FILENAME}" "$OUTPUT_PATH"
        fi
    fi
    
    echo "Download complete! Model saved to: $OUTPUT_PATH"
    echo ""
    echo "You can now use this model by selecting 'Llama 3 ${MODEL_SIZE}B (Local)' in the UI"
    echo "and entering the path: $OUTPUT_PATH in the Model Path field."
    echo ""
    echo "For best performance, the model is configured to utilize both of your RTX A4000 GPUs."
}

# Menu for model selection
echo "Please select which model to download:"
echo "1) Llama 3 8B (Recommended for faster responses)"
echo "2) Llama 3 70B (Higher quality, requires more VRAM)"
echo ""
read -p "Enter your choice (1 or 2): " MODEL_CHOICE

# Menu for quantization level
echo ""
echo "Select quantization level:"
echo "1) Q4_K_M (Fastest, smallest file size, ~4GB VRAM per model)"
echo "2) Q5_K_M (Balanced speed and quality, ~5GB VRAM per model)"
echo "3) Q8_0 (Highest quality, slower, ~8GB VRAM per model)"
echo ""
read -p "Enter your choice (1, 2, or 3): " QUANT_CHOICE

# Process selection
MODEL_SIZE="8"
if [ "$MODEL_CHOICE" = "2" ]; then
    MODEL_SIZE="70"
fi

QUANT_LEVEL="Q4_K_M"
if [ "$QUANT_CHOICE" = "2" ]; then
    QUANT_LEVEL="Q5_K_M"
elif [ "$QUANT_CHOICE" = "3" ]; then
    QUANT_LEVEL="Q8_0"
fi

# Set output file name
OUTPUT_NAME="llama-3-${MODEL_SIZE}b.gguf"

# Download the model
download_model $MODEL_SIZE $QUANT_LEVEL $OUTPUT_NAME

echo "Done! You can now run your app with optimized settings for your dual RTX A4000 GPUs."
echo "To use the model, select 'Llama 3 ${MODEL_SIZE}B (Local)' in the UI." 