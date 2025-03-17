#!/bin/bash
# Script to run all fine-tuning examples in sequence

echo "==================================================="
echo "Running Fine-Tuning Examples for Educational QA"
echo "==================================================="

# Ensure we're in the examples directory
cd "$(dirname "$0")"

# Run DSPy example
echo -e "\n\n==================================================="
echo "RUNNING DSPY PROMPT OPTIMIZATION EXAMPLE"
echo "==================================================="
python dspy_example.py

# Run OpenAI example
echo -e "\n\n==================================================="
echo "RUNNING OPENAI FINE-TUNING EXAMPLE"
echo "==================================================="
python openai_finetune.py

# Run HuggingFace example
echo -e "\n\n==================================================="
echo "RUNNING HUGGINGFACE LORA FINE-TUNING EXAMPLE"
echo "==================================================="
python huggingface_lora.py

echo -e "\n\n==================================================="
echo "ALL EXAMPLES COMPLETED"
echo "See README.md for explanation of the different approaches"
echo "===================================================" 