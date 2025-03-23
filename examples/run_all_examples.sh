#!/bin/bash
# Script to run all fine-tuning examples in sequence

# Terminal colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================="
echo -e "   🧠 Fine-Tuning Examples for Educational QA"
echo -e "===================================================${NC}"

# Ensure we're in the examples directory
cd "$(dirname "$0")"

# Check for necessary dependencies
check_dependencies() {
  echo -e "\n${YELLOW}Checking dependencies...${NC}"
  
  if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found. Please install Python 3.8+${NC}"
    exit 1
  fi
  
  if [ ! -f "../.env" ]; then
    echo -e "${YELLOW}Warning: No .env file found in the parent directory.${NC}"
    echo -e "         You might need API keys for some examples."
    echo -e "         Create a .env file with OPENAI_API_KEY=your_key"
  else
    echo -e "${GREEN}Found .env file for environment variables${NC}"
  fi
  
  echo -e "${GREEN}Dependencies check complete${NC}"
}

# Simple example that just shows concepts
run_simple_example() {
  echo -e "\n\n${BLUE}==================================================="
  echo -e "RUNNING SIMPLE OVERVIEW EXAMPLE (NO API KEYS NEEDED)"
  echo -e "===================================================${NC}"
  python simple_example.py
  
  echo -e "\n${YELLOW}Press Enter to continue to the next example...${NC}"
  read
}

# Run DSPy example
run_dspy_example() {
  echo -e "\n\n${BLUE}==================================================="
  echo -e "RUNNING DSPY PROMPT OPTIMIZATION EXAMPLE"
  echo -e "===================================================${NC}"
  echo -e "${YELLOW}Note: This example requires an OpenAI API key in your .env file${NC}"
  echo -e "${YELLOW}If you don't have an API key, it will fail gracefully${NC}"
  echo -e "${YELLOW}Run this example? (y/n)${NC}"
  read -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    python dspy_example.py
  else
    echo -e "${YELLOW}Skipping DSPy example${NC}"
  fi
}

# Run OpenAI example
run_openai_example() {
  echo -e "\n\n${BLUE}==================================================="
  echo -e "RUNNING OPENAI FINE-TUNING EXAMPLE"
  echo -e "===================================================${NC}"
  echo -e "${YELLOW}Note: This example requires an OpenAI API key in your .env file${NC}"
  echo -e "${YELLOW}Running this with actual fine-tuning would incur costs${NC}"
  echo -e "${YELLOW}Run this example? (y/n)${NC}"
  read -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    python openai_finetune.py
  else
    echo -e "${YELLOW}Skipping OpenAI example${NC}"
  fi
}

# Run HuggingFace example
run_huggingface_example() {
  echo -e "\n\n${BLUE}==================================================="
  echo -e "RUNNING HUGGINGFACE LORA FINE-TUNING EXAMPLE"
  echo -e "===================================================${NC}"
  echo -e "${YELLOW}Note: This example requires a GPU and HuggingFace libraries${NC}"
  echo -e "${YELLOW}It may take significant time and resources to run${NC}"
  echo -e "${YELLOW}Run this example? (y/n)${NC}"
  read -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    python huggingface_lora.py
  else
    echo -e "${YELLOW}Skipping HuggingFace example${NC}"
  fi
}

# Run all examples
check_dependencies
run_simple_example
run_dspy_example
run_openai_example
run_huggingface_example

echo -e "\n\n${BLUE}==================================================="
echo -e "ALL EXAMPLES COMPLETED"
echo -e "See README.md for explanation of the different approaches"
echo -e "===================================================${NC}" 