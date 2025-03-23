#!/bin/bash
# Script to run the simple overview of all fine-tuning examples

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
  
  echo -e "\n${GREEN}Simple example completed successfully!${NC}"
  echo -e "${YELLOW}This example provided an overview of all three fine-tuning approaches.${NC}"
  echo -e "${YELLOW}To run the full examples, you'll need to set up the required dependencies and API keys.${NC}"
}

# Show information about the actual examples
show_example_info() {
  echo -e "\n\n${BLUE}==================================================="
  echo -e "INFORMATION ABOUT FULL EXAMPLES"
  echo -e "===================================================${NC}"
  
  echo -e "\n${YELLOW}The full examples require additional setup:${NC}"
  echo -e "1. ${GREEN}DSPy Example (dspy_example.py):${NC}"
  echo -e "   - Requires DSPy library matching the example version"
  echo -e "   - Needs OpenAI API key in .env file"
  
  echo -e "\n2. ${GREEN}OpenAI Example (openai_finetune.py):${NC}"
  echo -e "   - Requires OpenAI library (compatible with the script)"
  echo -e "   - Needs OpenAI API key in .env file"
  echo -e "   - Running actual fine-tuning would incur costs"
  
  echo -e "\n3. ${GREEN}HuggingFace Example (huggingface_lora.py):${NC}"
  echo -e "   - Requires transformers, peft libraries"
  echo -e "   - Needs GPU for actual training"
  
  echo -e "\n${YELLOW}To run these examples after setting up dependencies:${NC}"
  echo -e "python dspy_example.py"
  echo -e "python openai_finetune.py"
  echo -e "python huggingface_lora.py"
}

# Run the overview example and show info
check_dependencies
run_simple_example
show_example_info

echo -e "\n\n${BLUE}==================================================="
echo -e "EXAMPLES OVERVIEW COMPLETED"
echo -e "See README.md for more detailed explanations"
echo -e "===================================================${NC}" 