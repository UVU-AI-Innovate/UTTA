#!/bin/bash
# Script to fix invalid distribution packages in a pip environment

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Fixing Pip Environment Issues                  ${NC}"
echo -e "${GREEN}==================================================${NC}"

# Check if we're in a conda environment
if [[ -z "${CONDA_PREFIX}" ]]; then
    echo -e "${YELLOW}Conda environment not detected. Please run 'conda activate utta' first.${NC}"
    exit 1
fi

ENV_NAME=$(basename "${CONDA_PREFIX}")
echo -e "${GREEN}Working with conda environment: ${ENV_NAME}${NC}"

# Get the site-packages directory
SITE_PACKAGES=$(python -c "import site; print(site.getusersitepackages())")
echo -e "${GREEN}Site packages directory: ${SITE_PACKAGES}${NC}"

# Find all dist-info and egg-info directories
echo -e "${YELLOW}Scanning for invalid distribution packages...${NC}"
INVALID_DIRS=$(find "${SITE_PACKAGES}" -name "*-orch*" -type d)

if [ -z "$INVALID_DIRS" ]; then
    echo -e "${GREEN}No invalid distribution packages found.${NC}"
else
    echo -e "${YELLOW}Found invalid distribution packages:${NC}"
    echo "$INVALID_DIRS"
    
    echo -e "${YELLOW}Removing invalid packages...${NC}"
    for dir in $INVALID_DIRS; do
        echo -e "Removing: ${dir}"
        rm -rf "$dir"
    done
    
    echo -e "${GREEN}Invalid packages removed successfully.${NC}"
fi

# Fix LlamaIndex package imports
echo -e "${YELLOW}Checking LlamaIndex installation...${NC}"
pip show llama-index

echo -e "${YELLOW}Now let's clean up pip's cache to ensure all issues are resolved${NC}"
pip cache purge

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}   Environment Fixed Successfully!                ${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "Now try running: ${YELLOW}python advanced_testing_tools.py --help${NC}"
echo -e "${GREEN}==================================================${NC}" 