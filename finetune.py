#!/usr/bin/env python3
"""
UTTA Fine-Tuning CLI Runner

This is a simple script to run the UTTA fine-tuning CLI from the root directory.
"""

import os
import sys
import argparse
from tools.fine_tuning_cli import main

if __name__ == "__main__":
    # Pass all arguments to the fine-tuning CLI
    sys.argv[0] = "tools/fine_tuning_cli.py"
    main()
