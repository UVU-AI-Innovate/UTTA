#!/usr/bin/env python
"""
DSPy Optimization Example for Teacher-Student Dialogue

This example demonstrates prompt optimization using DSPy for creating a teacher-student dialogue chatbot.
Key characteristics of this approach:
- No model weight updates (only prompt optimization)
- Very little training data needed (as few as 10-20 dialogue examples)
- No special infrastructure required
- Models multi-turn educational conversations rather than simple Q&A
- Incorporates pedagogical techniques like Socratic questioning and scaffolding
"""
import dspy_ai as dspy
import json
import random
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Load environment variables from .env file 