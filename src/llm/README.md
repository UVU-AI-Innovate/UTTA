# LLM Module

This directory contains components for interacting with language models to generate teaching responses.

## Overview

The LLM module provides an abstraction layer for working with various Large Language Models through a unified interface. The current implementation focuses on DSPy-based interfaces for educational response generation.

## Structure

- `dspy/` - Components for DSPy-based language model interfaces
  - `handler.py` - Contains the `EnhancedDSPyLLMInterface` class
  - `signatures.py` - DSPy signatures for educational response generation
  - `prompts.py` - System prompts and templates for teacher responses

## Components

### EnhancedDSPyLLMInterface

The `EnhancedDSPyLLMInterface` class provides:

- A unified API for generating educational responses
- Context-aware response generation using DSPy modules
- Graceful degradation when APIs are unavailable
- Customizable teaching style and grade level adaptation

## Usage

```python
from src.llm.dspy.handler import EnhancedDSPyLLMInterface

# Initialize the LLM interface
llm = EnhancedDSPyLLMInterface()

# Generate a response
response = llm.generate_response(
    prompt="Explain photosynthesis to a 5th grader",
    context=["Photosynthesis is the process by which plants use sunlight..."],
    metadata={
        "grade_level": 5, 
        "subject": "Science", 
        "learning_style": "visual"
    }
)
```

## DSPy Integration

The LLM module leverages DSPy for optimized prompt engineering and module composition. The DSPy signatures in `signatures.py` define structured inputs and outputs for teaching tasks, while the handler manages configuration, caching, and error handling.

When the primary API is unavailable, the interface falls back to simplified templates or cached responses to ensure the application remains functional. 