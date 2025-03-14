# UTTA Chatbot Module

This module provides interfaces for interacting with various Language Models (LLMs) in the UTTA framework.

## Components

### DSPyLLMHandler

`DSPyLLMHandler` is the primary interface for interacting with language models through DSPy:

- Provides a simple interface for generating responses
- Handles context management
- Manages API keys and rate limits
- Supports various language models (OpenAI, Anthropic, etc.)
- Includes error handling and retries

### LLMInterface

`LLMInterface` provides a lower-level interface for direct LLM access:

- Manages model initialization and configuration
- Handles prompt construction
- Processes responses
- Provides utility methods for common operations

## Usage

```python
from utta.chatbot import DSPyLLMHandler

# Initialize with your preferred model
llm = DSPyLLMHandler(model_name="gpt-3.5-turbo")

# Generate a response
response = llm.generate("Explain the concept of differentiated instruction.")

print(response)
```

## Advanced Features

The chatbot module provides several advanced capabilities:

- **Streaming**: Real-time token-by-token response streaming
- **Teaching Analysis**: Evaluation of teaching responses
- **Student Simulation**: Generation of realistic student reactions
- **Specialized Prompting**: Education-specific prompt techniques

## Configuration

The language model handlers can be configured through:

- Constructor parameters
- Environment variables (for API keys)
- Configuration files 