# Contributing to UTTA

Thank you for your interest in contributing to the Utah Teacher Training Assistant (UTTA) project! This guide will help you get started with contributing to our project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Making Contributions](#making-contributions)
5. [Pull Request Process](#pull-request-process)
6. [Style Guidelines](#style-guidelines)
7. [Testing](#testing)
8. [Documentation](#documentation)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to adhere to our [Code of Conduct](Code-of-Conduct).

## Getting Started

### Prerequisites

1. Complete the [Environment Setup](Environment-Setup)
2. Familiarize yourself with:
   - Python 3.10+
   - Git
   - Virtual environments (conda)
   - Testing frameworks (pytest)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/UTTA.git
   cd UTTA
   ```
3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/UVU-AI-Innovate/UTTA.git
   ```

## Development Setup

### 1. Create Development Environment

```bash
# Create and activate conda environment
conda create -n utta-dev python=3.10
conda activate utta-dev

# Install development dependencies
pip install -r requirements-dev.txt
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install
```

### 3. Configure Development Tools

```bash
# Install development tools
pip install black isort mypy pytest

# Set up git configuration
git config --local core.autocrlf input
git config --local core.whitespace trailing-space,space-before-tab
```

## Making Contributions

### 1. Choose an Issue

- Check [open issues](https://github.com/UVU-AI-Innovate/UTTA/issues)
- Comment on the issue you'd like to work on
- Wait for assignment or approval

### 2. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

1. Write your code
2. Add tests
3. Update documentation
4. Run local tests:
   ```bash
   pytest tests/
   ```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "type: brief description

Detailed description of changes made.

Fixes #123"
```

## Pull Request Process

### 1. Prepare PR

1. Update your branch:
   ```bash
   git checkout main
   git pull upstream main
   git checkout feature/your-feature-name
   git rebase main
   ```

2. Push changes:
   ```bash
   git push origin feature/your-feature-name
   ```

### 2. Create PR

1. Go to GitHub
2. Create new Pull Request
3. Fill out PR template
4. Link related issues
5. Request review

### 3. Review Process

1. Address reviewer comments
2. Update PR as needed
3. Ensure CI passes
4. Get approval

## Style Guidelines

### Python Code Style

```python
# Follow PEP 8 guidelines
def calculate_score(student_response: str, reference: str) -> float:
    """
    Calculate similarity score between student response and reference.

    Args:
        student_response: The student's answer text
        reference: The reference answer text

    Returns:
        float: Similarity score between 0 and 1
    """
    # Implementation
    pass
```

### Docstring Format

```python
class TeachingAssistant:
    """
    A class representing an AI teaching assistant.

    Attributes:
        name (str): The name of the teaching assistant
        model (str): The underlying language model
        config (Dict): Configuration parameters

    Example:
        >>> assistant = TeachingAssistant("Math Tutor")
        >>> response = assistant.answer("What is calculus?")
    """

    def __init__(self, name: str):
        self.name = name
```

### Import Order

```python
# Standard library
import os
import sys
from typing import Dict, List

# Third-party packages
import numpy as np
import torch
from transformers import AutoModel

# Local imports
from utta.core import TeachingAssistant
from utta.utils import load_config
```

## Testing

### 1. Writing Tests

```python
# tests/test_assistant.py
import pytest
from utta.core import TeachingAssistant

def test_assistant_initialization():
    """Test teaching assistant initialization."""
    assistant = TeachingAssistant("Test Assistant")
    assert assistant.name == "Test Assistant"
    assert assistant.is_ready()

@pytest.mark.parametrize("question,expected", [
    ("What is Python?", True),
    ("", False),
    (None, False)
])
def test_question_validation(question, expected):
    """Test question validation with various inputs."""
    assistant = TeachingAssistant("Test Assistant")
    assert assistant.validate_question(question) == expected
```

### 2. Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_assistant.py

# Run with coverage
pytest --cov=utta tests/

# Run with verbose output
pytest -v tests/
```

### 3. Test Coverage

```bash
# Generate coverage report
coverage run -m pytest
coverage report
coverage html
```

## Documentation

### 1. Code Documentation

- Use clear variable names
- Write descriptive docstrings
- Include type hints
- Add inline comments for complex logic

### 2. Wiki Documentation

- Keep pages focused
- Include examples
- Update navigation
- Link related pages

### 3. API Documentation

```python
def evaluate_response(
    response: str,
    criteria: List[str],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Evaluate a teaching assistant's response based on given criteria.

    Args:
        response: The response to evaluate
        criteria: List of evaluation criteria
        weights: Optional weights for each criterion

    Returns:
        Dictionary containing scores for each criterion

    Raises:
        ValueError: If criteria list is empty
        TypeError: If weights are provided but don't match criteria
    """
    if not criteria:
        raise ValueError("Criteria list cannot be empty")
    
    if weights and set(weights.keys()) != set(criteria):
        raise TypeError("Weights must match criteria exactly")
    
    # Implementation
    pass
```

## Best Practices

### 1. Code Quality

- Write modular code
- Follow SOLID principles
- Keep functions focused
- Use meaningful names
- Add error handling

### 2. Performance

- Profile code when needed
- Optimize critical paths
- Use appropriate data structures
- Consider memory usage
- Add caching where beneficial

### 3. Security

- Never commit secrets
- Validate inputs
- Use secure dependencies
- Follow security guidelines
- Report vulnerabilities

## Getting Help

1. Check the [FAQ](FAQ)
2. Search existing issues
3. Ask in discussions
4. Join our community channels

## Next Steps

1. Pick an [open issue](https://github.com/UVU-AI-Innovate/UTTA/issues)
2. Read the [DSPy Tutorial](DSPy-Tutorial)
3. Explore the [Dataset Preparation](Dataset-Preparation) guide 