# Tests

This directory contains test modules for verifying the functionality of the UTTA system.

## Test Organization

- **Unit Tests**: Individual component tests located in `unit/`
- **Integration Tests**: End-to-end tests located in `integration/`

## Unit Tests

Unit tests verify the functionality of individual components in isolation:

### Components Tests (`unit/test_components.py`)
- Tests for LLM interface
- Tests for document indexer
- Tests for metrics evaluator

### Web App Tests (`unit/test_web_app.py`)
- Tests for web interface components
- Tests for Streamlit session state management
- Tests for UI rendering

## Integration Tests

Integration tests verify that multiple components work together correctly:

### System Integration (`integration/test_integration.py`)
- End-to-end tests of the complete system
- Tests for knowledge base integration with LLM
- Tests for fine-tuning with evaluation metrics
- Tests for document indexing and retrieval in context

## Running Tests

From the project root:

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run specific test files
pytest tests/unit/test_components.py
```

## Test Data

Test data is stored in the `data/test/` directory and includes:
- Sample documents for testing indexing
- Example prompts and responses
- Test training data for fine-tuning

## Writing New Tests

When writing new tests:
1. Place unit tests in the appropriate file in `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use pytest fixtures for common setup
4. Mock external dependencies when appropriate
5. Include appropriate assertions and error messages 