# UTTA Tests

This directory contains tests for the UTTA framework.

## Running Tests

You can run all tests using the test runner script:

```bash
# From the UTTA directory
./run_tests.py
```

Or run individual test files:

```bash
# Run a specific test file
python tests/test_chatbot.py
```

## Test Structure

- **Core Tests**: Tests for core functionality (Vector DB, Document Processor)
- **Chatbot Tests**: Tests for LLM interfaces and chatbot functionality
- **Fine-Tuning Tests**: Tests for fine-tuning and optimization components

## Writing New Tests

When adding new functionality to UTTA, please include appropriate tests:

1. Create a new file named `test_[component].py`
2. Use the `unittest` framework for structured tests
3. Include both unit tests and integration tests where appropriate 