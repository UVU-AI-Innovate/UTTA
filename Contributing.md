# Contributing to UTTA

Thank you for your interest in contributing to the Universal Teaching and Training Assistant (UTTA) project! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read our [Code of Conduct](https://github.com/UVU-AI-Innovate/UTTA/wiki/Code-of-Conduct) before contributing. We expect all contributors to adhere to these guidelines to ensure a positive and inclusive community.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on our [GitHub Issues](https://github.com/UVU-AI-Innovate/UTTA/issues) page with the following information:

- A clear, descriptive title
- A detailed description of the bug
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Environment information (OS, Python version, etc.)

### Suggesting Features

We welcome feature suggestions! Please create an issue on our [GitHub Issues](https://github.com/UVU-AI-Innovate/UTTA/issues) page with the following information:

- A clear, descriptive title
- A detailed description of the feature
- Why this feature would be beneficial
- Any implementation ideas you have

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/UVU-AI-Innovate/UTTA.git
cd UTTA
```

2. Create and activate the conda environment:
```bash
conda create -n utta python=3.10
conda activate utta
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write unit tests for new functionality

## Testing

Run tests using the test runner script:

```bash
./run_tests.py
```

## Documentation

- Update documentation for any changes you make
- Follow the existing documentation style
- Add examples for new features

## Commit Messages

- Use clear, descriptive commit messages
- Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `style:` for formatting changes
  - `refactor:` for code refactoring
  - `test:` for adding or modifying tests
  - `chore:` for maintenance tasks

## Review Process

- All pull requests will be reviewed by at least one maintainer
- Feedback will be provided on the pull request
- Changes may be requested before merging

## License

By contributing to UTTA, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

Thank you for contributing to UTTA! 