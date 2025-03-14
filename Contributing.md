# Contributing to UTTA

Thank you for your interest in contributing to the Universal Teaching and Training Assistant (UTTA) project! This guide will help you get started with contributing to the project.

## Code of Conduct

Please read our [Code of Conduct](Code-of-Conduct) before contributing. We expect all contributors to adhere to these guidelines to ensure a positive and inclusive community.

## Table of Contents
- [How to Contribute](#how-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Coding Standards](#coding-standards)
- [Community](#community)

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

## Getting Started

1. **Fork the Repository**
   * Visit [UTTA on GitHub](https://github.com/UVU-AI-Innovate/UTTA)
   * Click the "Fork" button in the upper right corner

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/UTTA.git
   cd UTTA
   ```

3. **Set Up Remote**
   ```bash
   git remote add upstream https://github.com/UVU-AI-Innovate/UTTA.git
   ```

4. **Set Up Development Environment**
   ```bash
   conda create -n utta python=3.10
   conda activate utta
   pip install -e .
   ```

## Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
   Use prefixes like:
   * `feature/` for new features
   * `fix/` for bug fixes
   * `docs/` for documentation
   * `test/` for adding or updating tests

2. **Make Your Changes**
   * Write clean, well-documented code
   * Follow the existing code style and conventions
   * Keep commits focused and logical

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: Add new feature X"
   ```
   We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

4. **Stay Updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## Pull Request Process

1. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request**
   * Go to the [UTTA repository](https://github.com/UVU-AI-Innovate/UTTA)
   * Click "New Pull Request"
   * Select "compare across forks"
   * Select your fork and branch
   * Add a clear title and description

3. **PR Guidelines**
   * Link to any relevant issues
   * Include screenshots or examples if applicable
   * Ensure all tests pass
   * Update documentation as needed
   * Respond to review comments promptly

4. **After Approval**
   * Your PR will be merged by a maintainer
   * You can delete your branch after it's merged

## Testing

Run tests using the test runner script:

```bash
./run_tests.py
```

All new features should include appropriate tests, and bug fixes should include a test that would have caught the issue.

## Documentation

- Update documentation for any changes you make
- Follow the existing documentation style
- Add examples for new features

## Coding Standards

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write unit tests for new functionality

## Community

* **GitHub Discussions**: Ask questions and share ideas
* **GitHub Issues**: Report bugs and request features
* **Pull Requests**: Review and discuss code changes

## License

By contributing to UTTA, you agree that your contributions will be licensed under the project's MIT License. 