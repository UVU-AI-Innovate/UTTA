# Contributing to UTTA

Thank you for your interest in contributing to the UTTA project! This guide will help you get started with contributing to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

Please read our [Code of Conduct](Code-of-Conduct) before contributing to the project. By participating, you are expected to uphold this code.

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
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Install development dependencies
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

1. **Run Tests**
   ```bash
   pytest
   ```

2. **Adding Tests**
   * All new features should include appropriate tests
   * Bug fixes should include a test that would have caught the issue

## Documentation

1. **Code Documentation**
   * Use docstrings for functions, classes, and modules
   * Follow [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

2. **Wiki Documentation**
   * Update relevant wiki pages for user-facing changes
   * Create new wiki pages for major features

## Community

* **GitHub Discussions**: Ask questions and share ideas
* **GitHub Issues**: Report bugs and request features
* **Pull Requests**: Review and discuss code changes

Thank you for contributing to UTTA! 