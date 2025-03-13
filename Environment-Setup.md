# Environment Setup

This guide will help you set up your development environment for the UTTA project.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or newer
- Git
- Conda (recommended) or another virtual environment manager

## Setting Up Your Environment

### Using Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/UVU-AI-Innovate/UTTA.git
   cd UTTA
   ```

2. Create a new conda environment:
   ```bash
   conda create -n utta python=3.10
   ```

3. Activate the environment:
   ```bash
   conda activate utta
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. For development, install additional dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

### Using venv

1. Clone the repository:
   ```bash
   git clone https://github.com/UVU-AI-Innovate/UTTA.git
   cd UTTA
   ```

2. Create a new virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. For development, install additional dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

## Setting Up API Keys

Some features of UTTA require access to external APIs. Follow these steps to set up your API keys:

1. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   ```

2. Replace the placeholder values with your actual API keys.

## Verifying Your Setup

To verify that your environment is set up correctly, run:

```bash
python -c "import utta; print('UTTA setup successful!')"
```

If you see "UTTA setup successful!" then your environment is correctly set up!

## Troubleshooting

If you encounter any issues during setup, please check the following:

- Make sure you have the correct Python version (3.10+)
- Ensure all dependencies are properly installed
- Check that your API keys are correctly set in the `.env` file

If you're still having issues, please open a GitHub issue with details about your problem. 