from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="utta",
    version="0.1.0",
    author="UTTA Contributors",
    author_email="example@email.com",
    description="Utah Teacher Training Assistant - A framework for LLM-based teaching assistants",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UVU-AI-Innovate/UTTA",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "utta-finetune=tools.fine_tuning_cli:main",
        ],
    },
) 