# UTTA Technical Architecture

This page provides a comprehensive overview of the UTTA framework's technical architecture, components, and technologies used.

## System Architecture

UTTA is designed with a modular architecture that allows for flexibility and extensibility. The system consists of several key components:

```
UTTA/
├── utta/                    # Main package
│   ├── core/                # Core functionality (Vector DB, Document Processor)
│   ├── chatbot/             # Chatbot functionality and LLM handlers
│   ├── fine_tuning/         # Fine-tuning implementations
│   ├── evaluation/          # Evaluation tools and metrics
│   └── utils/               # Utility functions and helpers
├── examples/                # Usage examples
│   ├── simple_chatbot.py    # Basic chatbot example
│   └── fine_tuning/         # Fine-tuning examples
├── tools/                   # CLI tools
│   └── fine_tuning_cli.py   # Fine-tuning CLI tool
├── evaluation_results/      # Evaluation outputs
└── benchmark_results/       # Benchmark data
```

## Core Components

### 1. Core Module (`utta.core`)

The core module contains fundamental functionality of UTTA:

- **DocumentProcessor**: Handles parsing, chunking, and processing of various document formats
- **VectorDatabase**: Manages embeddings and semantic search capabilities

### 2. Chatbot Module (`utta.chatbot`)

The chatbot module handles interactions with language models:

- **LLMHandler**: Base interface for different LLM implementations
- **DSPyLLMHandler**: Implementation for DSPy-powered models

### 3. Fine-Tuning Module (`utta.fine_tuning`)

The fine-tuning module contains implementations for different fine-tuning approaches:

- **DSPyOptimizer**: Tools for prompt optimization using DSPy
- **OpenAITuner**: Implementation for fine-tuning OpenAI models
- **HuggingFaceTuner**: Implementation for fine-tuning open-source models

### 4. Command-Line Tools

UTTA provides a robust command-line interface for all core functionalities:

- **Simple Chatbot**: Terminal-based interactive chatbot
- **Fine-Tuning CLI**: Command-line tool for managing fine-tuning jobs
- **Evaluation Tools**: Terminal-based evaluation and benchmarking utilities

## Technology Stack

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Programming Language** | Python 3.10+ |
| **Package Management** | Conda, pip |
| **Version Control** | Git |
| **Documentation** | Markdown, GitHub Wiki |

### LLM and AI Technologies

| Category | Technologies |
|----------|-------------|
| **LLM Frameworks** | DSPy, OpenAI API, HuggingFace Transformers |
| **Embedding Models** | Sentence Transformers, OpenAI Embeddings |
| **Vector Databases** | FAISS |
| **Fine-tuning** | PEFT, LoRA, QLoRA, Accelerate |

### Data Processing

| Category | Technologies |
|----------|-------------|
| **Data Manipulation** | NumPy, Pandas |
| **Text Processing** | NLTK, BeautifulSoup4 |
| **Document Parsing** | PyPDF2, pdfplumber, python-docx, ebooklib |
| **Data Formats** | JSON, JSONL, CSV, YAML |

### Terminal Interface

| Category | Technologies |
|----------|-------------|
| **Command Line** | argparse, Python CLI |
| **Console Output** | Python logging, colorama |
| **Visualization** | Matplotlib, Seaborn, Plotly (for terminal-friendly outputs) |

### Evaluation and Metrics

| Category | Technologies |
|----------|-------------|
| **Metrics Libraries** | ROUGE, BERT-Score, SacreBLEU |
| **ML Evaluation** | Scikit-learn |
| **Logging** | Python logging, Loguru |

## Data Flow

1. **Input Processing**:
   - Command-line user queries or documents are processed by the Document Processor
   - Text is chunked and prepared for embedding

2. **Knowledge Retrieval**:
   - Embeddings are generated for the input
   - Vector database performs semantic search to find relevant information
   - Retrieved context is formatted for the LLM

3. **Response Generation**:
   - Prompt is constructed with retrieved context
   - LLM generates a response based on the prompt
   - Response is post-processed and returned to the user via the terminal

4. **Evaluation** (optional):
   - Responses can be evaluated against reference answers
   - Metrics are calculated and stored for analysis

## Command-Line Tool Usage

### Simple Chatbot

The terminal-based chatbot can be run using:

```bash
python examples/simple_chatbot.py
```

This launches an interactive session where you can ask questions and receive AI-generated responses.

### Fine-Tuning CLI

The fine-tuning CLI tool provides commands for various operations:

```bash
# Using the module directly
python tools/fine_tuning_cli.py --help

# Or using the installed entry point
utta-finetune --help

# DSPy Optimization
utta-finetune dspy optimize --examples examples.json --module teacher_responder

# OpenAI Fine-tuning
utta-finetune openai start --file training_data.jsonl --model gpt-3.5-turbo

# HuggingFace Fine-tuning
utta-finetune huggingface fine-tune --file training_data.jsonl --model microsoft/phi-2 --use-lora
```

## Integration Points

UTTA is designed to be integrated with various systems:

- **LLM Providers**: OpenAI, Anthropic, HuggingFace, etc.
- **Vector Databases**: FAISS, Pinecone, Weaviate, etc.
- **Educational Systems**: Can integrate with LMS platforms via API endpoints
- **Other Command-Line Tools**: Can be piped with other Unix/Linux utilities

## Deployment Options

UTTA supports multiple deployment scenarios:

- **Local Development**: Run locally for development and testing
- **Server Deployment**: Deploy on a dedicated server with CLI access
- **Cloud Deployment**: Deploy on cloud platforms as a CLI tool (AWS, GCP, Azure)
- **Containerization**: Docker support for containerized deployment

## Performance Considerations

- **Memory Usage**: Vector databases and LLMs can be memory-intensive
- **Latency**: Response time depends on LLM provider and retrieval speed
- **Resource Management**: Important for terminal applications on limited hardware
- **Caching**: Response caching for improved performance

## Security Considerations

- **API Keys**: Secure storage of LLM provider API keys
- **Data Privacy**: Options for local deployment to maintain data privacy
- **User Data**: Careful handling of user queries and responses
- **Content Filtering**: Options for filtering inappropriate content 