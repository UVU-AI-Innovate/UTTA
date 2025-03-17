# Data Directory

This directory contains data files used by the UTTA system, including knowledge base documents, training data, and cached resources.

## Directory Structure

```
data/
├── cache/        # Cache storage for embeddings and API responses
├── index/        # Document index storage
├── models/       # Trained and fine-tuned models
├── test/         # Test datasets and examples
└── uploads/      # User-uploaded documents
```

## Data Types

### Knowledge Base Documents

Knowledge base documents are stored in the `index/` directory and include:
- Educational content in various formats
- Teaching resources organized by subject and grade level
- Reference materials for model responses

### Cache

The `cache/` directory stores:
- Embeddings for document retrieval
- Cached LLM responses for performance optimization
- Temporary processing artifacts

### Models

The `models/` directory stores:
- Fine-tuned DSPy models
- Model configuration files
- Optimization checkpoints

### Test Data

The `test/` directory contains:
- Test examples for unit and integration tests
- Evaluation datasets
- Benchmark datasets for comparing model performance

### User Uploads

The `uploads/` directory contains:
- User-uploaded documents via the web interface
- Temporary storage for document processing

## Usage

Data in this directory is managed automatically by the application. Most users should not need to manually modify these files. 