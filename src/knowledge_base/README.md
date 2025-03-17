# Knowledge Base Module

This directory contains components for managing and retrieving educational documents used as context for generating responses.

## Overview

The Knowledge Base module provides functionality for indexing, storing, and retrieving educational materials. It serves as the foundation for context-aware teaching by making relevant information accessible to the LLM.

## Components

### DocumentIndexer

The `DocumentIndexer` class (`document_indexer.py`) provides:

- Document ingestion and processing
- Vector-based semantic search
- Filtering by metadata (subject, grade level, etc.)
- Persistence of document embeddings

## Features

- **Semantic Search**: Find documents based on meaning rather than keyword matching
- **Fallback Modes**: Operates with reduced functionality when dependencies are unavailable
- **Metadata Filtering**: Retrieve documents based on subject, grade level, or other attributes
- **Vectorization**: Converts documents into embeddings for efficient similarity search

## Usage

```python
from src.knowledge_base.document_indexer import DocumentIndexer

# Initialize the indexer
indexer = DocumentIndexer()

# Add a document
doc_id = indexer.add_document(
    content="Photosynthesis is the process by which plants use sunlight to produce energy.",
    metadata={
        "title": "Introduction to Photosynthesis",
        "subject": "Science",
        "grade_level": 5
    }
)

# Search for relevant documents
results = indexer.search(
    query="How do plants make food?",
    filters={"subject": "Science", "grade_level": 5},
    limit=3
)

# Process search results
for result in results:
    print(f"Relevance: {result.score}")
    print(f"Content: {result.content}")
    print(f"Metadata: {result.metadata}")
```

## Implementation Details

The indexer uses sentence-transformers for document embedding and similarity search. When these dependencies are unavailable, it falls back to simpler text-matching algorithms.

Document embeddings and metadata are cached to disk in the `data/index/` directory for persistence across application restarts. 