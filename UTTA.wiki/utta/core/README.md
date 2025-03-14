# UTTA Core Module

This module provides the core functionality for the UTTA framework, including document processing and vector database management.

## Components

### DocumentProcessor

`DocumentProcessor` handles text extraction, parsing, and chunking for various document formats:

- Text documents (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- Web content (HTML)
- EPUB files
- Markdown files

The document processor includes advanced text segmentation algorithms to ensure that chunks maintain semantic cohesion.

### VectorDatabase

`VectorDatabase` provides semantic search capabilities by:

- Converting text into embeddings
- Storing embeddings in a FAISS vector database
- Enabling efficient semantic search
- Supporting GPU acceleration
- Handling batched operations

## Usage

```python
from utta.core import DocumentProcessor, VectorDatabase

# Initialize
processor = DocumentProcessor()
vector_db = VectorDatabase()

# Process documents
chunks = processor.process_document("path/to/document.pdf")

# Add to vector database
vector_db.add_chunks(chunks)

# Search for relevant information
results = vector_db.search("What is differentiated instruction?", top_k=3)
```

## Configuration

Both components can be configured through environment variables or by passing parameters to their constructors. 