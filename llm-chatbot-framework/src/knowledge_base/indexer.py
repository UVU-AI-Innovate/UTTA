"""
Document Indexer Module

This module provides functionality for indexing and retrieving educational documents.
It supports various document formats and maintains a searchable index of document content.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

class DocumentIndexer:
    def __init__(self, index_path: Optional[str] = None):
        """Initialize the document indexer.
        
        Args:
            index_path: Optional path to store document index.
                       Defaults to 'data/document_index' in project root.
        """
        if index_path is None:
            index_path = Path(__file__).parent.parent.parent / "data" / "document_index"
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Load existing index
        self.document_index = self._load_index()

    def index_document(self, document: Dict[str, str]) -> str:
        """Index a new document.
        
        Args:
            document: Dictionary containing document information
                     (file_path, content)
        
        Returns:
            doc_id: Unique identifier for the indexed document
        """
        # Generate unique document ID
        content_hash = hashlib.sha256(document['content'].encode()).hexdigest()[:12]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_id = f"{timestamp}_{content_hash}"
        
        # Save document content
        doc_path = self.index_path / f"{doc_id}.json"
        doc_data = {
            'file_path': document['file_path'],
            'content': document['content'],
            'indexed_at': timestamp
        }
        
        with open(doc_path, 'w') as f:
            json.dump(doc_data, f, indent=2)
        
        # Update index
        self.document_index[doc_id] = {
            'path': str(doc_path),
            'file_path': document['file_path'],
            'indexed_at': timestamp
        }
        
        self._save_index()
        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, str]]:
        """Retrieve a document by its ID.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Document dictionary if found, None otherwise
        """
        if doc_id not in self.document_index:
            return None
        
        doc_path = self.document_index[doc_id]['path']
        with open(doc_path, 'r') as f:
            return json.load(f)

    def list_documents(self) -> List[Dict[str, str]]:
        """List all indexed documents.
        
        Returns:
            List of document metadata dictionaries
        """
        return [
            {
                'id': doc_id,
                'file_path': metadata['file_path'],
                'indexed_at': metadata['indexed_at']
            }
            for doc_id, metadata in self.document_index.items()
        ]

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            bool: True if document was removed, False if not found
        """
        if doc_id not in self.document_index:
            return False
        
        # Remove document file
        doc_path = Path(self.document_index[doc_id]['path'])
        if doc_path.exists():
            doc_path.unlink()
        
        # Remove from index
        del self.document_index[doc_id]
        self._save_index()
        
        return True

    def _load_index(self) -> Dict[str, Dict[str, str]]:
        """Load the document index from disk."""
        index_file = self.index_path / "document_index.json"
        if index_file.exists():
            with open(index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save the document index to disk."""
        index_file = self.index_path / "document_index.json"
        with open(index_file, 'w') as f:
            json.dump(self.document_index, f, indent=2) 