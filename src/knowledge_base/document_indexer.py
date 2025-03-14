"""Document indexer for managing teaching resources."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAVE_TRANSFORMERS = True
except ImportError:
    logger.warning("sentence-transformers not available, using fallback mode")
    HAVE_TRANSFORMERS = False

class DocumentIndexer:
    """Manages and indexes teaching resources."""
    
    def __init__(self, 
                index_dir: Optional[str] = None,
                embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the document indexer.
        
        Args:
            index_dir: Directory for storing indexed documents
            embedding_model: Model to use for document embeddings
        """
        try:
            # Set up storage
            self.index_dir = Path(index_dir or "data/index")
            os.makedirs(self.index_dir, exist_ok=True)
            
            # Initialize embedding model if available
            self.encoder = None
            if HAVE_TRANSFORMERS:
                try:
                    self.encoder = SentenceTransformer(embedding_model)
                except Exception as e:
                    logger.warning(f"Failed to load embedding model: {e}")
            
            # Load existing index
            self.document_index = self._load_index()
            self.embeddings = self._load_embeddings() if HAVE_TRANSFORMERS else {}
            
            logging.info("Initialized DocumentIndexer")
            
        except Exception as e:
            logger.error(f"Failed to initialize document indexer: {e}")
            raise
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load the document index from disk."""
        index_file = self.index_dir / "document_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                return {}
        return {}
    
    def _save_index(self):
        """Save the document index to disk."""
        index_file = self.index_dir / "document_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.document_index, f)
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")
    
    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load document embeddings from disk."""
        if not HAVE_TRANSFORMERS:
            return {}
            
        embedding_file = self.index_dir / "embeddings.npz"
        if embedding_file.exists():
            try:
                data = np.load(embedding_file)
                return {id: data[id] for id in data.files}
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                return {}
        return {}
    
    def _save_embeddings(self):
        """Save document embeddings to disk."""
        if not HAVE_TRANSFORMERS:
            return
            
        embedding_file = self.index_dir / "embeddings.npz"
        try:
            np.savez(embedding_file, **self.embeddings)
        except Exception as e:
            logger.warning(f"Failed to save embeddings: {e}")
    
    def add_document(self,
                   content: str,
                   metadata: Dict[str, Any]) -> str:
        """Add a document to the index.
        
        Args:
            content: The document content
            metadata: Document metadata (subject, grade level, etc.)
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            doc_id = hashlib.md5(content.encode()).hexdigest()
            
            # Create document embedding if available
            if HAVE_TRANSFORMERS and self.encoder:
                try:
                    embedding = self.encoder.encode(content)
                    self.embeddings[doc_id] = embedding
                except Exception as e:
                    logger.warning(f"Failed to create embedding: {e}")
            
            # Store document
            self.document_index[doc_id] = {
                "content": content,
                "metadata": metadata,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save to disk
            self._save_index()
            if HAVE_TRANSFORMERS:
                self._save_embeddings()
            
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def search(self,
             query: str,
             top_k: int = 5,
             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Metadata filters to apply
            
        Returns:
            List of matching documents with scores
        """
        try:
            scores = {}
            
            # Use embeddings if available
            if HAVE_TRANSFORMERS and self.encoder:
                try:
                    query_embedding = self.encoder.encode(query)
                    for doc_id, doc_embedding in self.embeddings.items():
                        similarity = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) *
                            np.linalg.norm(doc_embedding)
                        )
                        scores[doc_id] = similarity
                except Exception as e:
                    logger.warning(f"Failed to compute embeddings: {e}")
            
            # Fallback to simple text matching
            if not scores:
                query_lower = query.lower()
                for doc_id, doc_data in self.document_index.items():
                    content_lower = doc_data["content"].lower()
                    # Simple word overlap score
                    query_words = set(query_lower.split())
                    content_words = set(content_lower.split())
                    overlap = len(query_words & content_words)
                    scores[doc_id] = overlap / len(query_words) if query_words else 0
            
            # Apply filters
            if filters:
                scores = {
                    doc_id: score
                    for doc_id, score in scores.items()
                    if self._matches_filters(doc_id, filters)
                }
            
            # Get top results
            top_ids = sorted(
                scores.keys(),
                key=lambda x: scores[x],
                reverse=True
            )[:top_k]
            
            return [
                {
                    "id": doc_id,
                    "content": self.document_index[doc_id]["content"],
                    "metadata": self.document_index[doc_id]["metadata"],
                    "score": float(scores[doc_id])
                }
                for doc_id in top_ids
            ]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def _matches_filters(self,
                       doc_id: str,
                       filters: Dict[str, Any]) -> bool:
        """Check if a document matches the given filters."""
        doc_metadata = self.document_index[doc_id]["metadata"]
        return all(
            doc_metadata.get(key) == value
            for key, value in filters.items()
        ) 