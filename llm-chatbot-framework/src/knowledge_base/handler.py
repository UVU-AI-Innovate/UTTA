"""
Knowledge Base Handler Module

This module provides functionality for managing and accessing the educational content
knowledge base. It handles content storage, retrieval, and semantic search capabilities.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class KnowledgeBaseHandler:
    def __init__(self, storage_path: Optional[str] = None):
        """Initialize the knowledge base handler.
        
        Args:
            storage_path: Optional path to store knowledge base data.
                        Defaults to 'data/knowledge_base' in project root.
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent / "data" / "knowledge_base"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize sentence transformer for semantic search
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load existing content if available
        self.content_index = self._load_content_index()
        self.content_embeddings = self._load_embeddings()

    def add_content(self, content: Dict[str, str]) -> str:
        """Add new educational content to the knowledge base.
        
        Args:
            content: Dictionary containing content information
                    (topic, subtopic, content, grade_level)
        
        Returns:
            content_id: Unique identifier for the added content
        """
        content_id = f"{content['topic']}_{content['subtopic']}"
        
        # Save content to file
        content_path = self.storage_path / f"{content_id}.json"
        with open(content_path, 'w') as f:
            json.dump(content, f, indent=2)
        
        # Update index
        self.content_index[content_id] = {
            'path': str(content_path),
            'topic': content['topic'],
            'subtopic': content['subtopic'],
            'grade_level': content['grade_level']
        }
        
        # Generate and save embedding
        embedding = self.encoder.encode(content['content'])
        self.content_embeddings[content_id] = embedding
        
        self._save_content_index()
        self._save_embeddings()
        
        return content_id

    def get_content(self, topic: str, subtopic: str) -> Optional[Dict[str, str]]:
        """Retrieve content by topic and subtopic.
        
        Args:
            topic: Main topic of the content
            subtopic: Specific subtopic
        
        Returns:
            Content dictionary if found, None otherwise
        """
        content_id = f"{topic}_{subtopic}"
        if content_id not in self.content_index:
            return None
        
        content_path = self.content_index[content_id]['path']
        with open(content_path, 'r') as f:
            return json.load(f)

    def search_content(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Search for relevant content using semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant content dictionaries
        """
        if not self.content_embeddings:
            return []
        
        # Generate query embedding
        query_embedding = self.encoder.encode(query)
        
        # Calculate similarities
        similarities = {}
        for content_id, embedding in self.content_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities[content_id] = similarity
        
        # Get top-k results
        top_results = []
        for content_id, _ in sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            content = self.get_content(
                self.content_index[content_id]['topic'],
                self.content_index[content_id]['subtopic']
            )
            if content:
                top_results.append(content)
        
        return top_results

    def _load_content_index(self) -> Dict[str, Dict[str, str]]:
        """Load the content index from disk."""
        index_path = self.storage_path / "content_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_content_index(self):
        """Save the content index to disk."""
        index_path = self.storage_path / "content_index.json"
        with open(index_path, 'w') as f:
            json.dump(self.content_index, f, indent=2)

    def _load_embeddings(self) -> Dict[str, np.ndarray]:
        """Load content embeddings from disk."""
        embeddings_path = self.storage_path / "embeddings.npz"
        if embeddings_path.exists():
            data = np.load(embeddings_path)
            return {k: data[k] for k in data.files}
        return {}

    def _save_embeddings(self):
        """Save content embeddings to disk."""
        embeddings_path = self.storage_path / "embeddings.npz"
        np.savez(embeddings_path, **self.content_embeddings) 