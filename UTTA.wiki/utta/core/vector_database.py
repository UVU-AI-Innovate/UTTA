#!/usr/bin/env python3
"""
Vector Database for Educational Knowledge

This module provides a vector database implementation for storing and retrieving
knowledge chunks extracted from scientific educational books. It uses semantic
embeddings to enable advanced retrieval of relevant knowledge based on context
and meaning rather than just keywords.
"""

import os
import json
import sqlite3
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="vector_database.log"
)
logger = logging.getLogger(__name__)

class VectorDatabase:
    """
    A database for storing and retrieving vectorized knowledge chunks.
    
    This class creates embeddings from text chunks and provides semantic
    search functionality to find the most relevant knowledge chunks
    based on meaning similarity.
    """
    
    def __init__(self, db_path: str = os.path.join("knowledge_base", "vectors", "knowledge_vectors.db")):
        """
        Initialize the vector database.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.model = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        
        # Ensure the database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Check for dependencies
        self._check_dependencies()
        
        # Initialize the database
        self._init_db()
    
    def _check_dependencies(self):
        """Check and initialize required dependencies."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            
            # Log which device will be used
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using pytorch device_name: {device}")
            
            # Load the embedding model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.model.to(device)
            
            # Check if FAISS can use GPU
            try:
                import faiss.contrib.torch_utils
                self.use_gpu_index = torch.cuda.is_available()
                if self.use_gpu_index:
                    logger.info("FAISS will use GPU acceleration")
                else:
                    logger.info("FAISS will use CPU (no GPU available)")
            except ImportError:
                self.use_gpu_index = False
                logger.info("FAISS GPU utils not available, using CPU")
            
        except ImportError as e:
            logger.error(f"Required dependency missing: {e}")
            logger.error("Please install: pip install sentence-transformers faiss-cpu")
            raise
    
    def _init_db(self):
        """Initialize the SQLite database schema for vector storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                category TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                chunk_id INTEGER,
                vector BLOB NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks (id),
                PRIMARY KEY (chunk_id)
            )
            ''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_stats (
                chunk_id INTEGER,
                times_retrieved INTEGER DEFAULT 0,
                times_used INTEGER DEFAULT 0,
                effectiveness_score REAL DEFAULT 0.0,
                last_accessed TIMESTAMP,
                FOREIGN KEY (chunk_id) REFERENCES chunks (id),
                PRIMARY KEY (chunk_id)
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict[str, Any]], category: str = "general") -> None:
        """
        Add knowledge chunks to the database with vectors.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and optional 'metadata'
            category: Category for all chunks in this batch
        """
        if not chunks:
            logger.warning("No chunks provided to add")
            return
        
        if self.model is None:
            logger.error("Embedding model not initialized")
            return
        
        try:
            # Generate embeddings for all chunks in batch
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self.model.encode(texts, show_progress_bar=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert chunks and their vectors
            for i, chunk in enumerate(chunks):
                # Insert chunk
                metadata = json.dumps(chunk.get("metadata", {}))
                cursor.execute(
                    "INSERT INTO chunks (text, category, metadata) VALUES (?, ?, ?)",
                    (chunk["text"], category, metadata)
                )
                
                # Get the ID of the inserted chunk
                chunk_id = cursor.lastrowid
                
                # Insert vector
                vector_bytes = embeddings[i].astype(np.float32).tobytes()
                cursor.execute(
                    "INSERT INTO vectors (chunk_id, vector) VALUES (?, ?)",
                    (chunk_id, vector_bytes)
                )
                
                # Initialize usage stats
                cursor.execute(
                    "INSERT INTO usage_stats (chunk_id) VALUES (?)",
                    (chunk_id,)
                )
            
            conn.commit()
            conn.close()
            logger.info(f"Added {len(chunks)} chunks to database in category: {category}")
        except Exception as e:
            logger.error(f"Error adding chunks to database: {e}")
            raise
    
    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            category: Optional category to filter results
            
        Returns:
            List of matching chunks with similarity scores
        """
        if self.model is None:
            logger.error("Embedding model not initialized")
            return []
        
        try:
            # Encode the query to a vector
            query_vector = self.model.encode(query)
            
            # Build the category filter part of the SQL query
            category_filter = f"AND c.category = '{category}'" if category else ""
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get all vectors and corresponding chunks
            cursor.execute(f"""
            SELECT c.id, c.text, c.category, c.metadata, v.vector
            FROM chunks c
            JOIN vectors v ON c.id = v.chunk_id
            WHERE 1=1 {category_filter}
            """)
            
            results = []
            for row in cursor.fetchall():
                # Convert binary vector data back to numpy array
                vector_bytes = row['vector']
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(vector, query_vector) / (np.linalg.norm(vector) * np.linalg.norm(query_vector))
                
                results.append({
                    'id': row['id'],
                    'text': row['text'],
                    'category': row['category'],
                    'metadata': json.loads(row['metadata']),
                    'similarity': float(similarity)
                })
            
            # Record retrieval in usage stats
            for result in sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]:
                cursor.execute("""
                UPDATE usage_stats 
                SET times_retrieved = times_retrieved + 1, 
                    last_accessed = CURRENT_TIMESTAMP
                WHERE chunk_id = ?
                """, (result['id'],))
            
            conn.commit()
            conn.close()
            
            # Sort by similarity and return top_k results
            return sorted(results, key=lambda x: x['similarity'], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def record_chunk_used(self, chunk_id: int, effectiveness_score: float = 1.0) -> None:
        """
        Record that a chunk was actually used in a response, with an effectiveness score.
        
        Args:
            chunk_id: ID of the chunk used
            effectiveness_score: How effective the chunk was (0.0 to 1.0)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE usage_stats 
            SET times_used = times_used + 1,
                effectiveness_score = (effectiveness_score * times_used + ?) / (times_used + 1)
            WHERE chunk_id = ?
            """, (effectiveness_score, chunk_id))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error recording chunk usage: {e}")
    
    def get_chunks_by_category(self, category: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get chunks by category.
        
        Args:
            category: Category to filter by
            limit: Maximum number of chunks to return
            
        Returns:
            List of chunks in the specified category
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT c.id, c.text, c.category, c.metadata
            FROM chunks c
            WHERE c.category = ?
            LIMIT ?
            """, (category, limit))
            
            results = [{
                'id': row['id'],
                'text': row['text'],
                'category': row['category'],
                'metadata': json.loads(row['metadata'])
            } for row in cursor.fetchall()]
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Error getting chunks by category: {e}")
            return []
    
    def get_most_effective_chunks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most effective chunks based on usage statistics.
        
        Args:
            limit: Maximum number of chunks to return
            
        Returns:
            List of most effective chunks with their stats
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT c.id, c.text, c.category, c.metadata, 
                   u.times_retrieved, u.times_used, u.effectiveness_score
            FROM chunks c
            JOIN usage_stats u ON c.id = u.chunk_id
            WHERE u.times_used > 0
            ORDER BY u.effectiveness_score DESC, u.times_used DESC
            LIMIT ?
            """, (limit,))
            
            results = [{
                'id': row['id'],
                'text': row['text'],
                'category': row['category'],
                'metadata': json.loads(row['metadata']),
                'times_retrieved': row['times_retrieved'],
                'times_used': row['times_used'],
                'effectiveness_score': row['effectiveness_score']
            } for row in cursor.fetchall()]
            
            conn.close()
            return results
        except Exception as e:
            logger.error(f"Error getting most effective chunks: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total number of chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]
            
            # Chunks by category
            cursor.execute("SELECT category, COUNT(*) FROM chunks GROUP BY category")
            categories = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Usage statistics
            cursor.execute("SELECT SUM(times_retrieved), SUM(times_used) FROM usage_stats")
            retrieval_stats = cursor.fetchone()
            total_retrievals = retrieval_stats[0] or 0
            total_uses = retrieval_stats[1] or 0
            
            conn.close()
            
            return {
                'total_chunks': total_chunks,
                'categories': categories,
                'total_retrievals': total_retrievals,
                'total_uses': total_uses
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    def clear_database(self) -> None:
        """Clear all data from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM usage_stats")
            cursor.execute("DELETE FROM vectors")
            cursor.execute("DELETE FROM chunks")
            
            conn.commit()
            conn.close()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")


if __name__ == "__main__":
    # Example usage
    db = VectorDatabase()
    
    # Add some sample chunks
    sample_chunks = [
        {
            "text": "Effective classroom management requires clear expectations and consistent consequences.",
            "metadata": {"source": "sample", "topic": "classroom management"}
        },
        {
            "text": "Differentiated instruction helps meet the needs of diverse learners by adjusting content, process, and product.",
            "metadata": {"source": "sample", "topic": "teaching methods"}
        }
    ]
    
    db.add_chunks(sample_chunks, category="teaching_methodology")
    
    # Search example
    results = db.search("How can I manage difficult behavior in my classroom?")
    for result in results:
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['text']}")
        print()
    
    # Stats
    print("Database stats:", db.get_stats()) 