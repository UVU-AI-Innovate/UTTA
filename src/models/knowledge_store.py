from pathlib import Path
import sqlite3
import pickle
import numpy as np
from typing import List, Dict, Any

class KnowledgeStore:
    def __init__(self, db_path: str = "knowledge_base/knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    source TEXT,
                    chunk_type TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER,
                    embedding BLOB,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
            """)

    def add_document(self, content: str, source: str, chunk_type: str, metadata: Dict, embedding: np.ndarray):
        """Add a document and its embedding to the store."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (content, source, chunk_type, metadata) VALUES (?, ?, ?, ?)",
                (content, source, chunk_type, str(metadata))
            )
            doc_id = cursor.lastrowid
            
            # Store embedding as binary
            cursor.execute(
                "INSERT INTO embeddings (document_id, embedding) VALUES (?, ?)",
                (doc_id, pickle.dumps(embedding))
            )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents with their embeddings."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.id, d.content, d.source, d.chunk_type, d.metadata, e.embedding
                FROM documents d
                JOIN embeddings e ON d.id = e.document_id
            """)
            
            return [{
                'id': row[0],
                'content': row[1],
                'source': row[2],
                'chunk_type': row[3],
                'metadata': eval(row[4]),
                'embedding': pickle.loads(row[5])
            } for row in cursor.fetchall()]

    def get_documents_by_source(self, source: str) -> List[Dict[str, Any]]:
        """Get all documents from a specific source."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.id, d.content, d.chunk_type, d.metadata
                FROM documents d
                WHERE d.source = ?
            """, (source,))
            
            return [{
                'id': row[0],
                'content': row[1],
                'chunk_type': row[2],
                'metadata': eval(row[3])
            } for row in cursor.fetchall()]

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.id, d.content, d.source, d.chunk_type, d.metadata, e.embedding
                FROM documents d
                JOIN embeddings e ON d.id = e.document_id
            """)
            
            results = []
            for row in cursor.fetchall():
                doc_embedding = pickle.loads(row[5])
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                results.append({
                    'id': row[0],
                    'content': row[1],
                    'source': row[2],
                    'chunk_type': row[3],
                    'metadata': eval(row[4]),
                    'similarity': float(similarity)
                })
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]

    def clear(self):
        """Clear all documents from the store."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM embeddings")
            conn.execute("DELETE FROM documents") 