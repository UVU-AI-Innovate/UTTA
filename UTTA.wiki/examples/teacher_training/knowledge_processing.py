#!/usr/bin/env python3
"""
Knowledge Processor for Teacher Training

This module processes educational books and extracts structured knowledge
for use in the teacher training system. It handles the extraction, processing,
and storage of educational content in the vector database.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import core components
from src.core.vector_database import VectorDatabase
from src.core.document_processor import DocumentProcessor
from src.llm.dspy.handler import DSPyLLMHandler

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
SAMPLE_DOCS_DIR = os.path.join(DATA_DIR, 'sample_documents')
KNOWLEDGE_BASE_DIR = os.path.join(DATA_DIR, 'knowledge_base')
PROCESSED_DATA_DIR = os.path.join(KNOWLEDGE_BASE_DIR, 'processed')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeProcessor:
    """
    Processes educational content and research for the teacher training system.
    Extracts, organizes, and indexes educational knowledge for retrieval during simulations.
    """
    
    def __init__(self, vector_db: Optional[VectorDatabase] = None, 
                 llm_handler: Optional[DSPyLLMHandler] = None):
        """
        Initialize the knowledge processor.
        
        Args:
            vector_db: Vector database for storing processed knowledge
            llm_handler: LLM handler for knowledge extraction and processing
        """
        self.vector_db = vector_db
        self.llm_handler = llm_handler
        self.text_processor = DocumentProcessor()
        
        # Create necessary directories
        os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
    def process_educational_books(self, books_dir: str = SAMPLE_DOCS_DIR) -> List[Dict[str, Any]]:
        """
        Process educational books and extract structured knowledge.
        
        Args:
            books_dir: Directory containing educational books
            
        Returns:
            List of processed knowledge entries
        """
        logger.info(f"Processing educational books from {books_dir}")
        
        # Check if directory exists
        if not os.path.exists(books_dir):
            logger.error(f"Books directory not found: {books_dir}")
            return []
            
        # Get list of PDF files
        pdf_files = [f for f in os.listdir(books_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {books_dir}")
            return []
            
        # Process each book
        knowledge_entries = []
        for pdf_file in tqdm(pdf_files, desc="Processing books"):
            file_path = os.path.join(books_dir, pdf_file)
            try:
                # Extract text from PDF
                text_chunks = self.text_processor.extract_text_from_pdf(file_path)
                
                # Process each chunk
                for i, chunk in enumerate(text_chunks):
                    # Extract knowledge from chunk
                    entry = self._extract_knowledge_from_text(chunk, source=pdf_file, chunk_id=i)
                    if entry:
                        knowledge_entries.append(entry)
                        
                        # Add to vector database if available
                        if self.vector_db:
                            self.vector_db.add_document(
                                text=chunk,
                                metadata={
                                    "source": pdf_file,
                                    "chunk_id": i,
                                    "knowledge_type": entry.get("knowledge_type", "general"),
                                    "topics": entry.get("topics", [])
                                }
                            )
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {e}")
                
        # Save processed knowledge
        self._save_knowledge_entries(knowledge_entries)
        
        return knowledge_entries
        
    def _extract_knowledge_from_text(self, text: str, source: str, chunk_id: int) -> Dict[str, Any]:
        """
        Extract structured knowledge from text.
        
        Args:
            text: Text to process
            source: Source of the text
            chunk_id: Chunk identifier
            
        Returns:
            Structured knowledge entry
        """
        # Use LLM to extract knowledge if available
        if self.llm_handler:
            try:
                prompt = f"""
                Extract educational knowledge from the following text. 
                Identify the knowledge type (e.g., pedagogical_theory, classroom_management, 
                subject_content, student_development, assessment_methods).
                Also extract key topics, main points, and practical applications.
                
                TEXT:
                {text}
                
                OUTPUT FORMAT:
                {{
                    "knowledge_type": "type",
                    "topics": ["topic1", "topic2"],
                    "main_points": ["point1", "point2"],
                    "practical_applications": ["application1", "application2"]
                }}
                """
                
                result = self.llm_handler.generate(prompt)
                try:
                    # Parse JSON response
                    knowledge = json.loads(result)
                    knowledge["source"] = source
                    knowledge["chunk_id"] = chunk_id
                    knowledge["text"] = text[:500]  # Store first 500 chars of text
                    return knowledge
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON: {result[:100]}...")
            except Exception as e:
                logger.error(f"Error extracting knowledge with LLM: {e}")
        
        # Fallback to basic extraction
        return {
            "knowledge_type": "general",
            "topics": [],
            "main_points": [],
            "practical_applications": [],
            "source": source,
            "chunk_id": chunk_id,
            "text": text[:500]  # Store first 500 chars of text
        }
        
    def _save_knowledge_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Save knowledge entries to disk."""
        if not entries:
            return
            
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(entries)
        
        # Save as CSV
        output_path = os.path.join(PROCESSED_DATA_DIR, "knowledge_entries.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(entries)} knowledge entries to {output_path}")
        
        # Save as JSON for full data
        json_path = os.path.join(PROCESSED_DATA_DIR, "knowledge_entries.json")
        with open(json_path, 'w') as f:
            json.dump(entries, f, indent=2)
        logger.info(f"Saved knowledge entries as JSON to {json_path}")
        
    def load_knowledge_entries(self) -> List[Dict[str, Any]]:
        """Load processed knowledge entries from disk."""
        json_path = os.path.join(PROCESSED_DATA_DIR, "knowledge_entries.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    entries = json.load(f)
                logger.info(f"Loaded {len(entries)} knowledge entries from {json_path}")
                return entries
            except Exception as e:
                logger.error(f"Error loading knowledge entries: {e}")
                
        return []
        
    def build_knowledge_base(self) -> None:
        """Build or rebuild the knowledge base from processed entries."""
        entries = self.load_knowledge_entries()
        
        if not entries:
            logger.warning("No knowledge entries found. Processing books first...")
            entries = self.process_educational_books()
            
        if not entries:
            logger.error("Failed to build knowledge base: no entries available")
            return
            
        # Build vector database if available
        if self.vector_db:
            logger.info("Building vector database...")
            for entry in tqdm(entries, desc="Adding to vector DB"):
                text = entry.get("text", "")
                if not text and "source" in entry and "chunk_id" in entry:
                    # Try to reload the text from the original source
                    source_path = os.path.join(SAMPLE_DOCS_DIR, entry["source"])
                    if os.path.exists(source_path):
                        chunks = self.text_processor.extract_text_from_pdf(source_path)
                        if 0 <= entry["chunk_id"] < len(chunks):
                            text = chunks[entry["chunk_id"]]
                
                if text:
                    self.vector_db.add_document(
                        text=text,
                        metadata={
                            "source": entry.get("source", "unknown"),
                            "chunk_id": entry.get("chunk_id", 0),
                            "knowledge_type": entry.get("knowledge_type", "general"),
                            "topics": entry.get("topics", [])
                        }
                    )
            
            logger.info("Knowledge base built successfully")
        else:
            logger.warning("No vector database available for building knowledge base")

if __name__ == "__main__":
    # Example usage
    processor = KnowledgeProcessor()
    processor.process_educational_books()
    processor.build_knowledge_base() 