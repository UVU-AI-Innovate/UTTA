#!/usr/bin/env python3

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.knowledge_base.handler import KnowledgeBaseHandler
from src.knowledge_base.indexer import DocumentIndexer

def test_knowledge_base_creation():
    """Test creating and initializing the knowledge base."""
    print("\nTesting Knowledge Base Creation...")
    
    kb_handler = KnowledgeBaseHandler()
    
    # Test adding educational content
    test_content = {
        "topic": "mathematics",
        "subtopic": "algebra",
        "content": """
        Algebraic concepts for beginners:
        1. Variables are like containers that hold numbers
        2. Equations show relationships between variables
        3. Solving equations is like solving a puzzle
        """,
        "grade_level": "middle_school"
    }
    
    print("\nAdding test content to knowledge base...")
    kb_handler.add_content(test_content)
    
    # Verify content was added
    retrieved_content = kb_handler.get_content("mathematics", "algebra")
    print(f"\nRetrieved content: {retrieved_content}")
    print("-" * 80)

def test_content_retrieval():
    """Test retrieving relevant content from the knowledge base."""
    print("\nTesting Content Retrieval...")
    
    kb_handler = KnowledgeBaseHandler()
    
    # Test semantic search
    query = "How to teach variables to beginners?"
    print(f"\nSearching for: {query}")
    
    results = kb_handler.search_content(query)
    print(f"\nSearch results: {results}")
    print("-" * 80)

def test_document_indexing():
    """Test document indexing capabilities."""
    print("\nTesting Document Indexing...")
    
    indexer = DocumentIndexer()
    
    # Test indexing a sample document
    test_doc = {
        "file_path": "sample_lesson.txt",
        "content": """
        Lesson Plan: Introduction to Variables
        Objective: Students will understand what variables are and how to use them
        Materials: Worksheets, manipulatives
        Duration: 45 minutes
        """
    }
    
    print("\nIndexing test document...")
    doc_id = indexer.index_document(test_doc)
    
    # Test retrieval
    retrieved_doc = indexer.get_document(doc_id)
    print(f"\nRetrieved document: {retrieved_doc}")
    print("-" * 80)

def main():
    print("Starting Knowledge Base Integration Tests...")
    
    try:
        test_knowledge_base_creation()
        test_content_retrieval()
        test_document_indexing()
        print("\nAll knowledge base tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 