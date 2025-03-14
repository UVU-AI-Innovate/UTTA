#!/usr/bin/env python3
"""
LlamaIndex Integration for UTTA

This module provides integration with LlamaIndex for document retrieval and querying.
"""

import os
import json
import hashlib
import logging
import traceback
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

# LlamaIndex imports
from llama_index.core import (
    SimpleDirectoryReader, 
    Settings,
    StorageContext,
    load_index_from_storage,
    set_global_service_context
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.core.response.schema import Response
from llama_index.llms.openai import OpenAI

# Try to import Anthropic, but provide fallback
try:
    from llama_index.llms.anthropic import Anthropic
except ImportError:
    try:
        # Alternative import path for newer versions
        from llama_index.core.llms.anthropic import Anthropic
    except ImportError:
        # Fallback in case Anthropic is not available
        print("Warning: Anthropic integration not available. Installing required package.")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-llms-anthropic"])
        try:
            from llama_index.llms.anthropic import Anthropic
        except ImportError:
            print("Warning: Failed to import Anthropic. Will use mock LLM for this provider.")
            Anthropic = None

# Import MockLLM for testing
from llama_index.core.llms.mock import MockLLM

# Import our configuration
from llama_index_config import (
    get_llm_settings, 
    get_embedding_settings,
    get_document_settings, 
    get_cache_settings,
    get_retrieval_settings
)

# Try to import OpenAIEmbedding, but provide fallback
try:
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    try:
        # Alternative import path for newer versions
        from llama_index.core.embeddings.openai import OpenAIEmbedding 
    except ImportError:
        # Fallback to HuggingFace embeddings
        print("Warning: OpenAI embedding not available. Using HuggingFace embeddings.")
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            OpenAIEmbedding = HuggingFaceEmbedding  # Use as replacement
        except ImportError:
            try:
                from llama_index.core.embeddings.huggingface import HuggingFaceEmbedding
                OpenAIEmbedding = HuggingFaceEmbedding  # Use as replacement
            except ImportError:
                import subprocess
                import sys
                print("Installing huggingface embeddings...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-embeddings-huggingface"])
                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    OpenAIEmbedding = HuggingFaceEmbedding  # Use as replacement
                except ImportError:
                    print("Error: Failed to import embedding models. The application may not work correctly.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LlamaIndexKnowledgeManager:
    """
    Manages knowledge retrieval using LlamaIndex.
    
    This class provides an interface to load documents, create and query indices,
    and manage the lifecycle of LlamaIndex operations.
    """
    
    def __init__(
        self,
        documents_dir: Optional[str] = None,
        index_dir: Optional[str] = None,
        llm_provider: str = "openai",
        enable_caching: bool = True,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 3600,  # 1 hour default
    ):
        """
        Initialize the LlamaIndexKnowledgeManager.
        
        Args:
            documents_dir: Directory containing documents to be indexed.
            index_dir: Directory to store the index.
            llm_provider: LLM provider to use (openai, anthropic, or local).
            enable_caching: Whether to enable query caching.
            cache_dir: Directory to store the query cache.
            cache_ttl: Time-to-live for cache entries in seconds.
        """
        # Load settings
        doc_settings = get_document_settings()
        cache_settings = get_cache_settings()
        
        # Initialize directories
        self.documents_dir = documents_dir or doc_settings["documents_dir"]
        self.index_dir = index_dir or doc_settings["index_dir"]
        
        # Initialize caching settings
        self.enable_caching = enable_caching if enable_caching is not None else cache_settings["enable_caching"]
        self.cache_dir = cache_dir or cache_settings["cache_dir"]
        self.cache_ttl = cache_ttl or cache_settings["cache_ttl"]
        
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize LLM settings
        self.llm_provider = llm_provider
        self._init_llm()
        
        # Initialize the index and query engine
        self.index = None
        self.query_engine = None
        
        logger.info(f"LlamaIndexKnowledgeManager initialized with documents_dir={self.documents_dir}, " 
                   f"index_dir={self.index_dir}, llm_provider={llm_provider}")
    
    def _init_llm(self):
        """Initialize the LLM based on the provider."""
        provider = self.llm_provider.lower()
        
        if provider == "openai":
            try:
                # Get OpenAI settings
                openai_settings = get_llm_settings("openai")
                
                # Check if API key is set
                if not openai_settings.get("api_key"):
                    logging.warning("OpenAI API key not set. Using mock LLM.")
                    self.llm = MockLLM()
                    return
                
                # Initialize OpenAI LLM
                self.llm = OpenAI(
                    model=openai_settings.get("model", "gpt-3.5-turbo"),
                    temperature=openai_settings.get("temperature", 0.1),
                    api_key=openai_settings.get("api_key"),
                )
            except Exception as e:
                logging.error(f"Error initializing OpenAI LLM: {str(e)}")
                logging.warning("Using mock LLM as fallback.")
                self.llm = MockLLM()
        
        elif provider == "anthropic":
            try:
                # Check if Anthropic is available
                if Anthropic is None:
                    logging.warning("Anthropic not available. Using mock LLM.")
                    self.llm = MockLLM()
                    return
                
                # Get Anthropic settings
                anthropic_settings = get_llm_settings("anthropic")
                
                # Check if API key is set
                if not anthropic_settings.get("api_key"):
                    logging.warning("Anthropic API key not set. Using mock LLM.")
                    self.llm = MockLLM()
                    return
                
                # Initialize Anthropic LLM
                self.llm = Anthropic(
                    model=anthropic_settings.get("model", "claude-3-sonnet-20240229"),
                    temperature=anthropic_settings.get("temperature", 0.1),
                    api_key=anthropic_settings.get("api_key"),
                )
            except Exception as e:
                logging.error(f"Error initializing Anthropic LLM: {str(e)}")
                logging.warning("Using mock LLM as fallback.")
                self.llm = MockLLM()
        
        else:
            # Default to mock LLM for testing or unknown providers
            logging.info(f"Using mock LLM for provider: {provider}")
            self.llm = MockLLM()
        
        # Configure global settings
        Settings.llm = self.llm
        
        # Setup embedding model
        embed_settings = get_embedding_settings()
        Settings.embed_model = OpenAIEmbedding(
            model_name=embed_settings["model_name"],
            cache_folder=embed_settings.get("cache_folder")
        )
        
        # Set other settings
        doc_settings = get_document_settings()
        Settings.chunk_size = doc_settings["chunk_size"]
        Settings.chunk_overlap = doc_settings["chunk_overlap"]
        
        logger.info(f"LLM initialized: {self.llm_provider}")
    
    def load_or_create_index(self) -> VectorStoreIndex:
        """
        Load an existing index or create a new one if it doesn't exist.
        
        Returns:
            The loaded or created index.
        """
        try:
            if os.path.exists(self.index_dir):
                logger.info(f"Loading index from {self.index_dir}")
                storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
                self.index = load_index_from_storage(storage_context)
                logger.info("Index loaded successfully")
            else:
                logger.info(f"Index not found at {self.index_dir}, creating new index")
                self.index = self._create_new_index()
                logger.info("New index created successfully")
            
            return self.index
        except Exception as e:
            logger.error(f"Error loading or creating index: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_new_index(self) -> VectorStoreIndex:
        """
        Create a new index from documents.
        
        Returns:
            The newly created index.
        """
        logger.info(f"Creating new index from documents in {self.documents_dir}")
        
        if not os.path.exists(self.documents_dir):
            logger.error(f"Documents directory does not exist: {self.documents_dir}")
            raise FileNotFoundError(f"Documents directory not found: {self.documents_dir}")
        
        try:
            # Load documents
            documents = SimpleDirectoryReader(self.documents_dir).load_data()
            logger.info(f"Loaded {len(documents)} documents")
            
            # Create the index
            index = VectorStoreIndex.from_documents(documents)
            
            # Persist to disk
            index.storage_context.persist(persist_dir=self.index_dir)
            logger.info(f"Index persisted to {self.index_dir}")
            
            # Create query engine
            self.index = index
            
            return index
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _get_cache_key(self, query: str, top_k: int) -> str:
        """
        Generate a cache key for a query.
        
        Args:
            query: The query to generate a key for.
            top_k: Number of results to retrieve.
            
        Returns:
            The cache key as a string.
        """
        # Create a deterministic key from the query and top_k
        key_data = f"{query}_{top_k}_{self.llm_provider}".encode('utf-8')
        return hashlib.md5(key_data).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: The cache key.
            
        Returns:
            The path to the cache file.
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _save_to_cache(self, cache_key: str, response_dict: Dict[str, Any]):
        """
        Save a response to the cache.
        
        Args:
            cache_key: The cache key.
            response_dict: The response dictionary to cache.
        """
        if not self.enable_caching:
            return
        
        cache_path = self._get_cache_path(cache_key)
        
        # Add timestamp for TTL checks
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "response": response_dict
        }
        
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        
        logger.info(f"Saved response to cache: {cache_path}")
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a response from the cache if it exists and is not expired.
        
        Args:
            cache_key: The cache key.
            
        Returns:
            The cached response dictionary or None if not found or expired.
        """
        if not self.enable_caching:
            return None
        
        cache_path = self._get_cache_path(cache_key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            timestamp = datetime.fromisoformat(cache_data["timestamp"])
            if datetime.now() - timestamp > timedelta(seconds=self.cache_ttl):
                logger.info(f"Cache expired: {cache_path}")
                return None
            
            logger.info(f"Retrieved response from cache: {cache_path}")
            return cache_data["response"]
        except Exception as e:
            logger.error(f"Error reading cache: {str(e)}")
            return None
    
    def clear_cache(self):
        """Clear all cached responses."""
        if not self.enable_caching or not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, filename))
        
        logger.info("Cache cleared")
    
    def query_knowledge(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base with a natural language query.
        
        Args:
            query: The natural language query string.
            top_k: Number of results to retrieve.
            
        Returns:
            Dictionary containing the response and source documents.
        """
        logger.info(f"Querying knowledge base with: '{query}', top_k={top_k}")
        
        # Check cache first
        cache_key = self._get_cache_key(query, top_k)
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.info("Using cached response")
            return cached_response
        
        # Make sure index is loaded
        if not self.index:
            logger.info("Index not loaded, loading now")
            self.load_or_create_index()
        
        # Create query engine if not already created
        if not self.query_engine:
            retrieval_settings = get_retrieval_settings()
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=top_k or retrieval_settings["top_k"],
                node_postprocessors=[]
            )
        
        try:
            # Execute query
            response = self.query_engine.query(query)
            
            # Format response
            result = {
                "response": str(response),
                "sources": [
                    {
                        "text": node.node.get_text(),
                        "source": node.node.metadata.get("file_path", "Unknown"),
                        "score": node.score
                    }
                    for node in response.source_nodes
                ]
            }
            
            # Cache the response
            self._save_to_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "response": f"Error querying knowledge base: {str(e)}",
                "sources": [],
                "error": str(e)
            }

def demonstrate_llama_index():
    """Demonstrate the LlamaIndex integration."""
    print("\n===== Demonstrating LlamaIndex Integration =====\n")
    
    try:
        # Initialize the knowledge manager
        knowledge_manager = LlamaIndexKnowledgeManager(
            llm_provider="local"  # Use MockLLM for testing
        )
        
        # Load or create the index
        knowledge_manager.load_or_create_index()
        
        # Example queries
        queries = [
            "What are effective strategies for teaching fractions?",
            "How do I manage a classroom with diverse learning needs?",
            "What are signs of dyscalculia in second-grade students?"
        ]
        
        # Execute and display results for each query
        for query in queries:
            print(f"\nQuery: {query}")
            result = knowledge_manager.query_knowledge(query)
            
            print("\nResponse:")
            print(result["response"])
            
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['source']}")
                print(f"  {source['text'][:200]}...")  # Show first 200 chars
                print(f"  Score: {source['score']}")
            
            print("\n" + "-" * 50)
        
    except Exception as e:
        print(f"Error demonstrating LlamaIndex: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    demonstrate_llama_index() 