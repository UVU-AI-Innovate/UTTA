#!/usr/bin/env python3
"""
LlamaIndex Web Application Integration

This module provides the necessary utilities to integrate the LlamaIndex knowledge
management system with a web application framework like Flask or FastAPI.
"""

import logging
import functools
import traceback
from typing import Dict, Any, Callable, Optional, List, TypeVar, cast

# Import our LlamaIndex knowledge manager
from llama_index_integration import LlamaIndexKnowledgeManager
from llama_index_config import get_llm_settings, get_cache_settings, get_retrieval_settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton instance of the knowledge manager
_knowledge_manager: Optional[LlamaIndexKnowledgeManager] = None

# Type variable for our decorator
F = TypeVar('F', bound=Callable[..., Any])

def init_knowledge_manager(
    documents_dir: Optional[str] = None,
    index_dir: Optional[str] = None,
    llm_provider: Optional[str] = None,
    force_reload: bool = False
) -> LlamaIndexKnowledgeManager:
    """
    Initialize the global knowledge manager instance.
    
    Args:
        documents_dir: Directory containing documents to be indexed.
        index_dir: Directory to store the index.
        llm_provider: LLM provider to use (openai, anthropic, or local).
        force_reload: Whether to force reload the index.
        
    Returns:
        The initialized knowledge manager instance.
    """
    global _knowledge_manager
    
    # Only initialize if not already initialized
    if _knowledge_manager is None or force_reload:
        # Get settings from config
        cache_settings = get_cache_settings()
        
        # Use provided values or defaults from config
        _knowledge_manager = LlamaIndexKnowledgeManager(
            documents_dir=documents_dir,
            index_dir=index_dir,
            llm_provider=llm_provider or "openai",
            enable_caching=cache_settings["enable_caching"],
            cache_dir=cache_settings["cache_dir"],
            cache_ttl=cache_settings["cache_ttl"]
        )
        
        # Load or create the index
        _knowledge_manager.load_or_create_index()
        
        logger.info(f"Knowledge manager initialized with provider: {llm_provider or 'openai'}")
    
    return _knowledge_manager

def get_knowledge_manager() -> LlamaIndexKnowledgeManager:
    """
    Get the global knowledge manager instance, initializing it if necessary.
    
    Returns:
        The knowledge manager instance.
    """
    if _knowledge_manager is None:
        return init_knowledge_manager()
    return _knowledge_manager

def with_knowledge_manager(func: F) -> F:
    """
    Decorator to inject the knowledge manager into a function.
    
    This decorator ensures the knowledge manager is initialized and passed
    to the decorated function.
    
    Args:
        func: The function to decorate.
        
    Returns:
        The decorated function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Initialize the knowledge manager if not already done
        km = get_knowledge_manager()
        
        # Add the knowledge manager as a keyword argument
        kwargs['knowledge_manager'] = km
        
        # Call the original function
        return func(*args, **kwargs)
    
    return cast(F, wrapper)

def format_knowledge_response(
    response_data: Dict[str, Any], 
    max_source_preview: int = 200
) -> Dict[str, Any]:
    """
    Format a knowledge query response for web display.
    
    Args:
        response_data: Raw response data from the knowledge manager.
        max_source_preview: Maximum length of source text preview.
        
    Returns:
        Formatted response for web display.
    """
    try:
        if "error" in response_data:
            return {
                "success": False,
                "error": response_data["error"],
                "message": "An error occurred while querying the knowledge base"
            }
        
        # Format sources for display
        formatted_sources = []
        for source in response_data.get("sources", []):
            # Handle both old and new format
            if isinstance(source, dict):
                source_text = source.get("text", "")
                source_path = source.get("source", "Unknown")
                source_score = source.get("score", 0.0)
            else:
                source_text = source
                source_path = "Unknown"
                source_score = 0.0
            
            # Extract file name from path
            file_name = source_path.split("/")[-1] if "/" in source_path else source_path
            
            formatted_sources.append({
                "title": file_name,
                "text": source_text[:max_source_preview] + ("..." if len(source_text) > max_source_preview else ""),
                "full_text": source_text,
                "path": source_path,
                "score": source_score
            })
        
        return {
            "success": True,
            "response": response_data.get("response", ""),
            "sources": formatted_sources,
            "source_count": len(formatted_sources)
        }
    except Exception as e:
        logger.error(f"Error formatting knowledge response: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while formatting the response"
        }

@with_knowledge_manager
def query_educational_knowledge(
    query: str,
    top_k: Optional[int] = None,
    knowledge_manager: Optional[LlamaIndexKnowledgeManager] = None
) -> Dict[str, Any]:
    """
    Query the educational knowledge base and return formatted results.
    
    Args:
        query: The natural language query string.
        top_k: Number of results to retrieve.
        knowledge_manager: Knowledge manager (injected by decorator).
        
    Returns:
        Formatted response with results.
    """
    if not knowledge_manager:
        return {
            "success": False,
            "error": "Knowledge manager not initialized",
            "message": "The knowledge management system is not available"
        }
    
    try:
        # Get retrieval settings from config
        retrieval_settings = get_retrieval_settings()
        
        # Use provided top_k or default from config
        top_k = top_k or retrieval_settings["top_k"]
        
        # Query the knowledge base
        logger.info(f"Querying educational knowledge: '{query}' (top_k={top_k})")
        result = knowledge_manager.query_knowledge(query, top_k=top_k)
        
        # Format the response
        return format_knowledge_response(result)
    except Exception as e:
        logger.error(f"Error querying educational knowledge: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while querying the knowledge base"
        }

@with_knowledge_manager
def clear_knowledge_cache(knowledge_manager: Optional[LlamaIndexKnowledgeManager] = None) -> Dict[str, Any]:
    """
    Clear the knowledge query cache.
    
    Args:
        knowledge_manager: Knowledge manager (injected by decorator).
        
    Returns:
        Response indicating success or failure.
    """
    if not knowledge_manager:
        return {
            "success": False,
            "error": "Knowledge manager not initialized",
            "message": "The knowledge management system is not available"
        }
    
    try:
        knowledge_manager.clear_cache()
        return {
            "success": True,
            "message": "Knowledge cache cleared successfully"
        }
    except Exception as e:
        logger.error(f"Error clearing knowledge cache: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "message": "An error occurred while clearing the knowledge cache"
        }

# Example Flask route implementations (commented out)
"""
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/knowledge/query', methods=['POST'])
def api_knowledge_query():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', None)
    
    if not query:
        return jsonify({
            "success": False,
            "error": "No query provided",
            "message": "Please provide a query"
        }), 400
    
    result = query_educational_knowledge(query, top_k)
    return jsonify(result)

@app.route('/api/knowledge/cache/clear', methods=['POST'])
def api_clear_knowledge_cache():
    result = clear_knowledge_cache()
    return jsonify(result)

if __name__ == '__main__':
    # Initialize the knowledge manager on startup
    init_knowledge_manager()
    app.run(debug=True)
"""

# Example FastAPI implementation (commented out)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class KnowledgeQuery(BaseModel):
    query: str
    top_k: Optional[int] = None

@app.post("/api/knowledge/query")
async def api_knowledge_query(query_data: KnowledgeQuery):
    if not query_data.query:
        raise HTTPException(status_code=400, detail="No query provided")
    
    result = query_educational_knowledge(query_data.query, query_data.top_k)
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
    
    return result

@app.post("/api/knowledge/cache/clear")
async def api_clear_knowledge_cache():
    result = clear_knowledge_cache()
    
    if not result.get("success", False):
        raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
    
    return result

@app.on_event("startup")
async def startup_event():
    # Initialize the knowledge manager on startup
    init_knowledge_manager()
"""

if __name__ == "__main__":
    # Simple demonstration
    init_knowledge_manager(llm_provider="local")  # Use MockLLM for testing
    result = query_educational_knowledge("What are effective teaching strategies for math?")
    
    print("\n===== Web App Integration Demo =====\n")
    print(f"Query: What are effective teaching strategies for math?")
    print(f"\nResponse: {result['response']}")
    
    print("\nSources:")
    for i, source in enumerate(result["sources"], 1):
        print(f"\n{i}. {source['title']}")
        print(f"   {source['text']}")
        print(f"   Score: {source.get('score', 'N/A')}") 