#!/usr/bin/env python3
"""
LlamaIndex Configuration Module

This module provides centralized configuration management for the LlamaIndex integration.
It handles loading configuration from environment variables, providing default values,
and organizing settings into logical groups.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("llama_index_config")

# Default settings
DEFAULT_SETTINGS = {
    # LLM settings
    "llm": {
        "provider": os.environ.get("LLAMA_INDEX_LLM_PROVIDER", "openai"),
        # OpenAI settings
        "openai": {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "model": os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
            "temperature": float(os.environ.get("OPENAI_TEMPERATURE", "0.1")),
            "max_tokens": int(os.environ.get("OPENAI_MAX_TOKENS", "4096")),
        },
        # Anthropic settings
        "anthropic": {
            "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
            "temperature": float(os.environ.get("ANTHROPIC_TEMPERATURE", "0.1")),
            "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS", "4096")),
        },
        # Local model settings
        "local": {
            "model_path": os.environ.get("LOCAL_MODEL_PATH", ""),
            "temperature": float(os.environ.get("LOCAL_TEMPERATURE", "0.1")),
            "max_tokens": int(os.environ.get("LOCAL_MAX_TOKENS", "4096")),
        },
    },
    
    # Embedding settings
    "embedding": {
        "model_name": os.environ.get("LLAMA_INDEX_EMBEDDING_MODEL", 
                                      "sentence-transformers/all-MiniLM-L6-v2"),
        "cache_folder": os.environ.get("LLAMA_INDEX_EMBEDDING_CACHE", ".embedding_cache"),
    },
    
    # Document processing settings
    "document": {
        "chunk_size": int(os.environ.get("LLAMA_INDEX_CHUNK_SIZE", "1024")),
        "chunk_overlap": int(os.environ.get("LLAMA_INDEX_CHUNK_OVERLAP", "20")),
        "similarity_top_k": int(os.environ.get("LLAMA_INDEX_TOP_K", "5")),
        "similarity_cutoff": float(os.environ.get("LLAMA_INDEX_SIMILARITY_CUTOFF", "0.7")),
    },
    
    # Caching settings
    "cache": {
        "enable": os.environ.get("LLAMA_INDEX_ENABLE_CACHE", "true").lower() == "true",
        "ttl": int(os.environ.get("LLAMA_INDEX_CACHE_TTL", "3600")),  # 1 hour default
    },
    
    # Directory settings
    "directories": {
        "knowledge_dir": os.environ.get("KNOWLEDGE_DIR", "./knowledge"),
        "index_dir": os.environ.get("INDEX_DIR", "./index_storage"),
        "cache_dir": os.environ.get("CACHE_DIR", "./.cache"),
    },
}

def get_all_settings() -> Dict[str, Any]:
    """
    Get all configuration settings
    
    Returns:
        Dict containing all configuration settings
    """
    return DEFAULT_SETTINGS

def get_llm_settings(provider: Optional[str] = None) -> Dict[str, Any]:
    """
    Get LLM settings for the specified provider or the default provider
    
    Args:
        provider: Optional provider name (openai, anthropic, local)
        
    Returns:
        Dict containing LLM settings for the specified provider
    """
    if provider is None:
        provider = DEFAULT_SETTINGS["llm"]["provider"]
    
    if provider not in ["openai", "anthropic", "local"]:
        logger.warning(f"Unknown provider: {provider}, falling back to openai")
        provider = "openai"
    
    return DEFAULT_SETTINGS["llm"][provider]

def get_embedding_settings() -> Dict[str, Any]:
    """
    Get embedding model settings
    
    Returns:
        Dict containing embedding settings
    """
    return DEFAULT_SETTINGS["embedding"]

def get_document_settings() -> Dict[str, Any]:
    """
    Get document processing settings
    
    Returns:
        Dict containing document processing settings
    """
    return DEFAULT_SETTINGS["document"]

def get_cache_settings() -> Dict[str, Any]:
    """
    Get cache settings
    
    Returns:
        Dict containing cache settings
    """
    return DEFAULT_SETTINGS["cache"]

def get_directory_settings() -> Dict[str, Any]:
    """
    Get directory settings
    
    Returns:
        Dict containing directory settings
    """
    return DEFAULT_SETTINGS["directories"]

def is_cache_enabled() -> bool:
    """
    Check if caching is enabled
    
    Returns:
        Boolean indicating if caching is enabled
    """
    return DEFAULT_SETTINGS["cache"]["enable"]

def validate_configuration() -> bool:
    """
    Validate the current configuration
    
    Returns:
        Boolean indicating if the configuration is valid
    """
    current_provider = DEFAULT_SETTINGS["llm"]["provider"]
    provider_settings = DEFAULT_SETTINGS["llm"][current_provider]
    
    # Check API key for OpenAI and Anthropic
    if current_provider in ["openai", "anthropic"]:
        if not provider_settings.get("api_key"):
            logger.error(f"No API key found for {current_provider}")
            return False
    
    # Check model path for local
    if current_provider == "local" and not provider_settings.get("model_path"):
        logger.error("No model path specified for local provider")
        return False
    
    # Check if knowledge directory exists
    knowledge_dir = DEFAULT_SETTINGS["directories"]["knowledge_dir"]
    if not os.path.exists(knowledge_dir):
        logger.warning(f"Knowledge directory does not exist: {knowledge_dir}")
        # Don't fail, but warn
    
    return True

# Validate configuration when module is loaded
validate_configuration() 