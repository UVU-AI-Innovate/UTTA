#!/usr/bin/env python3
"""
Production Test Script for LlamaIndex Integration

This script allows testing the LlamaIndex integration with real LLM providers
(OpenAI or Anthropic) in a production-like environment.
"""

import os
import sys
import time
import json
import argparse
import logging
from typing import Dict, Any, List, Optional

# Import LlamaIndex components
from llama_index_integration import LlamaIndexKnowledgeManager
from llama_index_config import get_llm_settings, get_all_settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger("llama_index_test")

def list_available_providers() -> List[str]:
    """List available LLM providers based on API keys in environment."""
    available = ["local"]  # MockLLM is always available
    
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    
    return available

def test_knowledge_query(
    provider: str,
    query: str,
    documents_dir: Optional[str] = None,
    top_k: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Test a knowledge query with a specific provider.
    
    Args:
        provider: LLM provider to use
        query: Query string to test
        documents_dir: Directory containing documents
        top_k: Number of results to retrieve
        verbose: Whether to print detailed output
        
    Returns:
        Query result dictionary
    """
    logger.info(f"Testing query with provider: {provider}")
    logger.info(f"Query: {query}")
    
    start_time = time.time()
    
    try:
        # Initialize the knowledge manager
        manager = LlamaIndexKnowledgeManager(
            documents_dir=documents_dir,
            llm_provider=provider
        )
        
        # Load or create the index
        manager.load_or_create_index()
        
        # Execute the query
        result = manager.query_knowledge(query, top_k=top_k)
        
        # Calculate time
        query_time = time.time() - start_time
        
        # Add execution time to result
        result["execution_time"] = query_time
        
        # Print output if verbose
        if verbose:
            print("\n" + "=" * 50)
            print(f"PROVIDER: {provider}")
            print(f"QUERY: {query}")
            print(f"TIME: {query_time:.2f} seconds")
            print("-" * 50)
            print("\nRESPONSE:")
            print(result["response"])
            print("\nSOURCES:")
            for i, source in enumerate(result["sources"], 1):
                if isinstance(source, dict):
                    source_text = source.get("text", "")
                    source_path = source.get("source", "Unknown")
                    source_score = source.get("score", 0.0)
                    print(f"\n{i}. {source_path}")
                    print(f"   Score: {source_score}")
                    print(f"   {source_text[:200]}...")
                else:
                    print(f"\n{i}. {source[:200]}...")
            print("\n" + "=" * 50)
        
        return result
    
    except Exception as e:
        logger.error(f"Error testing query with {provider}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "execution_time": time.time() - start_time
        }

def benchmark_providers(
    query: str,
    providers: Optional[List[str]] = None,
    documents_dir: Optional[str] = None,
    top_k: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Benchmark multiple providers with the same query.
    
    Args:
        query: Query string to test
        providers: List of providers to test (if None, test all available)
        documents_dir: Directory containing documents
        top_k: Number of results to retrieve
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with results from each provider
    """
    if providers is None:
        providers = list_available_providers()
    
    results = {}
    
    for provider in providers:
        logger.info(f"Benchmarking provider: {provider}")
        result = test_knowledge_query(
            provider=provider,
            query=query,
            documents_dir=documents_dir,
            top_k=top_k,
            verbose=verbose
        )
        results[provider] = result
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Query: {query}")
    print("-" * 60)
    for provider, result in results.items():
        error = result.get("error")
        time_taken = result.get("execution_time", 0)
        if error:
            print(f"{provider}: ERROR - {error}")
        else:
            source_count = len(result.get("sources", []))
            print(f"{provider}: {time_taken:.2f}s, {source_count} sources")
    print("=" * 60)
    
    return results

def print_config():
    """Print current configuration settings."""
    settings = get_all_settings()
    
    print("\n" + "=" * 60)
    print("CURRENT CONFIGURATION")
    print("=" * 60)
    
    # Print LLM settings
    llm_settings = settings.get("llm", {})
    print(f"LLM Provider: {os.environ.get('LLAMA_INDEX_LLM_PROVIDER', 'openai')}")
    print(f"OpenAI Model: {os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')}")
    print(f"OpenAI API Key: {'Set' if os.environ.get('OPENAI_API_KEY') else 'Not Set'}")
    print(f"Anthropic Model: {os.environ.get('ANTHROPIC_MODEL', 'claude-3-sonnet-20240229')}")
    print(f"Anthropic API Key: {'Set' if os.environ.get('ANTHROPIC_API_KEY') else 'Not Set'}")
    
    # Print document settings
    doc_settings = settings.get("document", {})
    print(f"Chunk Size: {doc_settings.get('chunk_size', 500)}")
    print(f"Chunk Overlap: {doc_settings.get('chunk_overlap', 50)}")
    
    # Print path settings
    path_settings = settings.get("paths", {})
    print(f"Documents Directory: {path_settings.get('books_dir', 'Not Set')}")
    print(f"Index Directory: {path_settings.get('index_dir', 'Not Set')}")
    
    # Print retrieval settings
    retrieval_settings = settings.get("retrieval", {})
    print(f"Default Top K: {retrieval_settings.get('top_k', 5)}")
    print(f"Similarity Cutoff: {retrieval_settings.get('similarity_cutoff', 0.7)}")
    
    print("=" * 60)

def example_queries() -> List[str]:
    """Return a list of example educational queries for testing."""
    return [
        "What are effective strategies for teaching fractions?",
        "How do I manage a classroom with diverse learning needs?",
        "What are signs of dyscalculia in second-grade students?",
        "How can I incorporate technology in elementary math education?",
        "What are best practices for formative assessment?"
    ]

def interactive_test():
    """Interactive testing mode allowing user to input queries."""
    print("\n===== LlamaIndex Interactive Testing Mode =====\n")
    
    # Print available providers
    available = list_available_providers()
    print(f"Available providers: {', '.join(available)}")
    
    # Choose provider
    while True:
        provider = input(f"\nSelect provider ({'/'.join(available)}): ").strip().lower()
        if provider in available:
            break
        print(f"Invalid provider. Choose from: {', '.join(available)}")
    
    # Choose document directory
    settings = get_all_settings()
    default_dir = settings["paths"]["books_dir"]
    doc_dir = input(f"\nDocument directory [{default_dir}]: ").strip()
    if not doc_dir:
        doc_dir = default_dir
    
    # Set top_k
    default_top_k = settings["retrieval"]["top_k"]
    top_k_input = input(f"\nTop K results [{default_top_k}]: ").strip()
    top_k = int(top_k_input) if top_k_input and top_k_input.isdigit() else default_top_k
    
    print("\n" + "=" * 60)
    print(f"Starting interactive test with provider: {provider}")
    print(f"Document directory: {doc_dir}")
    print(f"Top K: {top_k}")
    print("=" * 60)
    
    # Initialize the knowledge manager
    manager = LlamaIndexKnowledgeManager(
        documents_dir=doc_dir,
        llm_provider=provider
    )
    
    # Load or create the index
    manager.load_or_create_index()
    
    # Example queries
    examples = example_queries()
    print("\nExample queries:")
    for i, example in enumerate(examples, 1):
        print(f"{i}. {example}")
    
    # Interactive loop
    while True:
        print("\n" + "-" * 60)
        query_input = input("\nEnter query (or 'exit' to quit, 'clear' to clear cache): ").strip()
        
        if query_input.lower() == 'exit':
            break
        elif query_input.lower() == 'clear':
            manager.clear_cache()
            print("Cache cleared.")
            continue
        elif query_input.isdigit() and 1 <= int(query_input) <= len(examples):
            # Use an example query
            query = examples[int(query_input) - 1]
        else:
            query = query_input
        
        if not query:
            continue
        
        print(f"\nQuerying: {query}")
        start_time = time.time()
        
        try:
            result = manager.query_knowledge(query, top_k=top_k)
            query_time = time.time() - start_time
            
            print(f"\nResponse (in {query_time:.2f} seconds):")
            print(result["response"])
            
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                if isinstance(source, dict):
                    source_text = source.get("text", "")
                    source_path = source.get("source", "Unknown")
                    source_score = source.get("score", 0.0)
                    print(f"\n{i}. {os.path.basename(source_path)}")
                    print(f"   Score: {source_score}")
                    print(f"   {source_text[:200]}...")
                else:
                    print(f"\n{i}. {source[:200]}...")
            
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nExiting interactive test.")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test LlamaIndex integration")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--query", "-q", type=str, help="Query string to test")
    group.add_argument("--interactive", "-i", action="store_true", help="Interactive testing mode")
    group.add_argument("--benchmark", "-b", action="store_true", help="Benchmark all available providers")
    group.add_argument("--config", "-c", action="store_true", help="Print current configuration")
    
    parser.add_argument("--provider", "-p", type=str, choices=["openai", "anthropic", "local"], 
                        help="LLM provider to use")
    parser.add_argument("--documents-dir", "-d", type=str, help="Directory containing documents")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to retrieve")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed output")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.config:
        print_config()
        return
    
    if args.interactive:
        interactive_test()
        return
    
    if args.benchmark:
        query = args.query or example_queries()[0]
        benchmark_providers(
            query=query,
            documents_dir=args.documents_dir,
            top_k=args.top_k,
            verbose=args.verbose
        )
        return
    
    if args.query:
        provider = args.provider or os.environ.get("LLAMA_INDEX_LLM_PROVIDER") or "openai"
        test_knowledge_query(
            provider=provider,
            query=args.query,
            documents_dir=args.documents_dir,
            top_k=args.top_k,
            verbose=True  # Always verbose for single query
        )
        return
    
    # If no action specified, print help
    print("No action specified. Use one of: --query, --interactive, --benchmark, or --config")
    print("Use -h or --help for more information")

if __name__ == "__main__":
    main() 