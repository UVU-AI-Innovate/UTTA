"""Test the setup and dependencies."""
import sys
import streamlit as st
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def test_environment():
    """Test if all dependencies are working."""
    print(f"Python version: {sys.version}")
    
    # Test Streamlit
    print("\nTesting Streamlit...")
    try:
        st.runtime
        print("✓ Streamlit is working")
    except Exception as e:
        print(f"✗ Streamlit error: {e}")
    
    # Test Ollama
    print("\nTesting Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/version")
        print(f"✓ Ollama is running: {response.json()}")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
    
    # Test embeddings
    print("\nTesting embeddings...")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(["Test sentence"])
        print(f"✓ Embeddings working, shape: {embeddings.shape}")
    except Exception as e:
        print(f"✗ Embeddings error: {e}")
    
    # Test FAISS
    print("\nTesting FAISS...")
    try:
        index = faiss.IndexFlatL2(384)
        print("✓ FAISS is working")
    except Exception as e:
        print(f"✗ FAISS error: {e}")

if __name__ == "__main__":
    test_environment() 