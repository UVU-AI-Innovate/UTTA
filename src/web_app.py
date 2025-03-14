"""Streamlit web interface for the teaching assistant."""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src directory to path
src_dir = Path(__file__).parent.parent
sys.path.append(str(src_dir))

# Import dependencies with error handling
try:
    import streamlit as st
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    logger.error("Please install required packages: pip install streamlit python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Import local modules with error handling
try:
    from llm.dspy.handler import EnhancedDSPyLLMInterface
    from knowledge_base.document_indexer import DocumentIndexer
    from metrics.automated_metrics import AutomatedMetricsEvaluator
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    logger.error("Please ensure all project files are in the correct location")
    sys.exit(1)

# Initialize session state
def init_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'llm_interface' not in st.session_state:
        try:
            st.session_state.llm_interface = EnhancedDSPyLLMInterface()
        except Exception as e:
            logger.error(f"Failed to initialize LLM interface: {e}")
            st.error("Failed to initialize LLM interface. Please check your API keys and configuration.")
            return False
    if 'document_indexer' not in st.session_state:
        try:
            st.session_state.document_indexer = DocumentIndexer()
        except Exception as e:
            logger.error(f"Failed to initialize document indexer: {e}")
            st.warning("Document indexer initialization failed. Some features may be limited.")
    if 'metrics_evaluator' not in st.session_state:
        try:
            st.session_state.metrics_evaluator = AutomatedMetricsEvaluator()
        except Exception as e:
            logger.error(f"Failed to initialize metrics evaluator: {e}")
            st.warning("Metrics evaluator initialization failed. Evaluation features will be limited.")
    return True

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file and return its path."""
    try:
        save_dir = Path("data/uploads")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = save_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        st.error("Failed to save uploaded file")
        return None

def main():
    """Main web application."""
    st.set_page_config(
        page_title="Teaching Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Teaching Assistant")
    
    # Initialize session state
    if not init_session_state():
        st.error("Failed to initialize application. Please check the logs and configuration.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Student profile
        st.subheader("Student Profile")
        grade_level = st.selectbox("Grade Level", range(1, 13))
        learning_style = st.selectbox(
            "Learning Style",
            ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
        )
        
        # Context settings
        st.subheader("Teaching Context")
        subject = st.selectbox(
            "Subject",
            ["Mathematics", "Science", "Language Arts", "Social Studies"]
        )
        topic = st.text_input("Topic")
        
        # Document upload
        st.subheader("Knowledge Base")
        uploaded_file = st.file_uploader("Upload Teaching Material")
        if uploaded_file:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    metadata = {
                        "subject": subject,
                        "grade_level": grade_level,
                        "type": uploaded_file.type
                    }
                    st.session_state.document_indexer.add_document(content, metadata)
                    st.success("Document added to knowledge base")
                except Exception as e:
                    logger.error(f"Failed to process uploaded document: {e}")
                    st.error("Failed to process document")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            with st.chat_message(role):
                st.write(content)
                
                # Show evaluation for assistant responses
                if role == "assistant" and "evaluation" in message:
                    with st.expander("Response Evaluation"):
                        metrics = message["evaluation"]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Clarity", f"{metrics['clarity']:.2f}")
                            st.metric("Engagement", f"{metrics['engagement']:.2f}")
                            st.metric("Pedagogical Approach", f"{metrics['pedagogical_approach']:.2f}")
                        with col2:
                            st.metric("Emotional Support", f"{metrics['emotional_support']:.2f}")
                            st.metric("Content Accuracy", f"{metrics['content_accuracy']:.2f}")
                            st.metric("Age Appropriateness", f"{metrics['age_appropriateness']:.2f}")
    
    # Chat input
    if prompt := st.chat_input():
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get context from knowledge base
        context = []
        try:
            results = st.session_state.document_indexer.search(
                prompt,
                filters={"subject": subject} if subject else None
            )
            context = [r["content"] for r in results]
        except Exception as e:
            logger.warning(f"Failed to search knowledge base: {e}")
        
        # Generate response
        try:
            response = st.session_state.llm_interface.generate_response(
                prompt,
                context=context,
                metadata={
                    "grade_level": grade_level,
                    "learning_style": learning_style,
                    "subject": subject,
                    "topic": topic
                }
            )
            
            # Evaluate response
            try:
                metrics = st.session_state.metrics_evaluator.evaluate_response(
                    response,
                    prompt,
                    grade_level,
                    context
                )
                
                # Add assistant message with evaluation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "evaluation": metrics.__dict__
                })
            except Exception as e:
                logger.warning(f"Failed to evaluate response: {e}")
                # Add assistant message without evaluation
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            st.error("Failed to generate response. Please try again.")
            
        # Rerun to update display
        st.rerun()

if __name__ == "__main__":
    main() 