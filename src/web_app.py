"""Streamlit web interface for the teaching assistant."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

try:
    import streamlit as st
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import required packages: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Import local modules
try:
    from llm.dspy.handler import EnhancedDSPyLLMInterface
    from knowledge_base.document_indexer import DocumentIndexer
    from metrics.automated_metrics import AutomatedMetricsEvaluator
except ImportError as e:
    logger.error(f"Failed to import local modules: {e}")
    sys.exit(1)

# Page configuration
st.set_page_config(
    page_title="Teaching Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .feedback-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .user-message {
        background-color: #e6f3ff;
        margin-left: 20%;
    }
    .assistant-message {
        background-color: #f0f2f6;
        margin-right: 20%;
    }
    .metrics-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        # Initialize components
        try:
            st.session_state.llm = EnhancedDSPyLLMInterface()
            st.session_state.indexer = DocumentIndexer()
            st.session_state.evaluator = AutomatedMetricsEvaluator()
            
            # Initialize chat history
            st.session_state.messages = []
            
            # Initialize settings
            st.session_state.settings = {
                "grade_level": 5,
                "subject": "General",
                "learning_style": "Visual"
            }
            
            st.session_state.initialized = True
            logger.info("Session state initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize session state: {e}")
            st.error("Failed to initialize application. Please check the logs.")
            return False
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

def display_metrics(metrics):
    """Display evaluation metrics in a organized layout."""
    st.markdown("#### Response Evaluation")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Clarity", f"{metrics.clarity:.2f}")
        st.metric("Engagement", f"{metrics.engagement:.2f}")
    
    with col2:
        st.metric("Pedagogical Approach", f"{metrics.pedagogical_approach:.2f}")
        st.metric("Emotional Support", f"{metrics.emotional_support:.2f}")
    
    with col3:
        st.metric("Content Accuracy", f"{metrics.content_accuracy:.2f}")
        st.metric("Age Appropriateness", f"{metrics.age_appropriateness:.2f}")
    
    if metrics.feedback:
        with st.expander("View Detailed Feedback", expanded=True):
            for item in metrics.feedback:
                st.markdown(f"- {item}")

def display_chat_history():
    """Display chat history with metrics."""
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        
        # Style messages differently based on role
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">👤 You: {content}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 Assistant: {content}</div>', 
                       unsafe_allow_html=True)
            
            # Display metrics if available
            if "metrics" in msg:
                with st.expander("View Response Evaluation"):
                    display_metrics(msg["metrics"])

def sidebar_settings():
    """Configure settings in the sidebar."""
    st.sidebar.title("Teaching Assistant Settings")
    
    # Student Profile
    st.sidebar.header("Student Profile")
    grade_level = st.sidebar.slider("Grade Level", 1, 12, 
                                  st.session_state.settings["grade_level"])
    
    subject = st.sidebar.selectbox("Subject",
        ["General", "Mathematics", "Science", "Language Arts", "Social Studies"],
        index=0 if st.session_state.settings["subject"] == "General" else None
    )
    
    learning_style = st.sidebar.selectbox("Learning Style",
        ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
        index=0 if st.session_state.settings["learning_style"] == "Visual" else None
    )
    
    # Update settings
    st.session_state.settings.update({
        "grade_level": grade_level,
        "subject": subject,
        "learning_style": learning_style
    })
    
    # Document Upload
    st.sidebar.header("Knowledge Base")
    uploaded_file = st.sidebar.file_uploader("Upload Teaching Material",
                                           type=["txt", "md", "pdf"])
    
    if uploaded_file:
        if st.sidebar.button("Add to Knowledge Base"):
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                try:
                    with open(file_path, "r") as f:
                        content = f.read()
                    
                    doc_id = st.session_state.indexer.add_document(
                        content,
                        metadata={
                            "subject": subject,
                            "grade_level": grade_level,
                            "type": uploaded_file.type
                        }
                    )
                    st.sidebar.success("Document added successfully!")
                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    st.sidebar.error("Failed to process document")

def main():
    """Main application."""
    if not initialize_session_state():
        return
    
    # Configure sidebar
    sidebar_settings()
    
    # Main content
    st.title("📚 Teaching Assistant")
    st.markdown("""
    Welcome to the Teaching Assistant! I'm here to help you with your teaching needs.
    Configure the settings in the sidebar and start chatting below.
    """)
    
    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        display_chat_history()
        
        # Chat input
        if prompt := st.chat_input("Ask a question or enter a topic..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            try:
                # Get context from knowledge base
                context = []
                try:
                    results = st.session_state.indexer.search(
                        prompt,
                        filters={"subject": st.session_state.settings["subject"]}
                    )
                    context = [r["content"] for r in results]
                except Exception as e:
                    logger.warning(f"Failed to search knowledge base: {e}")
                
                # Generate response
                with st.spinner("Generating response..."):
                    response = st.session_state.llm.generate_response(
                        prompt,
                        context=context,
                        metadata=st.session_state.settings
                    )
                
                # Evaluate response
                try:
                    metrics = st.session_state.evaluator.evaluate_response(
                        response,
                        prompt,
                        st.session_state.settings["grade_level"],
                        context
                    )
                    
                    # Add assistant message with metrics
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "metrics": metrics
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to evaluate response: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                # Rerun to update display
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error("Failed to generate response. Please try again.")

if __name__ == "__main__":
    main() 