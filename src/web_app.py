"""Streamlit web interface for the teaching assistant."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import threading
import torch
import asyncio

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
    import dspy
    import spacy
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

# Initialize spaCy model at module level
try:
    if not spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    nlp = None

# Initialize components in the main thread
_thread_local = threading.local()

def init_torch():
    """Initialize PyTorch settings."""
    try:
        # Set PyTorch settings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)  # Prevent threading issues
        logger.info(f"PyTorch initialized with device: {device}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PyTorch: {e}")
        return False

def setup_async_event_loop():
    """Set up the asyncio event loop for the current thread."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return True
    except Exception as e:
        logger.error(f"Failed to set up event loop: {e}")
        return False

@st.cache_resource(show_spinner=False)
def get_llm():
    """Get or initialize LLM interface."""
    try:
        init_torch()
        setup_async_event_loop()
        llm = EnhancedDSPyLLMInterface()
        logger.info("Initialized DSPy LLM interface")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize DSPy interface: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_indexer():
    """Get or initialize document indexer."""
    try:
        init_torch()
        indexer = DocumentIndexer()
        logger.info("Initialized DocumentIndexer")
        return indexer
    except Exception as e:
        logger.error(f"Failed to initialize document indexer: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_evaluator():
    """Get or initialize metrics evaluator."""
    try:
        evaluator = AutomatedMetricsEvaluator()
        logger.info("Initialized AutomatedMetricsEvaluator")
        return evaluator
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return None

@st.cache_resource
def initialize_components():
    """Initialize all components with caching."""
    try:
        init_torch()
        llm = get_llm()
        indexer = get_indexer()
        evaluator = get_evaluator()
        return all([llm, indexer, evaluator])
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

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
    .error-message {
        color: #ff4b4b;
        padding: 10px;
        border-radius: 5px;
        background-color: #ffe5e5;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'initialized' not in st.session_state:
        # Initialize chat history and settings
        st.session_state.messages = []
        st.session_state.settings = {
            "grade_level": 5,
            "subject": "General",
            "learning_style": "Visual"
        }
        st.session_state.initialized = True
        logger.info("Session state initialized successfully")
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
    if not metrics:
        return
        
    st.markdown("#### Response Evaluation")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Clarity", f"{getattr(metrics, 'clarity', 0):.2f}")
        st.metric("Engagement", f"{getattr(metrics, 'engagement', 0):.2f}")
    
    with col2:
        st.metric("Pedagogical Approach", f"{getattr(metrics, 'pedagogical_approach', 0):.2f}")
        st.metric("Emotional Support", f"{getattr(metrics, 'emotional_support', 0):.2f}")
    
    with col3:
        st.metric("Content Accuracy", f"{getattr(metrics, 'content_accuracy', 0):.2f}")
        st.metric("Age Appropriateness", f"{getattr(metrics, 'age_appropriateness', 0):.2f}")
    
    if hasattr(metrics, 'feedback') and metrics.feedback:
        st.markdown("##### Detailed Feedback")
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
                with st.expander("View Response Evaluation", expanded=False):
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
                    
                    indexer = get_indexer()
                    if indexer:
                        doc_id = indexer.add_document(
                            content,
                            metadata={
                                "subject": subject,
                                "grade_level": grade_level,
                                "type": uploaded_file.type
                            }
                        )
                        st.sidebar.success("Document added successfully!")
                    else:
                        st.sidebar.error("Document indexer not available")
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
    
    # Initialize components with caching
    if not initialize_components():
        st.error("Failed to initialize components. Please check the logs and try reloading the page.")
        return
    
    # Get thread-local components
    llm = get_llm()
    indexer = get_indexer()
    evaluator = get_evaluator()
    
    if not all([llm, indexer, evaluator]):
        st.error("Some components are not available. Please check the logs and try reloading the page.")
        if nlp is None:
            st.error("SpaCy model is not available. Some features may be limited.")
        return
    
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
                    results = indexer.search(
                        prompt,
                        filters={"subject": st.session_state.settings["subject"]}
                    )
                    context = [r["content"] for r in results]
                except Exception as e:
                    logger.warning(f"Failed to search knowledge base: {e}")
                    st.warning("Could not search knowledge base. Proceeding without context.")
                
                # Generate response
                with st.spinner("Generating response..."):
                    try:
                        response = llm.generate_response(
                            prompt,
                            context=context,
                            metadata=st.session_state.settings
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate response: {e}")
                        st.error("Failed to generate response. Please try again.")
                        return
                
                # Evaluate response if evaluator is available
                metrics = None
                if evaluator:
                    try:
                        metrics = evaluator.evaluate_response(
                            response,
                            prompt,
                            st.session_state.settings["grade_level"],
                            context
                        )
                    except Exception as e:
                        logger.warning(f"Failed to evaluate response: {e}")
                        st.warning("Could not evaluate response. Proceeding without metrics.")
                
                # Add assistant message with metrics if available
                message = {
                    "role": "assistant",
                    "content": response
                }
                if metrics:
                    message["metrics"] = metrics
                
                st.session_state.messages.append(message)
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error in chat interaction: {e}")
                st.error("An error occurred. Please try again.")

if __name__ == "__main__":
    main() 