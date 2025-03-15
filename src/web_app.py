"""Streamlit web interface for the teaching assistant."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import threading
import traceback

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
    HAS_STREAMLIT = True
except ImportError as e:
    logger.error(f"Streamlit not available: {e}")
    HAS_STREAMLIT = False
    
try:
    from dotenv import load_dotenv
    # Load environment variables
    load_dotenv()
except ImportError as e:
    logger.error(f"python-dotenv not available: {e}")

try:
    import torch
    HAS_TORCH = True
except ImportError as e:
    logger.error(f"PyTorch not available: {e}")
    HAS_TORCH = False

try:
    import spacy
    HAS_SPACY = True
    # Initialize spaCy model at module level
    try:
        if not spacy.util.is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully loaded spaCy model")
    except Exception as e:
        logger.error(f"Failed to load spaCy model: {e}")
        nlp = None
except ImportError as e:
    logger.error(f"spaCy not available: {e}")
    HAS_SPACY = False
    nlp = None

# Import local modules with error handling
try:
    from llm.dspy.handler import EnhancedDSPyLLMInterface
    HAS_LLM = True
except ImportError as e:
    logger.error(f"DSPy handler not available: {e}")
    HAS_LLM = False
    
try:
    from knowledge_base.document_indexer import DocumentIndexer
    HAS_INDEXER = True
except ImportError as e:
    logger.error(f"Document indexer not available: {e}")
    HAS_INDEXER = False
    
try:
    from metrics.automated_metrics import AutomatedMetricsEvaluator
    HAS_METRICS = True
except ImportError as e:
    logger.error(f"Metrics evaluator not available: {e}")
    HAS_METRICS = False

# Initialize components in the main thread
_thread_local = threading.local()

def init_torch():
    """Initialize PyTorch settings."""
    if not HAS_TORCH:
        logger.warning("PyTorch not available, skipping initialization")
        return False
        
    try:
        # Set PyTorch settings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Use pytorch device_name: {device}")
        torch.set_num_threads(1)  # Prevent threading issues
        return True
    except Exception as e:
        logger.error(f"Failed to initialize PyTorch: {e}")
        return False

@st.cache_resource(show_spinner=False)
def get_llm():
    """Get or initialize LLM interface."""
    if not HAS_LLM:
        logger.warning("LLM interface not available")
        return None
        
    try:
        init_torch()
        llm = EnhancedDSPyLLMInterface()
        logger.info("Initialized DSPy LLM interface")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize DSPy interface: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_indexer():
    """Get or initialize document indexer."""
    if not HAS_INDEXER:
        logger.warning("Document indexer not available")
        return None
        
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
    if not HAS_METRICS:
        logger.warning("Metrics evaluator not available")
        return None
        
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
        
        # We can continue even if some components aren't available
        logger.info(f"Components initialized: LLM: {llm is not None}, Indexer: {indexer is not None}, Evaluator: {evaluator is not None}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return False

# Page configuration
if HAS_STREAMLIT:
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
    if not HAS_STREAMLIT:
        return False
        
    if 'initialized' not in st.session_state:
        # Initialize chat history and settings
        st.session_state.messages = []
        st.session_state.settings = {
            "grade_level": 5,
            "subject": "General",
            "learning_style": "Visual"
        }
        st.session_state.initialized = True
        st.session_state.error = None
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
        if HAS_STREAMLIT:
            st.error("Failed to save uploaded file")
        return None

def display_metrics(metrics):
    """Display evaluation metrics in a organized layout."""
    if not HAS_STREAMLIT or not metrics:
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
        st.markdown("\n".join([f"- {item}" for item in metrics.feedback]))

def display_chat_history():
    """Display chat history with metrics."""
    if not HAS_STREAMLIT:
        return
        
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
            if "metrics" in msg and msg["metrics"]:
                with st.expander("View Response Metrics", expanded=False):
                    display_metrics(msg["metrics"])

def handle_error(error_msg, exception=None):
    """Handle errors gracefully."""
    if exception:
        logger.error(f"{error_msg}: {str(exception)}")
        logger.error(traceback.format_exc())
    else:
        logger.error(error_msg)
        
    if HAS_STREAMLIT:
        st.session_state.error = error_msg
        st.error(error_msg)

def process_user_query(query: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
    """Process user query and generate response with metrics."""
    if not query:
        return {"success": False, "message": "Empty query"}
        
    try:
        # Get components
        llm = get_llm()
        evaluator = get_evaluator()
        
        if not llm:
            return {"success": False, "message": "LLM interface not available"}
        
        # Extract settings
        settings = st.session_state.settings if HAS_STREAMLIT else {"grade_level": 5}
        grade_level = int(settings.get("grade_level", 5))
        
        # Generate response
        response = llm.generate_response(
            prompt=query,
            context=context,
            metadata=settings
        )
        
        # Evaluate response if metrics available
        metrics = None
        if evaluator:
            metrics = evaluator.evaluate_response(
                response=response,
                prompt=query,
                grade_level=grade_level,
                context=context
            )
        
        return {
            "success": True,
            "response": response,
            "metrics": metrics
        }
        
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        handle_error(error_msg, e)
        return {"success": False, "message": error_msg}

def main():
    """Main function for the Streamlit app."""
    if not HAS_STREAMLIT:
        logger.error("Streamlit not available, cannot run web interface")
        return
    
    # Initialize
    initialize_session_state()
    initialize_components()
    
    # Display title
    st.title("💬 Teaching Assistant")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Teaching Context")
        
        st.subheader("Student")
        grade_level = st.slider("Grade Level", min_value=1, max_value=12, value=st.session_state.settings["grade_level"])
        subject = st.selectbox(
            "Subject", 
            ["General", "Math", "Science", "History", "Language Arts", "Social Studies"],
            index=0
        )
        learning_style = st.selectbox(
            "Learning Style",
            ["Visual", "Auditory", "Reading/Writing", "Kinesthetic", "Mixed"],
            index=0
        )
        
        # Update settings in session state
        st.session_state.settings = {
            "grade_level": grade_level,
            "subject": subject,
            "learning_style": learning_style
        }
        
        # Document management
        st.subheader("Knowledge Base")
        uploaded_file = st.file_uploader("Add document", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            if st.button("Add to Knowledge Base"):
                # Save file and add to indexer
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    indexer = get_indexer()
                    if indexer:
                        try:
                            with open(file_path, "r") as f:
                                content = f.read()
                            
                            # Add to index
                            doc_id = indexer.add_document(
                                content=content,
                                metadata={
                                    "title": uploaded_file.name,
                                    "subject": subject,
                                    "grade_level": grade_level
                                }
                            )
                            st.success(f"Document added with ID: {doc_id}")
                        except Exception as e:
                            handle_error(f"Failed to process document: {str(e)}", e)
                    else:
                        st.warning("Document indexer not available")
        
        # Status indicators
        st.subheader("System Status")
        llm_status = "✅ Available" if get_llm() else "❌ Unavailable"
        indexer_status = "✅ Available" if get_indexer() else "❌ Unavailable"
        metrics_status = "✅ Available" if get_evaluator() else "❌ Unavailable"
        
        st.text(f"LLM: {llm_status}")
        st.text(f"Indexer: {indexer_status}")
        st.text(f"Metrics: {metrics_status}")
    
    # Display chat history
    display_chat_history()
    
    # Display error if any
    if HAS_STREAMLIT and hasattr(st.session_state, 'error') and st.session_state.error:
        st.error(st.session_state.error)
    
    # Chat input
    if prompt := st.chat_input("Ask the teaching assistant..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Clear any previous errors
        st.session_state.error = None
        
        # Process query
        indexer = get_indexer()
        context = None
        
        # Search for relevant documents if indexer available
        if indexer:
            try:
                results = indexer.search(
                    query=prompt, 
                    filters={"subject": st.session_state.settings["subject"]}
                )
                if results:
                    context = [r["content"] for r in results]
            except Exception as e:
                handle_error(f"Search error: {str(e)}", e)
        
        # Process the query
        result = process_user_query(prompt, context)
        
        if result["success"]:
            # Add assistant message to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["response"],
                "metrics": result["metrics"]
            })
        else:
            # Handle error
            error_msg = result.get("message", "Unknown error")
            handle_error(error_msg)
            
            # Add fallback response
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I'm sorry, I encountered an error: {error_msg}",
                "metrics": None
            })
        
        # Rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    if HAS_STREAMLIT:
        main()
    else:
        print("ERROR: Streamlit not available, cannot run web interface")
        sys.exit(1) 