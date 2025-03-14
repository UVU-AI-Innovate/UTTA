"""
Streamlit web application for the Teacher Training Simulator.
Provides an interactive interface for teachers to practice and analyze teaching scenarios.
Uses DSPy for efficient and reliable LLM interactions.
"""

import streamlit as st

# Set page configuration first - this must be the first Streamlit command
st.set_page_config(
    page_title="Teacher Training Simulator",
    page_icon="👨‍🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from dotenv import load_dotenv
import subprocess
import getpass
import logging
import random

# Import our DSPy adapter and implementations
from dspy_adapter import create_llm_interface, EnhancedLLMInterface
from dspy_llm_handler import PedagogicalLanguageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file
load_dotenv()

class WebInterface:
    def __init__(self):
        """Initialize the web interface and session state with defaults."""
        # Initialize session state variables
        if 'llm_interface' not in st.session_state:
            st.session_state.llm_interface = None
        if 'pedagogical_processor' not in st.session_state:
            st.session_state.pedagogical_processor = None
        if 'scenario' not in st.session_state:
            st.session_state.scenario = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'analysis' not in st.session_state:
            st.session_state.analysis = None
        if 'strategies' not in st.session_state:
            st.session_state.strategies = []
        if 'teacher_feedback' not in st.session_state:
            st.session_state.teacher_feedback = None
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = None
        if 'student_profile' not in st.session_state:
            st.session_state.student_profile = None
        if 'reflection' not in st.session_state:
            st.session_state.reflection = None
        if 'app_mode' not in st.session_state:
            st.session_state.app_mode = "Chat Interface"
        if 'model_name' not in st.session_state:
            st.session_state.model_name = "gpt-3.5-turbo"
        if 'classroom_management_strategies' not in st.session_state:
            st.session_state.classroom_management_strategies = []
        
        # Auto-initialize the LLM on startup with default model
        if not st.session_state.llm_interface:
            try:
                self.initialize_llm(st.session_state.model_name)
            except Exception as e:
                logging.error(f"Error initializing LLM: {e}", exc_info=True)

    # Dictionary of classroom management strategies from educational books
    classroom_management_books = {
        "Teach Like a Champion (Doug Lemov)": [
            {"name": "No Opt Out", "description": "Turn a student's 'I don't know' into a success by ensuring they answer the question correctly with support."},
            {"name": "Right Is Right", "description": "Set and defend high standards for correct answers, ensuring precision and accuracy."},
            {"name": "Cold Call", "description": "Call on students regardless of whether they have raised their hands, strategically engaging all students."},
            {"name": "Wait Time", "description": "Allow students time to think after asking a question before calling on someone to answer."},
            {"name": "Stretch It", "description": "Extend student responses by asking follow-up questions that require deeper thinking."}
        ],
        "Classroom Management That Works (Robert Marzano)": [
            {"name": "Establishing Rules and Procedures", "description": "Create clear expectations for behavior and routines in the classroom."},
            {"name": "Disciplinary Interventions", "description": "Use a graduated system of responses to disruptive behavior."},
            {"name": "Teacher-Student Relationships", "description": "Build positive connections with students based on appropriate levels of dominance and cooperation."},
            {"name": "Mental Set", "description": "Maintain an objective, business-like attitude about behavior while remaining emotionally aware."},
            {"name": "Student Responsibility", "description": "Provide opportunities for students to take on meaningful responsibility for classroom management."}
        ],
        "The First Days of School (Harry Wong)": [
            {"name": "Classroom Procedures", "description": "Teach specific procedures for every classroom activity and transition."},
            {"name": "Classroom Routines", "description": "Establish consistent, repeated patterns of behavior for daily activities."},
            {"name": "Positive Expectations", "description": "Communicate belief in students' ability to succeed academically and behaviorally."},
            {"name": "Classroom Organization", "description": "Structure the physical environment and materials to optimize learning and minimize disruption."},
            {"name": "Professional Behavior", "description": "Model and require respectful, appropriate behavior at all times."}
        ],
        "Conscious Classroom Management (Rick Smith)": [
            {"name": "Prevention Through Connection", "description": "Create personal connections to prevent behavior problems before they begin."},
            {"name": "Using Body Language", "description": "Leverage non-verbal communication to manage the classroom effectively."},
            {"name": "Collaborative Problem-Solving", "description": "Work with students to find solutions rather than imposing them."},
            {"name": "Managing Moments of Escalation", "description": "Use specific strategies to de-escalate tension and conflict."},
            {"name": "Creating Community", "description": "Build a sense of belonging and mutual respect within the classroom."}
        ]
    }

    def setup_page(self):
        """Set up the page with title, favicon, and basic layout."""
        st.markdown("""
        <style>
            /* Global styling */
            :root {
                --primary-color: #4F46E5;
                --primary-light: #EEF2FF;
                --secondary-color: #10B981;
                --secondary-light: #ECFDF5;
                --neutral-50: #F9FAFB;
                --neutral-100: #F3F4F6;
                --neutral-200: #E5E7EB;
                --neutral-300: #D1D5DB;
                --neutral-700: #374151;
                --neutral-800: #1F2937;
                --neutral-900: #111827;
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --radius-sm: 0.25rem;
                --radius: 0.5rem;
                --radius-md: 0.75rem;
                --radius-lg: 1rem;
            }
            
            /* Main container styling */
            .main .block-container {
                padding-top: 1rem;
                padding-bottom: 1rem;
                max-width: 1200px;
            }
            
            /* App container */
            .app-container {
                display: flex;
                flex-direction: column;
                min-height: calc(100vh - 80px);
                border-radius: var(--radius);
                background-color: white;
                box-shadow: var(--shadow);
                overflow: hidden;
            }
            
            /* Header styling */
            .app-header {
                padding: 1rem 1.5rem;
                background: linear-gradient(135deg, var(--primary-color), #6366F1);
                color: white;
                border-radius: var(--radius) var(--radius) 0 0;
                margin-bottom: 1rem;
            }
            
            .app-header h1 {
                font-size: 1.75rem;
                margin: 0;
                font-weight: 600;
                color: white;
                text-align: center;
            }
            
            .app-header p {
                margin: 0.5rem 0 0;
                opacity: 0.9;
                font-size: 1rem;
                text-align: center;
            }
            
            h1, h2, h3, h4 {
                color: var(--neutral-900);
                margin-top: 0;
            }
            
            h2 {
                font-size: 1.5rem;
                font-weight: 600;
            }
            
            h3 {
                font-size: 1.25rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
            }
            
            h4 {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                color: var(--neutral-700);
            }
            
            /* Chat container */
            .chat-container {
                border-radius: var(--radius);
                padding: 1rem;
                margin-bottom: 1rem;
                height: 550px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                background-color: var(--neutral-50);
                border: 1px solid var(--neutral-200);
            }
            
            /* Custom chat bubbles */
            .stChatMessage [data-testid="chatAvatarIcon-user"] {
                background-color: var(--primary-color) !important;
            }
            
            .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
                background-color: var(--secondary-color) !important;
            }
            
            .stChatMessage [data-testid="stChatMessageContent"] {
                border-radius: var(--radius) !important;
                padding: 0.75rem 1rem !important;
                box-shadow: var(--shadow-sm) !important;
            }
            
            .stChatMessage.user [data-testid="stChatMessageContent"] {
                background-color: var(--secondary-light) !important;
                border: 1px solid #D1FAE5 !important;
            }
            
            .stChatMessage.assistant [data-testid="stChatMessageContent"] {
                background-color: var(--primary-light) !important;
                border: 1px solid #DDD6FE !important;
            }
            
            /* Info panel container - simplified */
            .info-panel {
                height: 670px;
                overflow-y: auto;
                padding-right: 10px;
                display: flex;
                flex-direction: column;
                gap: 1rem;
                scrollbar-width: thin;
                scrollbar-color: var(--neutral-300) var(--neutral-100);
            }
            
            .info-panel::-webkit-scrollbar {
                width: 6px;
            }
            
            .info-panel::-webkit-scrollbar-track {
                background: var(--neutral-100);
                border-radius: var(--radius);
            }
            
            .info-panel::-webkit-scrollbar-thumb {
                background-color: var(--neutral-300);
                border-radius: var(--radius);
            }
            
            /* Chat container scrollbar styling */
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: var(--neutral-100);
                border-radius: var(--radius);
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background-color: var(--neutral-300);
                border-radius: var(--radius);
            }
            
            /* Fix for stChatMessage spacing */
            .stChatMessage {
                margin-bottom: 1rem;
            }
            
            /* Fix for chat input positioning */
            .stChatInputContainer {
                position: sticky;
                bottom: 0;
                background-color: white;
                padding: 1rem 0;
                z-index: 100;
                border-top: 1px solid var(--neutral-200);
            }
            
            /* Sidebar styling - simplified */
            [data-testid="stSidebar"] {
                background-color: var(--neutral-50);
                border-right: 1px solid var(--neutral-200);
            }
            
            .sidebar .sidebar-content {
                background-color: var(--neutral-50);
            }
            
            .sidebar .block-container {
                padding-top: 2rem;
            }
            
            /* Sidebar sections */
            .sidebar-section {
                background-color: white;
                border-radius: var(--radius);
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid var(--neutral-200);
                box-shadow: var(--shadow-sm);
            }
            
            .sidebar-title {
                font-size: 1rem;
                font-weight: 600;
                margin-bottom: 0.75rem;
                color: var(--neutral-800);
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            /* Cards for scenario information - simplified */
            .scenario-card {
                border-radius: var(--radius);
                padding: 1.25rem;
                margin-bottom: 1rem;
                background-color: white;
                border: 1px solid var(--neutral-200);
                box-shadow: var(--shadow-sm);
                transition: all 0.2s;
            }
            
            .scenario-header {
                position: sticky;
                top: 0;
                background-color: white;
                z-index: 100;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: var(--radius);
                border: 1px solid var(--neutral-200);
                box-shadow: var(--shadow-sm);
            }
            
            .conversation-header {
                position: sticky;
                top: 0;
                background-color: white;
                z-index: 50;
                padding: 1rem;
                margin-bottom: 1rem;
                border-radius: var(--radius);
                border: 1px solid var(--neutral-200);
                text-align: center;
            }
            
            .scenario-title {
                font-weight: 600;
                color: var(--neutral-900);
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .scenario-title-icon {
                color: var(--primary-color);
            }
            
            /* Better list styling */
            .info-list {
                list-style-type: none;
                padding-left: 0.5rem;
                margin-top: 0.5rem;
            }
            
            .info-list li {
                position: relative;
                padding-left: 1.5rem;
                margin-bottom: 0.5rem;
                line-height: 1.5;
            }
            
            .info-list li:before {
                content: "";
                position: absolute;
                left: 0;
                top: 0.6rem;
                height: 0.5rem;
                width: 0.5rem;
                border-radius: 50%;
                background-color: var(--primary-color);
            }
            
            /* Control panel for buttons - simplified */
            .control-panel {
                display: flex;
                flex-direction: column;
                gap: 0.75rem;
                padding: 1.25rem;
                border-radius: var(--radius);
                border: 1px solid var(--neutral-200);
                background-color: white;
                box-shadow: var(--shadow-sm);
            }
            
            /* Button styling */
            .stButton button {
                width: 100%;
                border-radius: var(--radius-sm);
                padding: 0.5rem 1rem;
                font-weight: 500;
                transition: all 0.2s;
                border: 1px solid var(--neutral-200);
            }
            
            .stButton button:first-child {
                background-color: var(--primary-color);
                color: white;
            }
            
            .stButton button:nth-child(2) {
                background-color: var(--neutral-50);
                color: var(--neutral-800);
            }
            
            .stButton button:hover {
                opacity: 0.9;
                box-shadow: var(--shadow);
            }
        </style>
        """, unsafe_allow_html=True)

    def initialize_llm(self, model_name="gpt-3.5-turbo"):
        """Initialize the LLM interface with specified model."""
        try:
            logging.info(f"Initializing DSPy LLM interface with model: {model_name}")
            
            # Store the model name in session state
            st.session_state.model_name = model_name
            
            # Create the LLM interface
            llm_interface = create_llm_interface(model_name, enhanced=True)
            
            # Explicitly ensure DSPy is configured
            if hasattr(llm_interface, 'dspy_interface') and hasattr(llm_interface.dspy_interface, 'configure_dspy_settings'):
                success = llm_interface.dspy_interface.configure_dspy_settings()
                if not success:
                    st.error("Failed to configure DSPy settings. Using fallback mode.")
                    logging.error("Failed to configure DSPy settings. Using fallback mode.")
                    
            # Store the interface in session state
            st.session_state.llm_interface = llm_interface
            
            # Initialize pedagogical processor
            logging.info(f"Initializing pedagogical processor with model: {model_name}")
            processor = PedagogicalLanguageProcessor(model_name)
            st.session_state.pedagogical_processor = processor
            logging.info("Pedagogical processor set successfully")
            
            # Set the processor in the interface
            if hasattr(llm_interface, 'set_pedagogical_processor'):
                llm_interface.set_pedagogical_processor(processor)
            
            return llm_interface
        except Exception as e:
            st.error(f"Error initializing LLM: {str(e)}")
            logging.error(f"LLM initialization error: {e}", exc_info=True)
            return None

    def create_scenario(self):
        """Create a teaching scenario with specified parameters."""
        st.markdown("## Create a Teaching Scenario")
        
        with st.form(key="scenario_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                subject = st.selectbox(
                    "Subject", 
                    ["Mathematics", "Science", "Language Arts", "Social Studies", "Art", "Music", "Physical Education"]
                )
                
                grade_level = st.selectbox(
                    "Grade Level",
                    ["Kindergarten", "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th", "11th", "12th"]
                )
                
                learning_objectives = st.text_area(
                    "Learning Objectives (one per line)",
                    "Understand fractions as parts of a whole\nCompare fractions with common denominators\nVisualize fractions using diagrams"
                )
            
            with col2:
                learning_styles = st.multiselect(
                    "Student Learning Styles",
                    ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"],
                    ["Visual", "Kinesthetic"]
                )
                
                challenges = st.multiselect(
                    "Student Challenges",
                    ["Attention span", "Abstract concepts", "Reading comprehension", "Math anxiety", 
                     "Organization", "Memory", "Processing speed", "English language learner"],
                    ["Abstract concepts"]
                )
                
                strengths = st.multiselect(
                    "Student Strengths",
                    ["Creativity", "Verbal skills", "Visual-spatial thinking", "Memory", 
                     "Logical reasoning", "Collaboration", "Self-motivation", "Persistence"],
                    ["Creativity", "Visual-spatial thinking"]
                )
            
            submit_button = st.form_submit_button("Create Scenario")
        
        if submit_button:
            if not st.session_state.llm_interface:
                st.error("Please initialize the LLM interface first.")
                return
            
            try:
                # Format learning objectives as a list
                objectives_list = [obj.strip() for obj in learning_objectives.split('\n') if obj.strip()]
                
                # Create student profile
                student_profile = {
                    "grade_level": grade_level,
                    "learning_style": [style.lower() for style in learning_styles],
                    "challenges": [challenge.lower() for challenge in challenges],
                    "strengths": [strength.lower() for strength in strengths]
                }
                
                # Store the student profile in session state
                st.session_state.student_profile = student_profile
                
                # Create scenario context
                scenario_context = {
                    "subject": subject,
                    "difficulty": "intermediate",  # Default
                    "grade_level": grade_level,
                    "learning_objectives": objectives_list,
                    "student_profile": student_profile
                }
                
                # Use the pedagogical processor to create a scenario
                with st.spinner("Creating scenario..."):
                    scenario = st.session_state.pedagogical_processor.create_scenario(scenario_context)
                
                # Add additional fields to the scenario
                scenario["subject"] = subject
                scenario["grade_level"] = grade_level
                scenario["learning_objectives"] = objectives_list
                
                # Store the scenario
                st.session_state.scenario = scenario
                
                # Clear previous history if any
                st.session_state.history = []
                st.session_state.messages = []
                st.session_state.analysis = None
                st.session_state.strategies = []
                st.session_state.teacher_feedback = None
                st.session_state.reflection = None
                st.session_state.classroom_management_strategies = []
                
                # Display success message
                st.success("Teaching scenario created successfully!")
                
                # Display the scenario details
                self.display_scenario(scenario)
                
            except Exception as e:
                logging.error(f"Error creating scenario: {e}")
                st.error(f"Error creating scenario: {str(e)}")
    
    def display_scenario(self, scenario):
        """Display the details of the current teaching scenario."""
        st.markdown("## Scenario Details")
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Scenario")
            st.markdown(f"**Subject:** {scenario.get('subject', 'Not specified')}")
            st.markdown(f"**Description:** {scenario.get('description', 'No description available')}")
            
            # Learning objectives
            st.markdown("### Learning Objectives")
            objectives = scenario.get('learning_objectives', [])
            if objectives:
                for i, obj in enumerate(objectives, 1):
                    st.markdown(f"{i}. {obj}")
            else:
                st.markdown("No learning objectives specified")
        
        with col2:
            st.markdown("### Student Profile")
            
            student_profile = st.session_state.student_profile
            if student_profile:
                st.markdown(f"**Grade Level:** {student_profile.get('grade_level', 'Not specified')}")
                
                # Learning styles
                learning_styles = student_profile.get('learning_style', [])
                if learning_styles:
                    st.markdown(f"**Learning Styles:** {', '.join(style.capitalize() for style in learning_styles)}")
                
                # Challenges
                challenges = student_profile.get('challenges', [])
                if challenges:
                    st.markdown("**Challenges:**")
                    for challenge in challenges:
                        st.markdown(f"- {challenge.capitalize()}")
                
                # Strengths
                strengths = student_profile.get('strengths', [])
                if strengths:
                    st.markdown("**Strengths:**")
                    for strength in strengths:
                        st.markdown(f"- {strength.capitalize()}")
            else:
                st.markdown("No student profile available")
            
            # Additional scenario information
            st.markdown("### Potential Challenges")
            challenges = scenario.get('challenges', [])
            if challenges:
                for challenge in challenges:
                    st.markdown(f"- {challenge}")
            else:
                st.markdown("No challenges specified")

    def display_simulation_interface(self):
        """Display the interface for practicing teaching interactions."""
        st.markdown("## Practice Teaching Interaction")
        
        if not st.session_state.scenario:
            st.info("Please create a scenario first to begin the simulation.")
            return
        
        # Display conversation history
        if st.session_state.history:
            self.display_conversation()
        
        # Teacher input
        with st.form(key="interaction_form"):
            teacher_input = st.text_area(
                "Your response as the teacher",
                placeholder="Enter your teaching response...",
                height=150
            )
            
            submit = st.form_submit_button("Submit Response")
            
            if submit and teacher_input:
                if not st.session_state.llm_interface:
                    st.error("Please initialize the LLM interface first.")
                    return
                
                try:
                    # Add teacher message to history
                    st.session_state.history.append({
                        "role": "teacher",
                        "content": teacher_input,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Generate student response
                    with st.spinner("Student is responding..."):
                        student_response = st.session_state.llm_interface.simulate_student_response(
                            teacher_input=teacher_input,
                            student_profile=st.session_state.student_profile,
                            scenario_context=st.session_state.scenario
                        )
                    
                    # Add student message to history
                    st.session_state.history.append({
                        "role": "student",
                        "content": student_response,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    
                    # Analyze teaching approach after at least one exchange
                    if len(st.session_state.history) >= 2 and not st.session_state.analysis:
                        with st.spinner("Analyzing teaching approach..."):
                            analysis = st.session_state.llm_interface.analyze_teaching_strategies(
                                teacher_input=teacher_input,
                                student_profile=st.session_state.student_profile,
                                scenario_context=st.session_state.scenario
                            )
                            st.session_state.analysis = analysis
                    
                    # Refresh the page to show updated conversation
                    st.rerun()
                    
                except Exception as e:
                    logging.error(f"Error in simulation: {e}")
                    st.error(f"Error generating response: {str(e)}")
        
        # Quick tips and reminders
        with st.expander("Teaching Tips"):
            st.markdown("""
            ### Quick Reminders
            - Ask open-ended questions to engage critical thinking
            - Provide specific, constructive feedback
            - Adapt your approach based on the student's learning style
            - Use visual aids for visual learners
            - Break down complex concepts into smaller parts
            - Check for understanding before moving on
            """)
    
    def display_conversation(self):
        """Display the conversation history in a chat-like interface."""
        st.markdown("### Conversation")
        
        # Create a container for the conversation
        conversation_container = st.container()
        
        with conversation_container:
            for message in st.session_state.messages:
                role = message["role"]
                content = message["content"]
                timestamp = message.get("timestamp", "")
                
                # Use different styles for different roles
                if role == "teacher":
                    st.markdown(
                        f"""
                        <div class="teacher-message">
                            <div class="message-header">
                                <span class="role">Teacher</span>
                                <span class="timestamp">{timestamp}</span>
                            </div>
                            <div class="message-content">{content}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                elif role == "student":
                    st.markdown(
                        f"""
                        <div class="student-message">
                            <div class="message-header">
                                <span class="role">Student</span>
                                <span class="timestamp">{timestamp}</span>
                            </div>
                            <div class="message-content">{content}</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                # Add a small spacing between messages
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    
    def display_analysis(self):
        """Display analysis of the teaching interaction."""
        st.markdown("## Teaching Analysis")
        
        if not st.session_state.history:
            st.info("Complete a teaching interaction first to see analysis.")
            return
        
        # Get analysis if not already available
        if not st.session_state.analysis and st.session_state.llm_interface:
            with st.spinner("Analyzing teaching approach..."):
                last_teacher_message = None
                for message in reversed(st.session_state.history):
                    if message["role"] == "teacher":
                        last_teacher_message = message["content"]
                        break
                
                if last_teacher_message:
                    # Add classroom management strategies to the analysis context
                    analysis_context = {
                        "student_profile": st.session_state.student_profile,
                        "scenario_context": st.session_state.scenario,
                        "classroom_management_strategies": st.session_state.classroom_management_strategies if hasattr(st.session_state, 'classroom_management_strategies') else []
                    }
                    
                    analysis = st.session_state.llm_interface.analyze_teaching_strategies(
                        teacher_input=last_teacher_message,
                        student_profile=st.session_state.student_profile,
                        scenario_context=st.session_state.scenario
                    )
                    
                    # If classroom management strategies are being used, request specific feedback on them
                    if st.session_state.classroom_management_strategies:
                        strategies_str = "\n".join([f"- {s['name']} ({s['book']}): {s['description']}" for s in st.session_state.classroom_management_strategies])
                        
                        strategy_prompt = f"""
                        Analyze how effectively the teacher has implemented the following classroom management strategies in their teaching:
                        
                        {strategies_str}
                        
                        Teacher's last response: "{last_teacher_message}"
                        
                        For each strategy, provide:
                        1. A rating out of 10
                        2. Specific evidence from the teacher's response
                        3. Constructive feedback on how to better implement the strategy
                        
                        Format your response as JSON with the following structure:
                        {{
                            "strategy_analysis": [
                                {{
                                    "strategy_name": "Strategy name",
                                    "rating": 0-10,
                                    "evidence": "What the teacher did related to this strategy",
                                    "feedback": "Constructive suggestions for improvement"
                                }}
                            ],
                            "overall_strategy_implementation": "Overall assessment of strategy implementation",
                            "next_steps": "What the teacher should focus on next"
                        }}
                        """
                        
                        try:
                            strategy_analysis_response = st.session_state.llm_interface.get_chat_response([
                                {"role": "system", "content": "You are an expert in classroom management strategies and teacher evaluation."},
                                {"role": "user", "content": strategy_prompt}
                            ])
                            
                            # Try to parse the JSON response
                            import json
                            try:
                                # Find JSON in the response - look for content between curly braces
                                import re
                                json_match = re.search(r'\{.*\}', strategy_analysis_response, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    strategy_analysis = json.loads(json_str)
                                    analysis["strategy_analysis"] = strategy_analysis
                                else:
                                    logging.error("Could not extract JSON from strategy analysis response")
                            except Exception as e:
                                logging.error(f"Error parsing strategy analysis JSON: {e}", exc_info=True)
                        except Exception as e:
                            logging.error(f"Error getting strategy analysis: {e}", exc_info=True)
                    
                    st.session_state.analysis = analysis
        
        # Display the analysis
        if st.session_state.analysis:
            analysis = st.session_state.analysis
            
            # Create columns for the analysis display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Strengths
                st.markdown("### Teaching Strengths")
                strengths = analysis.get("strengths", [])
                if strengths:
                    for strength in strengths:
                        st.markdown(f"- {strength}")
                else:
                    st.markdown("No specific strengths identified.")
                
                # Areas for improvement
                st.markdown("### Areas for Improvement")
                areas = analysis.get("areas_for_improvement", [])
                if areas:
                    for area in areas:
                        st.markdown(f"- {area}")
                else:
                    st.markdown("No specific areas for improvement identified.")
                
                # Rationale
                if "rationale" in analysis:
                    st.markdown("### Analysis Rationale")
                    st.markdown(analysis["rationale"])
                
                # Strategy Analysis (if available)
                if "strategy_analysis" in analysis and isinstance(analysis["strategy_analysis"], dict):
                    st.markdown("### Classroom Management Strategy Implementation")
                    
                    strategy_analysis = analysis["strategy_analysis"]
                    
                    # Display individual strategy assessments
                    if "strategy_analysis" in strategy_analysis and isinstance(strategy_analysis["strategy_analysis"], list):
                        for strategy in strategy_analysis["strategy_analysis"]:
                            with st.expander(f"{strategy.get('strategy_name', 'Strategy')} - Rating: {strategy.get('rating', 'N/A')}/10"):
                                st.markdown(f"**Evidence:** {strategy.get('evidence', 'No evidence provided')}")
                                st.markdown(f"**Feedback:** {strategy.get('feedback', 'No feedback provided')}")
                    
                    # Display overall assessment
                    if "overall_strategy_implementation" in strategy_analysis:
                        st.markdown("#### Overall Assessment")
                        st.markdown(strategy_analysis["overall_strategy_implementation"])
                    
                    # Display next steps
                    if "next_steps" in strategy_analysis:
                        st.markdown("#### Recommended Next Steps")
                        st.markdown(strategy_analysis["next_steps"])
            
            with col2:
                # Effectiveness score
                if "effectiveness_score" in analysis:
                    score = analysis["effectiveness_score"]
                    st.markdown("### Effectiveness Score")
                    
                    # Create a gauge chart
                    fig = px.bar(
                        x=["Score"], 
                        y=[score], 
                        range_y=[0, 10],
                        color=[score],
                        color_continuous_scale=["red", "yellow", "green"],
                        range_color=[0, 10],
                        labels={"y": "Score", "x": ""}
                    )
                    
                    fig.update_layout(
                        height=300,
                        width=250,
                        xaxis_title=None,
                        yaxis_title="Score (1-10)",
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Score interpretation
                    if score >= 8:
                        st.success("Excellent teaching approach!")
                    elif score >= 6:
                        st.info("Good teaching approach with some room for improvement.")
                else:
                    st.warning("Analysis not available. Complete a teaching interaction to see analysis.")

    def display_resources(self):
        """Display educational resources related to the current teaching scenario."""
        st.markdown("## Teaching Resources")
        
        if not st.session_state.scenario:
            st.info("Please create a scenario first to get relevant resources.")
            return
        
        # Get scenario details for resource recommendations
        subject = st.session_state.scenario.get("subject", "")
        grade_level = st.session_state.scenario.get("grade_level", "")
        student_profile = st.session_state.student_profile or {}
        learning_styles = student_profile.get("learning_style", [])
        
        # Generate resource recommendations based on scenario
        if not st.session_state.llm_interface:
            st.error("Please initialize the LLM interface first.")
            return
        
        # Resource recommendations
        if st.button("Get Teaching Resource Recommendations"):
            with st.spinner("Generating resource recommendations..."):
                resource_prompt = f"""
                Recommend teaching resources for a {grade_level} {subject} class.
                The students have these learning styles: {', '.join(learning_styles)}.
                
                Include:
                1. Books and articles
                2. Online teaching platforms
                3. Interactive activities
                4. Assessment tools
                5. Professional development resources
                
                Return your recommendations in markdown format with clear headings and brief descriptions.
                """
                
                resources = st.session_state.llm_interface.get_chat_response([
                    {"role": "system", "content": "You are an educational resource specialist."},
                    {"role": "user", "content": resource_prompt}
                ])
                
                st.session_state.resources = resources
        
        # Display stored resources
        if "resources" in st.session_state and st.session_state.resources:
            st.markdown(st.session_state.resources)
        
        # Additional general resources
        with st.expander("General Teaching Resources"):
            st.markdown("""
            ### Online Platforms
            - **Khan Academy** - Free video lessons and practice exercises
            - **Edutopia** - Research-based strategies for teaching
            - **TeachersPayTeachers** - Lesson plans and teaching materials
            
            ### Teaching Methods
            - **Universal Design for Learning (UDL)** - Framework for flexible learning environments
            - **Differentiated Instruction** - Tailoring instruction to individual needs
            - **Project-Based Learning** - Engaging students through projects
            
            ### Assessment Tools
            - **Formative** - Create and share formative assessments
            - **Kahoot** - Game-based learning platform
            - **Quizlet** - Digital flashcards and study tools
            """)
    
    def chat_interface(self):
        """Display the chat interface for the teacher to interact with the simulated student."""
        # Get the scenario and student profile
        if "scenario" not in st.session_state:
            self.create_default_scenario()
        
        scenario = st.session_state.scenario
        student_profile = scenario.get('student_profile', {})
        
        # Initialize the LLM if not already done
        if not st.session_state.llm_interface:
            self.initialize_llm()
        
        # Add app header
        st.markdown("""
        <div class="app-header">
            <h1>Teacher Training Simulator</h1>
            <p>Practice and improve your teaching skills through realistic student interactions</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Layout the interface with a simple two-column design
        col1, col2 = st.columns([7, 3])
        
        with col1:
            # Enhanced header with more scenario information
            st.markdown(f"""
            <div class="scenario-header">
                <div class="scenario-title">
                    <span class="scenario-title-icon">📚</span> {scenario.get('subject', 'Subject')} - {scenario.get('topic', 'Topic')}
                </div>
                <p><strong>Grade Level:</strong> {scenario.get('grade_level', 'Not specified')}</p>
                <p><strong>Scenario Description:</strong> {scenario.get('scenario_description', 'No description available')}</p>
                <p><strong>Current Activity:</strong> {scenario.get('current_activity', 'No activity specified')}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Initialize the messages if they don't exist
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Add a welcome message from the student if this is the start of conversation
            if not st.session_state.messages:
                with st.spinner("Student is introducing themselves..."):
                    try:
                        # Generate a more natural introduction from the student
                        initial_prompt = f"""
                        You are a student named {student_profile.get('name', 'Jamie')} in a {scenario.get('grade_level', '7th grade')} classroom. 
                        
                        SCENARIO CONTEXT:
                        - Subject: {scenario.get('subject', 'Mathematics')}
                        - Topic: {scenario.get('topic', 'Introduction to Algebra')}
                        - Current activity: {scenario.get('current_activity', 'Working on problems')}
                        - Your learning style: {', '.join(student_profile.get('learning_style', ['visual']))}
                        - Your challenges: {', '.join(student_profile.get('challenges', ['math anxiety']))}
                        
                        The teacher has just entered the room and you are making eye contact.
                        Give a brief, realistic greeting to the teacher that subtly shows your personality and 
                        relates to the current subject/activity.
                        
                        Don't introduce yourself with a long description - just a natural, brief greeting
                        that a {student_profile.get('grade_level', '7th grade')} student would say, like "Hi" or "Good morning",
                        possibly with a question or comment about the subject.
                        Keep it to 1-2 sentences maximum and make it relevant to the scenario.
                        """
                        
                        # Fallback introduction for any errors
                        student_name = student_profile.get('name', 'Jamie')
                        fallback_intro = f"Hi, teacher! I'm {student_name}."
                        
                        # Try to generate a response
                        try:
                            # Format conversation history for context
                            history = self._format_conversation_history(st.session_state.messages)
                            
                            # Create context with conversation history
                            context_with_history = scenario.copy()
                            context_with_history['conversation_history'] = history
                            context_with_history['session_state'] = {'last_student_response': ''}
                            
                            # Try to generate a student introduction
                            student_intro = st.session_state.llm_interface.simulate_student_response(
                                teacher_input=initial_prompt,
                                student_profile=student_profile,
                                scenario_context=context_with_history
                            )
                            
                            # Clean up the intro to make sure it's not too formal or descriptive
                            if not student_intro or len(student_intro.split()) > 25 or "Error" in student_intro:
                                student_intro = fallback_intro
                        except Exception as e:
                            logging.error(f"Error generating initial student response: {e}", exc_info=True)
                            student_intro = fallback_intro
                        
                        # Add the student introduction to the messages
                        st.session_state.messages.append({"role": "student", "content": student_intro, "timestamp": self._get_timestamp()})
                    except Exception as e:
                        st.error(f"Error setting up student introduction: {str(e)}")
                        logging.error(f"Error setting up student introduction: {e}", exc_info=True)
                        st.session_state.messages.append({
                            "role": "student", 
                            "content": f"Hi teacher! I'm {student_profile.get('name', 'Jamie')}.",
                            "timestamp": self._get_timestamp()
                        })
            
            # Display conversation header
            st.markdown("""
            <div class="conversation-header">
                <h3>Classroom Conversation</h3>
                <p>The student has started the conversation. Respond as the teacher to help them learn.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display active classroom management strategies if any are selected
            if st.session_state.classroom_management_strategies:
                with st.expander("Active Classroom Management Strategies"):
                    st.markdown("<div style='background-color: #F0FFF4; padding: 15px; border-radius: 5px; border: 1px solid #C6F6D5;'>", unsafe_allow_html=True)
                    st.markdown("**Your Selected Teaching Strategies:**")
                    
                    # Group strategies by book
                    strategies_by_book = {}
                    for strategy in st.session_state.classroom_management_strategies:
                        book = strategy.get("book", "Unknown")
                        if book not in strategies_by_book:
                            strategies_by_book[book] = []
                        strategies_by_book[book].append(strategy)
                    
                    # Display strategies grouped by book
                    for book, strategies in strategies_by_book.items():
                        st.markdown(f"##### From: {book}")
                        for i, strategy in enumerate(strategies, 1):
                            st.markdown(f"**{strategy['name']}**: {strategy['description']}")
                            
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Add a tip on how to use the strategies
                    st.markdown("<div style='margin-top: 10px; background-color: #FFFFD0; padding: 15px; border-radius: 5px; border: 1px solid #E2E8F0;'>", unsafe_allow_html=True)
                    st.markdown("""**How to use these strategies:**
                    - Actively implement these techniques in your responses
                    - The student will respond according to how effectively you apply the strategies
                    - Mix different strategies for a comprehensive approach
                    """)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Add example problems if they exist
            if scenario.get('example_problems'):
                with st.expander("Example Problems"):
                    st.markdown("<div style='background-color: #F0F7FF; padding: 15px; border-radius: 5px; border: 1px solid #D0E3FF;'>", unsafe_allow_html=True)
                    st.markdown("**Current Worksheet Problems:**")
                    for i, problem in enumerate(scenario.get('example_problems', []), 1):
                        st.markdown(f"{i}. {problem}")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    if scenario.get('teaching_challenge'):
                        st.markdown("<div style='margin-top: 10px; background-color: #FFF0F0; padding: 15px; border-radius: 5px; border: 1px solid #FFD0D0;'>", unsafe_allow_html=True)
                        st.markdown(f"**Teaching Challenge:** {scenario.get('teaching_challenge')}")
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Use a container with fixed height and scrolling for the conversation
            with st.container():
                # Create a scrollable container for messages
                st.markdown('<div class="chat-container" id="chat-messages">', unsafe_allow_html=True)
                
                # Display the conversation
                for message in st.session_state.messages:
                    with st.chat_message(message["role"], avatar="👨‍🎓" if message["role"] == "student" else "👨‍🏫"):
                        st.write(f"{message['timestamp']}")
                        st.markdown(message["content"])
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Auto-scroll to bottom (using JavaScript)
                st.markdown("""
                <script>
                    function scrollToBottom() {
                        const chatContainer = document.querySelector('#chat-messages');
                        if (chatContainer) {
                            chatContainer.scrollTop = chatContainer.scrollHeight;
                        }
                    }
                    setTimeout(scrollToBottom, 500);
                </script>
                """, unsafe_allow_html=True)
            
            # Get teacher input
            teacher_input = st.chat_input("Type your response as the teacher...")
            
            if teacher_input:
                # Add teacher message to chat history
                st.session_state.messages.append({"role": "teacher", "content": teacher_input, "timestamp": self._get_timestamp()})
                
                # Process student response in the background
                try:
                    # Format conversation history for context
                    history = self._format_conversation_history(st.session_state.messages)
                    
                    # Get the last student response to avoid repetition
                    last_student_response = ""
                    for msg in reversed(st.session_state.messages):
                        if msg["role"] == "student":
                            last_student_response = msg["content"]
                            break
                    
                    # Add classroom management strategies to the context if any are selected
                    strategy_instructions = ""
                    if st.session_state.classroom_management_strategies:
                        strategy_instructions = "\n\nThe teacher is using the following classroom management strategies:\n"
                        for strategy in st.session_state.classroom_management_strategies:
                            strategy_instructions += f"- {strategy['name']}: {strategy['description']}\n"
                        strategy_instructions += "\nThe student should respond appropriately to these teaching strategies. If the teacher effectively uses the strategy, the student should show a slightly more positive, engaged, or improved response. If the strategy is not effectively applied, the student can show confusion or maintain their previous state."
                    
                    # Create an enhanced teacher input with explicit instructions to be conversational
                    enhanced_teacher_input = f"""
                    The teacher just said: "{teacher_input}"
                    
                    You are {student_profile.get('name', 'Jamie')}, a {student_profile.get('grade_level', '7th grade')} student.
                    Respond naturally as yourself in a brief, realistic way.
                    
                    KEY INSTRUCTIONS:
                    1. DO NOT repeat your introduction or describe yourself again
                    2. DO NOT say "As a 7th-grade student who..." or anything like that
                    3. Respond directly to what the teacher just said
                    4. Keep your response short (1-3 sentences)
                    5. Use casual, age-appropriate language
                    6. Show your personality through your response
                    7. If the teacher just said "hello" or greeted you, respond naturally and maybe mention something about the class or ask a question
                    8. NEVER list your traits, challenges or learning style directly - these should only be implied by your response
                    9. IMPORTANT: NEVER repeat any of your previous responses verbatim
                    10. Each of your responses should be unique and move the conversation forward
                    {strategy_instructions}
                    
                    Your previous response was: "{last_student_response}"
                    DO NOT repeat this response - say something completely new and different that makes sense in the context of this conversation.
                    """
                    
                    # Create context with conversation history and last response
                    context_with_history = scenario.copy()
                    context_with_history['conversation_history'] = history
                    context_with_history['previous_response'] = last_student_response
                    context_with_history['session_state'] = {'last_student_response': last_student_response}
                    context_with_history['instruction'] = "Be brief and conversational, like a real student in class"
                    
                    # Add classroom management strategies to the context
                    if st.session_state.classroom_management_strategies:
                        context_with_history['classroom_management_strategies'] = st.session_state.classroom_management_strategies
                    
                    # Try to get the student's response with fallbacks
                    try:
                        response = st.session_state.llm_interface.simulate_student_response(
                            teacher_input=enhanced_teacher_input,
                            student_profile=student_profile,
                            scenario_context=context_with_history
                        )
                        
                        # Check if response is identical to any previous student response
                        previous_responses = [msg["content"] for msg in st.session_state.messages if msg["role"] == "student"]
                        if response in previous_responses:
                            logging.warning("Generated response is identical to a previous one. Using fallback.")
                            
                            # Random conversational responses that can work in various contexts
                            fallback_options = [
                                "I'm not sure I understand. Could you explain that differently?",
                                "That's interesting. Can we talk more about how this works?",
                                "I've been thinking about what you said. Can you give another example?",
                                "OK, I think I'm starting to get it.",
                                "Let me try to understand this from another angle.",
                                "I'm still confused about part of this.",
                                "So what does that mean for our assignment?",
                                "I hadn't thought about it that way before."
                            ]
                            
                            # Choose a random fallback that's not in previous responses
                            filtered_fallbacks = [opt for opt in fallback_options if opt not in previous_responses]
                            if filtered_fallbacks:
                                response = random.choice(filtered_fallbacks)
                            else:
                                # Create a unique response by adding a small random modifier
                                response = f"I see. {random.choice(['Actually, ', 'Well, ', 'Hmm, ', 'So, '])}I'm still working through this."
                    except Exception as e:
                        logging.error(f"Error generating student response: {e}", exc_info=True)
                        # Set up a fallback response
                        response = "I'm not sure how to respond to that."
                    
                    # Check if response is empty or has an error
                    if not response or "Error" in response:
                        # Create a simple fallback response based on the teacher's input
                        if "?" in teacher_input:
                            response = "I'm not really sure about that. Can you explain it differently?"
                        else:
                            response = "Okay, I understand. What should we do next?"
                    
                    # Check if response seems like a self-description rather than a natural response
                    # We'll look for common patterns that indicate the student is describing themselves
                    description_indicators = [
                        "as a", "my learning style", "I am a", "my strengths", 
                        "my challenges", "I struggle with", "my teachers say"
                    ]
                    
                    is_description = any(indicator in response.lower() for indicator in description_indicators)
                    
                    # If it's a description or too long, generate a simpler response
                    if is_description or len(response.split()) > 50:
                        logging.info("Detected self-description in student response, generating simpler response")
                        
                        # Generate a very simple, age-appropriate response
                        if "hello" in teacher_input.lower() or "hi" in teacher_input.lower() or "hey" in teacher_input.lower():
                            fallback_options = [
                                f"Hi! Are we starting algebra today?",
                                f"Hey! Do we need our textbooks for today's lesson?",
                                f"Hello! I was just trying to figure out this math problem.",
                                f"Hi there! Is this going to be on the test?",
                                f"Hey teacher, I was wondering what we're covering today.",
                                f"Hi! I've been stuck on problem number 5, can you help me?"
                            ]
                            # Avoid repeating the last student response
                            last_response = ""
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "student":
                                    last_response = msg["content"]
                                    break
                            
                            # Filter out any options that match the previous response
                            filtered_options = [opt for opt in fallback_options if opt != last_response]
                            
                            # If we have options left, choose from them, otherwise use a more generic response
                            if filtered_options:
                                response = random.choice(filtered_options)
                            else:
                                response = "Sorry, I didn't catch what you said. Can you explain what we're doing today?"
                        elif "?" in teacher_input:
                            fallback_options = [
                                "Um, I'm not really sure.",
                                "I don't know, can you explain it again?",
                                "Maybe? I get confused with all these x's and y's.",
                                "I think so... but I'm not 100% sure.",
                                "Could you break that down for me?",
                                "I'm still trying to understand that concept.",
                                "Not completely. Could you show an example?"
                            ]
                            # Avoid repeating the last student response
                            last_response = ""
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "student":
                                    last_response = msg["content"]
                                    break
                            
                            # Filter out any options that match the previous response
                            filtered_options = [opt for opt in fallback_options if opt != last_response]
                            
                            # If we have options left, choose from them, otherwise use a more generic response
                            if filtered_options:
                                response = random.choice(filtered_options)
                            else:
                                response = "I'm confused about that question. Can we try a different approach?"
                        else:
                            fallback_options = [
                                "Okay, I'll try that.",
                                "That makes sense, I think.",
                                "So we're supposed to solve for x?",
                                "Can you show me an example first?",
                                "Math is so confusing sometimes.",
                                "I'll give it my best shot.",
                                "Let me see if I understand the process correctly.",
                                "That's clearer now, thanks!"
                            ]
                            # Avoid repeating the last student response
                            last_response = ""
                            for msg in reversed(st.session_state.messages):
                                if msg["role"] == "student":
                                    last_response = msg["content"]
                                    break
                            
                            # Filter out any options that match the previous response
                            filtered_options = [opt for opt in fallback_options if opt != last_response]
                            
                            # If we have options left, choose from them, otherwise use a more generic response
                            if filtered_options:
                                response = random.choice(filtered_options)
                            else:
                                response = "I'll work on that. Is there something specific I should focus on?"
                    
                    # Add the response to messages
                    timestamp = self._get_timestamp()
                    st.session_state.messages.append({"role": "student", "content": response, "timestamp": timestamp})
                    
                except Exception as e:
                    error_message = f"Error generating student response: {str(e)}"
                    logging.error(f"{error_message}", exc_info=True)
                    
                    # Provide a fallback response
                    fallback_response = f"Sorry, what do you mean?"
                    timestamp = self._get_timestamp()
                    st.session_state.messages.append({"role": "student", "content": fallback_response, "timestamp": timestamp})
                
                # Rerun the app to display the updated conversation
                st.rerun()
        
        # Simplified information panel in the side column
        with col2:
            # Wrap all info cards in a scrollable container
            st.markdown('<div class="info-panel">', unsafe_allow_html=True)
            
            # Student Profile Card - Simplified
            st.markdown("""
            <div class="scenario-card">
                <div class="scenario-title">
                    <span class="scenario-title-icon">👨‍🎓</span> Student Profile
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Name:** {student_profile.get('name', 'Not specified')}")
            st.markdown(f"**Grade:** {student_profile.get('grade_level', 'Not specified')}")
            st.markdown(f"**Age:** {student_profile.get('age', 'Not specified')}")
            
            # Add personality and interests
            if student_profile.get('personality'):
                st.markdown(f"**Personality:** {student_profile.get('personality', '')}")
            
            if student_profile.get('interests'):
                st.markdown("**Interests:**")
                interests = student_profile.get('interests', [])
                if isinstance(interests, list):
                    st.markdown(', '.join(interests))
                else:
                    st.markdown(interests)
            
            # Simplified Learning Style display
            if student_profile.get('learning_style'):
                st.markdown("**Learning Style:**")
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for style in student_profile.get('learning_style', []):
                    st.markdown(f"<li>{style}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
                
            # Simplified Challenges display
            if student_profile.get('challenges'):
                st.markdown("**Challenges:**")
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for challenge in student_profile.get('challenges', []):
                    st.markdown(f"<li>{challenge}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
            
            # Simplified Strengths display
            if student_profile.get('strengths'):
                st.markdown("**Strengths:**")
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for strength in student_profile.get('strengths', []):
                    st.markdown(f"<li>{strength}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add a Current Scenario card
            st.markdown("""
            <div class="scenario-card">
                <div class="scenario-title">
                    <span class="scenario-title-icon">📝</span> Current Scenario
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Subject:** {scenario.get('subject', 'Not specified')}")
            st.markdown(f"**Topic:** {scenario.get('topic', 'Not specified')}")
            
            if scenario.get('classroom_setting'):
                st.markdown(f"**Setting:** {scenario.get('classroom_setting', '')}")
                
            if scenario.get('difficulty'):
                st.markdown(f"**Difficulty:** {scenario.get('difficulty', '').capitalize()}")
                
            # Add scenario description if available
            if scenario.get('scenario_description'):
                with st.expander("Scenario Description"):
                    st.markdown(scenario.get('scenario_description', ''))
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Learning Objectives - If available
            if scenario.get('learning_objectives'):
                st.markdown("""
                <div class="scenario-card">
                    <div class="scenario-title">
                        <span class="scenario-title-icon">🎯</span> Learning Objectives
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown('<ul class="info-list">', unsafe_allow_html=True)
                for objective in scenario.get('learning_objectives', []):
                    st.markdown(f"<li>{objective}</li>", unsafe_allow_html=True)
                st.markdown('</ul>', unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Control panel for buttons - Simplified
            st.markdown("""
            <div class="control-panel">
                <div class="scenario-title">
                    <span class="scenario-title-icon">⚙️</span> Controls
                </div>
            """, unsafe_allow_html=True)
            
            # Add a button to reset the conversation
            if st.button("Reset Conversation", key="reset_conversation"):
                st.session_state.messages = []
                st.rerun()
            
            # Add a button to start a new scenario
            if st.button("Create New Scenario", key="new_scenario"):
                st.session_state.page = "create_scenario"
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Close the info panel div
            st.markdown('</div>', unsafe_allow_html=True)

    def _format_conversation_history(self, messages):
        """Format the conversation history to provide context for the LLM."""
        formatted_history = []
        
        for message in messages:
            role = "Teacher" if message["role"] == "teacher" else "Student"
            formatted_history.append(f"{role}: {message['content']}")
        
        return "\n".join(formatted_history)

    def _get_timestamp(self):
        """Get the current timestamp for messages."""
        return datetime.now().strftime("%H:%M:%S")

    def create_default_scenario(self):
        """Create a default scenario if none exists."""
        if not st.session_state.scenario:
            # Create a more detailed default student profile
            default_student_profile = {
                "name": "Jamie",
                "grade_level": "7th grade",
                "age": 12,
                "learning_style": ["visual", "hands-on"],
                "challenges": ["math anxiety", "difficulty with abstract concepts", "attention span"],
                "strengths": ["creativity", "verbal expression", "collaborative work"],
                "interests": ["art", "music", "working with friends"],
                "personality": "curious but easily frustrated when concepts seem too abstract"
            }
            
            # Create a more detailed default scenario
            default_scenario = {
                "subject": "Mathematics",
                "topic": "Introduction to Algebra",
                "grade_level": "7th grade",
                "difficulty": "moderate",
                "classroom_setting": "Small group work on algebraic expressions",
                "learning_objectives": [
                    "Understand variables as representing unknown quantities",
                    "Translate word problems into algebraic expressions",
                    "Solve simple linear equations"
                ],
                "scenario_description": "Students are working on translating word problems into algebraic expressions. Jamie has been struggling with the concept of variables and is showing signs of frustration. The teacher needs to help Jamie understand the concept while maintaining engagement and confidence.",
                "current_activity": "Students are working on a worksheet with word problems that need to be translated into algebraic expressions. Jamie is stuck on a problem about finding the unknown number.",
                "student_profile": default_student_profile,
                "teaching_challenge": "Help the student overcome math anxiety while building understanding of abstract algebraic concepts",
                "example_problems": [
                    "If a number plus 5 equals 12, what is the number?",
                    "Sarah is 3 years older than twice Miguel's age. If Sarah is 15, how old is Miguel?"
                ]
            }
            
            st.session_state.scenario = default_scenario
            st.session_state.student_profile = default_student_profile
            
            logging.info("Created detailed default scenario")
            return True
        return False

    def run(self):
        """Run the Streamlit web application."""
        # Setup the page
        self.setup_page()
        
        # Create default scenario if needed
        self.create_default_scenario()
        
        # Sidebar for navigation and settings
        self.setup_sidebar()
        
        # Main content area based on selected mode
        if st.session_state.app_mode == "Chat Interface":
            self.chat_interface()
        elif st.session_state.app_mode == "Create New Scenario":
            self.create_scenario()
        elif st.session_state.app_mode == "Analysis":
            self.display_analysis()
        elif st.session_state.app_mode == "Resources":
            self.display_resources()
        else:
            # Default to chat interface
            self.chat_interface()

    def setup_sidebar(self):
        """Set up the sidebar with navigation and settings."""
        with st.sidebar:
            # Settings section - simplified
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
            
            st.markdown("**Select Language Model**")
            
            # Model selection - simplified options
            model_options = ["GPT-3.5 Turbo", "GPT-4"]
            selected_model = st.selectbox(
                "Select a model", 
                model_options, 
                index=0, 
                label_visibility="collapsed"
            )
            
            # Map the display name to the model identifier
            model_mapping = {
                "GPT-3.5 Turbo": "gpt-3.5-turbo",
                "GPT-4": "gpt-4"
            }
            
            # Display current model
            st.markdown(f"**Current model:** {selected_model}")
            
            # Initialize or switch model if needed
            if "model_name" not in st.session_state or st.session_state.model_name != model_mapping[selected_model]:
                if st.button("Apply Model"):
                    with st.spinner(f"Initializing {selected_model}..."):
                        self.initialize_llm(model_mapping[selected_model])
                        st.success(f"Model switched to {selected_model}")
                        st.session_state.messages = []  # Reset conversation with new model
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Classroom Management Strategies section
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">🧠 Classroom Management Strategies</div>', unsafe_allow_html=True)
            
            # Book selection
            selected_book = st.selectbox(
                "Select a book",
                list(self.classroom_management_books.keys()),
                index=0,
                key="strategy_book"
            )
            
            # Strategy selection from the chosen book
            if selected_book:
                strategies = self.classroom_management_books[selected_book]
                strategy_names = [strategy["name"] for strategy in strategies]
                
                selected_strategies = st.multiselect(
                    "Select strategies to apply",
                    strategy_names,
                    key="selected_strategies"
                )
                
                # Display descriptions of selected strategies
                if selected_strategies:
                    st.markdown("### Selected Strategies")
                    for strategy_name in selected_strategies:
                        # Find the strategy object with this name
                        for strategy in strategies:
                            if strategy["name"] == strategy_name:
                                with st.expander(strategy_name):
                                    st.markdown(f"**{strategy_name}**")
                                    st.markdown(strategy["description"])
                                break
                    
                    # Update session state with selected strategies
                    if st.button("Apply Strategies"):
                        # Collect full strategy objects (with descriptions) for the selected strategy names
                        full_selected_strategies = []
                        for strategy_name in selected_strategies:
                            for strategy in strategies:
                                if strategy["name"] == strategy_name:
                                    full_selected_strategies.append({
                                        "name": strategy["name"],
                                        "description": strategy["description"],
                                        "book": selected_book
                                    })
                                    break
                        
                        st.session_state.classroom_management_strategies = full_selected_strategies
                        st.success(f"Applied {len(full_selected_strategies)} classroom management strategies!")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Navigation section - simplified
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
            
            # Simple navigation options
            mode_options = [
                {"label": "Chat Interface", "value": "chat", "icon": "💬"},
                {"label": "Create New Scenario", "value": "create_scenario", "icon": "🔧"}
            ]
            
            # Create radio buttons for navigation
            selected_mode = st.radio(
                "Select Mode",
                [f"{option['icon']} {option['label']}" for option in mode_options],
                label_visibility="collapsed"
            )
            
            # Set the page based on selection
            for option in mode_options:
                if f"{option['icon']} {option['label']}" == selected_mode:
                    st.session_state.page = option['value']
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Options section - simplified
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Options</div>', unsafe_allow_html=True)
            
            # Reset buttons
            if st.button("Reset Conversation", key="sidebar_reset_conversation"):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("Reset Everything", key="sidebar_reset_everything"):
                # Clear all session state variables
                for key in list(st.session_state.keys()):
                    if key != "model_name":
                        del st.session_state[key]
                # Explicitly reset classroom management strategies
                st.session_state.classroom_management_strategies = []
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    app = WebInterface()
    app.run() 