"""
Streamlit web application for the Teacher Training Simulator.
Provides an interactive interface for teachers to practice and analyze teaching scenarios.
"""

import streamlit as st
import os
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from ai_agent import TeacherTrainingAgent as AIAgent

class WebInterface:
    def __init__(self):
        """Initialize the web interface and session state."""
        if 'agent' not in st.session_state:
            st.session_state.agent = AIAgent()
        if 'scenario' not in st.session_state:
            st.session_state.scenario = None
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'analysis' not in st.session_state:
            st.session_state.analysis = None
        if 'strategies' not in st.session_state:
            st.session_state.strategies = []

    def setup_page(self):
        """Configure the Streamlit page."""
        st.set_page_config(
            page_title="Teacher Training Simulator",
            page_icon="🎓",
            layout="wide"
        )
        st.title("🎓 Teacher Training Simulator")

    def display_scenario(self):
        """Display the current teaching scenario with detailed analysis options."""
        if not st.session_state.scenario:
            st.session_state.scenario = st.session_state.agent.generate_scenario()

        scenario = st.session_state.scenario
        
        # Scenario Overview
        with st.expander("📚 Current Scenario", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Context")
                st.write(f"**Subject:** {scenario['subject'].title()}")
                st.write(f"**Time of Day:** {scenario['time_of_day'].replace('_', ' ').title()}")
                st.write(f"**Learning Objectives:**")
                for obj in scenario['learning_objectives']:
                    st.write(f"- {obj}")

            with col2:
                st.subheader("Student Profile")
                st.write(f"**Learning Style:** {scenario['student_context']['learning_style']}")
                st.write(f"**Current Challenges:**")
                for challenge in scenario['student_context']['current_challenges']:
                    st.write(f"- {challenge}")
                st.write(f"**Behavioral Context:** {scenario['behavioral_context']['manifestation']}")

        # Situation Analysis
        with st.expander("🔍 Analyze Situation", expanded=True):
            st.subheader("Situation Analysis")
            
            # Teacher's Analysis
            analysis = st.text_area(
                "What do you observe in this situation? Describe the student's behavior and possible underlying causes.",
                height=100,
                key="situation_analysis"
            )
            
            if analysis:
                # Store analysis
                st.session_state.analysis = analysis
                
                # Provide feedback on analysis
                st.write("### Key Elements Identified:")
                analysis_feedback = self.analyze_teacher_observation(analysis, scenario)
                
                for category, elements in analysis_feedback.items():
                    st.write(f"**{category}:**")
                    for element in elements:
                        st.write(f"- {element}")

        # Strategy Selection
        with st.expander("📋 Select Teaching Strategies", expanded=True):
            st.subheader("Teaching Strategies")
            
            # Get relevant strategies based on scenario
            available_strategies = self.get_relevant_strategies(scenario)
            
            # Group strategies by category
            for category, strategies in available_strategies.items():
                st.write(f"**{category}:**")
                for strategy in strategies:
                    if st.checkbox(strategy['name'], key=f"strategy_{strategy['name']}"):
                        if strategy not in st.session_state.strategies:
                            st.session_state.strategies.append(strategy)

            # Display selected strategies
            if st.session_state.strategies:
                st.write("### Selected Strategies:")
                for strategy in st.session_state.strategies:
                    st.write(f"- {strategy['name']}")
                    st.write(f"  *{strategy['description']}*")

        # Response Formulation
        with st.expander("💭 Formulate Response", expanded=True):
            st.subheader("Teacher Response")
            
            # Show strategy recommendations
            if st.session_state.strategies:
                st.write("**Incorporating selected strategies:**")
                for strategy in st.session_state.strategies:
                    st.write(f"- Consider {strategy['example']}")

            # Teacher's response
            response = st.text_area(
                "What would you say or do in this situation?",
                height=100,
                key="teacher_response"
            )

            if st.button("Submit Response"):
                if response:
                    # Evaluate response
                    evaluation = st.session_state.agent.evaluate_response(response, scenario)
                    
                    # Display evaluation
                    self.display_evaluation(evaluation)
                    
                    # Store interaction
                    st.session_state.history.append({
                        'timestamp': datetime.now(),
                        'scenario': scenario,
                        'analysis': st.session_state.analysis,
                        'strategies': st.session_state.strategies.copy(),
                        'response': response,
                        'evaluation': evaluation
                    })
                    
                    # Clear strategies for next interaction
                    st.session_state.strategies = []
                    
                    # Option for new scenario
                    if st.button("Generate New Scenario"):
                        st.session_state.scenario = st.session_state.agent.generate_scenario()
                        st.experimental_rerun()
                else:
                    st.warning("Please enter a response before submitting.")

    def analyze_teacher_observation(self, analysis, scenario):
        """Analyze the teacher's observation of the situation."""
        # Initialize feedback categories
        feedback = {
            "Behavioral Observations": [],
            "Learning Style Considerations": [],
            "Context Recognition": [],
            "Potential Interventions": []
        }
        
        # Analyze behavioral observations
        behavior_keywords = ["fidgeting", "distracted", "frustrated", "confused", "engaged"]
        for keyword in behavior_keywords:
            if keyword in analysis.lower():
                feedback["Behavioral Observations"].append(
                    f"Identified {keyword} behavior"
                )
        
        # Analyze learning style considerations
        learning_style = scenario['student_context']['learning_style']
        if learning_style.lower() in analysis.lower():
            feedback["Learning Style Considerations"].append(
                f"Recognized {learning_style} learning style"
            )
        
        # Analyze context recognition
        context_elements = {
            "time": scenario['time_of_day'],
            "subject": scenario['subject'],
            "trigger": scenario['behavioral_context']['trigger']
        }
        for element, value in context_elements.items():
            if value.lower() in analysis.lower():
                feedback["Context Recognition"].append(
                    f"Noted {element}: {value}"
                )
        
        # Analyze potential interventions
        intervention_keywords = ["strategy", "approach", "help", "support", "guide"]
        for keyword in intervention_keywords:
            if keyword in analysis.lower():
                feedback["Potential Interventions"].append(
                    f"Considering {keyword}-based intervention"
                )
        
        return feedback

    def get_relevant_strategies(self, scenario):
        """Get teaching strategies relevant to the current scenario."""
        # Group strategies by category
        strategies = {
            "Time-Based Strategies": [
                {
                    "name": f"{scenario['time_of_day'].title()} Energy Management",
                    "description": "Strategies suited for student energy levels at this time",
                    "example": "using structured activities to maintain focus"
                }
            ],
            "Learning Style Strategies": [
                {
                    "name": f"{scenario['student_context']['learning_style'].title()} Learning Approach",
                    "description": f"Techniques optimized for {scenario['student_context']['learning_style']} learners",
                    "example": "using visual aids and demonstrations"
                }
            ],
            "Behavioral Management": [
                {
                    "name": f"Address {scenario['behavioral_context']['type'].title()}",
                    "description": "Techniques to manage current behavioral state",
                    "example": "providing clear, step-by-step instructions"
                }
            ],
            "Subject-Specific": [
                {
                    "name": f"{scenario['subject'].title()} Support",
                    "description": "Subject-specific teaching strategies",
                    "example": "breaking down complex problems into smaller steps"
                }
            ]
        }
        
        return strategies

    def display_evaluation(self, evaluation):
        """Display the evaluation results with detailed feedback."""
        st.write("### Response Evaluation")
        
        # Display score with color coding
        score = evaluation['score']
        if score >= 0.8:
            st.success(f"Score: {score*100:.0f}%")
        elif score >= 0.6:
            st.warning(f"Score: {score*100:.0f}%")
        else:
            st.error(f"Score: {score*100:.0f}%")
        
        # Display strengths
        if evaluation['feedback']:
            st.write("**Strengths:**")
            for strength in evaluation['feedback']:
                st.write(f"✓ {strength}")
        
        # Display suggestions
        if evaluation['suggestions']:
            st.write("**Suggestions for Improvement:**")
            for suggestion in evaluation['suggestions']:
                st.write(f"→ {suggestion}")
        
        # Display student reaction
        st.write("**Student Reaction:**")
        st.info(evaluation['student_reaction'])
        
        # Display state changes
        if 'state_changes' in evaluation:
            st.write("**Impact on Student State:**")
            cols = st.columns(len(evaluation['state_changes']))
            for col, (state, change) in zip(cols, evaluation['state_changes'].items()):
                with col:
                    st.metric(
                        label=state.title(),
                        value=f"{change*100:.0f}%",
                        delta=f"{change*100:+.0f}%"
                    )

    def display_history(self):
        """Display the session history with analysis."""
        if not st.session_state.history:
            st.info("No interactions recorded yet.")
            return
        
        st.write("## Session History")
        
        # Summary metrics
        total_interactions = len(st.session_state.history)
        avg_score = sum(h['evaluation']['score'] for h in st.session_state.history) / total_interactions
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Interactions", total_interactions)
        with col2:
            st.metric("Average Score", f"{avg_score*100:.1f}%")
        with col3:
            st.metric("Best Score", f"{max(h['evaluation']['score']*100 for h in st.session_state.history):.1f}%")
        
        # Progress chart
        scores = [h['evaluation']['score'] for h in st.session_state.history]
        df = pd.DataFrame({
            'Interaction': range(1, len(scores) + 1),
            'Score': scores
        })
        fig = px.line(df, x='Interaction', y='Score', title='Progress Over Time')
        st.plotly_chart(fig)
        
        # Detailed history
        for i, interaction in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Interaction {len(st.session_state.history)-i+1}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Scenario:**")
                    st.write(f"Subject: {interaction['scenario']['subject']}")
                    st.write(f"Time: {interaction['scenario']['time_of_day']}")
                    
                    st.write("**Analysis:**")
                    st.write(interaction['analysis'])
                    
                    st.write("**Selected Strategies:**")
                    for strategy in interaction['strategies']:
                        st.write(f"- {strategy['name']}")
                
                with col2:
                    st.write("**Response:**")
                    st.write(interaction['response'])
                    
                    st.write("**Evaluation:**")
                    st.write(f"Score: {interaction['evaluation']['score']*100:.0f}%")
                    st.write("Student Reaction:")
                    st.info(interaction['evaluation']['student_reaction'])

def main():
    """Run the Streamlit web application."""
    app = WebInterface()
    app.setup_page()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Current Scenario", "Session History"])
    
    if page == "Current Scenario":
        app.display_scenario()
    else:
        app.display_history()

if __name__ == "__main__":
    main() 