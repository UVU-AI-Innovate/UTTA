"""
DSPy Language Model Interface and Processing Module

This module provides a comprehensive interface for interacting with Language
Models (LLMs) in the teaching simulation system using DSPy. It handles all communication
with the LLM backend while providing context management, error handling, and
pedagogical processing capabilities.

Key Features:
    - Standardized LLM communication using DSPy
    - Advanced prompt optimization
    - Context-aware prompt building
    - Pedagogical language processing
    - Teaching response analysis
    - Student reaction generation
    - Error handling and recovery
    - Configurable model parameters

Components:
    - DSPyLLMInterface: Base interface for LLM interaction
    - EnhancedDSPyLLMInterface: Advanced interface with streaming and specialized prompts
    - PedagogicalLanguageProcessor: Educational language processor

Dependencies:
    - dspy: For LLM interface and prompt programming
    - torch: For GPU operations
    - typing: For type hints
"""

import os
import json
import time
import random
import logging
import threading
import re
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# DSPy imports
import dspy
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShot
from dspy.signatures import Signature

# For error handling and retry logic
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import OpenAI
from langchain_openai import ChatOpenAI as openai

@dataclass
class TeachingContext:
    """Context for teaching interactions."""
    student_level: str
    topic: str
    previous_interactions: List[Dict[str, str]]
    learning_objectives: List[str]

@dataclass
class TeachingResponse:
    """Structured teaching response."""
    content: str
    scaffolding_level: int
    engagement_elements: List[str]
    follow_up_questions: List[str]
    suggested_activities: List[str]

class DSPyConfigManager:
    """
    Singleton class to manage DSPy configuration across threads.
    Ensures thread-safe access to DSPy configuration state.
    """
    _instance = None
    _lock = threading.Lock()
    is_configured = False
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DSPyConfigManager, cls).__new__(cls)
                cls._instance.configure_attempts = 0
                cls._instance.lm = None
            return cls._instance
    
    def configure_dspy_settings(self, model_name="gpt-3.5-turbo", max_attempts=3):
        """
        Configure DSPy settings with thread safety.
        
        Args:
            model_name: The name of the model to use
            max_attempts: Maximum number of configuration attempts
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        with self._lock:
            # If already configured, return success
            if self.is_configured and self.lm is not None:
                logging.info("DSPy already configured, reusing configuration")
                return True
            
            # Prevent excessive retries
            if self.configure_attempts >= max_attempts:
                logging.error(f"Failed to configure DSPy after {max_attempts} attempts")
                return False
            
            self.configure_attempts += 1
            
            try:
                # Get the API key from environment
                api_key = os.environ.get('OPENAI_API_KEY', None)
                if not api_key:
                    logging.error("No OpenAI API key found in environment")
                    return False
                
                # Ensure API key is in environment
                os.environ['OPENAI_API_KEY'] = api_key
                
                # Configure the language model
                if "gpt" in model_name.lower():
                    # Use DSPy's OpenAIProvider correctly - it reads API key from environment
                    from dspy.clients.openai import OpenAIProvider
                    self.lm = dspy.LM(model=model_name, provider=OpenAIProvider(), temperature=0.7)
                else:
                    # Fallback to langchain_openai adapter
                    self.lm = openai(
                        model=model_name,
                        api_key=api_key,
                        temperature=0.7
                    )
                
                # Configure DSPy with the language model
                dspy.configure(lm=self.lm)
                
                # Mark as configured
                self.is_configured = True
                logging.info(f"DSPy settings configured with model: {model_name}")
                return True
                
            except Exception as e:
                logging.error(f"Error configuring DSPy settings: {e}", exc_info=True)
                self.is_configured = False
                return False
    
    def ensure_configuration(self, model_name="gpt-3.5-turbo"):
        """
        Ensure DSPy is configured before use.
        
        Args:
            model_name: The name of the model to use
            
        Returns:
            bool: True if configuration was successful, False otherwise
        """
        if not self.is_configured:
            return self.configure_dspy_settings(model_name)
        return True

# Create global configuration manager
dspy_config = DSPyConfigManager()

# Define DSPy signatures for teaching-specific tasks
class TeachingResponse(Signature):
    """Response signature for teaching recommendations and scenarios."""
    response: str = dspy.OutputField(desc="Educational response appropriate for the student and context, or a JSON-formatted scenario")

class ScenarioResponse(Signature):
    """Response signature for structured teaching scenarios."""
    scenario_title: str = dspy.OutputField(desc="A concise title for the scenario")
    scenario_description: str = dspy.OutputField(desc="A detailed description of the classroom situation")
    learning_objectives: List[str] = dspy.OutputField(desc="The educational goals for this lesson")
    student_background: str = dspy.OutputField(desc="Description of the student(s) involved")
    teacher_challenge: str = dspy.OutputField(desc="The specific challenge the teacher faces")

class StudentResponse(Signature):
    """Response signature for simulated student responses."""
    response: str = dspy.OutputField(desc="Authentic first-person response from the student's perspective, reflecting their grade level, learning style, challenges, and personality. Do not break character.")

class TeachingAnalysis(Signature):
    """Response signature for teaching analysis."""
    strengths: List[str] = dspy.OutputField(desc="Teaching strengths identified in the response")
    areas_for_improvement: List[str] = dspy.OutputField(desc="Potential areas for improvement")
    effectiveness_score: int = dspy.OutputField(desc="Score from 1-10 rating the teaching effectiveness")
    rationale: str = dspy.OutputField(desc="Explanation of the analysis")

# Define DSPy modules for different teaching tasks
class TeacherResponder(dspy.Module):
    """Module for generating teaching responses and scenarios."""
    
    def __init__(self):
        super().__init__()
        # We'll support both free-form text responses and structured scenario responses
        self.generate_teaching_scenario = Predict(TeachingResponse)
        self.generate_structured_scenario = Predict(ScenarioResponse)
    
    def forward(self, context, student_profile, question):
        """Generate a teaching scenario or response."""
        # First try to use the structured signature if it looks like a scenario request
        if "scenario" in question.lower() or "classroom" in question.lower():
            try:
                # For structured scenarios, use the specialized signature
                return self.generate_structured_scenario(
                    context=context,
                    student_profile=student_profile, 
                    question=question
                )
            except Exception as e:
                logging.error(f"Error generating structured scenario, falling back to text response: {e}")
                # Fall back to the text response if the structured one fails
                pass
        
        # Otherwise use the text response signature with a JSON schema prompt
        json_schema_prompt = """
        Your response should be a valid JSON object with the following structure:
        {
            "scenario_title": "A concise title for the scenario",
            "scenario_description": "A detailed description of the classroom situation",
            "learning_objectives": ["The educational goals for this lesson"],
            "student_background": "Description of the student(s) involved",
            "teacher_challenge": "The specific challenge the teacher faces"
        }
        
        Ensure your response is properly formatted as JSON. Do not include any text before or after the JSON object.
        """
        
        # Combine the original question with the JSON schema prompt
        enhanced_question = f"{question}\n\n{json_schema_prompt}"
        
        # Generate the response
        return self.generate_teaching_scenario(
            context=context,
            student_profile=student_profile, 
            question=enhanced_question
        )

class StudentReactionGenerator:
    """Generates simulated student reactions for testing."""
    
    def __init__(self):
        """Initialize the reaction generator."""
        self.model = dspy.LM("gpt-3.5-turbo")
        
        class ReactionSignature(dspy.Signature):
            """Signature for student reactions."""
            teaching_response: str
            student_profile: dict
            reaction: str
            questions: str
            misconceptions: str
        
        self.generator = dspy.ChainOfThought(ReactionSignature)
    
    def generate_reaction(self,
                         teaching_response: str,
                         student_profile: Dict[str, str]) -> Dict[str, str]:
        """Generate a simulated student reaction."""
        result = self.generator(
            teaching_response=teaching_response,
            student_profile=student_profile
        )
        
        return {
            'response': result.reaction,
            'questions': result.questions.split('\n'),
            'misconceptions': result.misconceptions.split('\n')
        }

class TeachingAnalyzer:
    """Analyzes teaching responses for quality and effectiveness."""
    
    def __init__(self):
        """Initialize the teaching analyzer."""
        self.model = dspy.LM("gpt-3.5-turbo")
        
        class AnalysisSignature(dspy.Signature):
            """Signature for teaching response analysis."""
            teaching_response: str
            student_profile: dict
            context: str
            clarity_score: float
            engagement_score: float
            accuracy_score: float
            suggestions: str
            
        self.analyzer = dspy.ChainOfThought(AnalysisSignature)
    
    def analyze_response(self,
                        teaching_response: str,
                        student_profile: Dict[str, str],
                        context: str = "") -> Dict[str, Any]:
        """Analyze a teaching response for quality metrics."""
        result = self.analyzer(
            teaching_response=teaching_response,
            student_profile=student_profile,
            context=context
        )
        
        return {
            'clarity_score': float(result.clarity_score),
            'engagement_score': float(result.engagement_score),
            'accuracy_score': float(result.accuracy_score),
            'suggestions': result.suggestions.split('\n')
        }

class DSPyLLMInterface:
    """Interface for DSPy-based LLM interactions."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the DSPy LLM interface."""
        self.model = dspy.LM(model_name)
        dspy.settings.configure(lm=self.model)
        
        self.reaction_generator = StudentReactionGenerator()
        self.teaching_analyzer = TeachingAnalyzer()
        
        # Create signature for teaching response generation
        class TeachingSignature(dspy.Signature):
            """Signature for teaching response generation."""
            student_query: str
            student_profile: dict
            context: str
            teaching_response: str
            follow_up_questions: str
            
        self.teaching_generator = dspy.ChainOfThought(TeachingSignature)
    
    def generate_teaching_response(self,
                                 student_query: str,
                                 student_profile: Dict[str, str],
                                 context: str = "") -> Dict[str, Any]:
        """Generate a teaching response to a student query."""
        result = self.teaching_generator(
            student_query=student_query,
            student_profile=student_profile,
            context=context
        )
        
        response_data = {
            'response': result.teaching_response,
            'follow_up_questions': result.follow_up_questions.split('\n')
        }
        
        # Analyze the teaching response
        analysis = self.teaching_analyzer.analyze_response(
            teaching_response=result.teaching_response,
            student_profile=student_profile,
            context=context
        )
        response_data.update({'analysis': analysis})
        
        # Generate simulated student reaction
        reaction = self.reaction_generator.generate_reaction(
            teaching_response=result.teaching_response,
            student_profile=student_profile
        )
        response_data.update({'student_reaction': reaction})
        
        return response_data

class EnhancedDSPyLLMInterface(DSPyLLMInterface):
    """Enhanced interface with advanced teaching capabilities."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize the enhanced DSPy LLM interface."""
        super().__init__(model_name)
        
        # Create signature for scaffolded learning
        class ScaffoldSignature(dspy.Signature):
            """Signature for scaffolded learning responses."""
            student_query: str
            student_profile: dict
            context: str
            prerequisites: str
            concept_breakdown: str
            examples: str
            practice_problems: str
            
        self.scaffold_generator = dspy.ChainOfThought(ScaffoldSignature)
        
        # Create signature for metacognitive prompts
        class MetacognitiveSignature(dspy.Signature):
            """Signature for metacognitive prompts."""
            student_response: str
            student_profile: dict
            reflection_prompts: str
            learning_strategies: str
            self_assessment_questions: str
            
        self.metacognitive_generator = dspy.ChainOfThought(MetacognitiveSignature)
    
    def generate_scaffolded_response(self,
                                   student_query: str,
                                   student_profile: Dict[str, str],
                                   context: str = "") -> Dict[str, Any]:
        """Generate a scaffolded learning response."""
        result = self.scaffold_generator(
            student_query=student_query,
            student_profile=student_profile,
            context=context
        )
        
        return {
            'prerequisites': result.prerequisites.split('\n'),
            'concept_breakdown': result.concept_breakdown.split('\n'),
            'examples': result.examples.split('\n'),
            'practice_problems': result.practice_problems.split('\n')
        }
    
    def generate_metacognitive_prompts(self,
                                     student_response: str,
                                     student_profile: Dict[str, str]) -> Dict[str, Any]:
        """Generate metacognitive prompts to enhance learning."""
        result = self.metacognitive_generator(
            student_response=student_response,
            student_profile=student_profile
        )
        
        return {
            'reflection_prompts': result.reflection_prompts.split('\n'),
            'learning_strategies': result.learning_strategies.split('\n'),
            'self_assessment_questions': result.self_assessment_questions.split('\n')
        }
    
    def generate_comprehensive_response(self,
                                     student_query: str,
                                     student_profile: Dict[str, str],
                                     context: str = "") -> Dict[str, Any]:
        """Generate a comprehensive teaching response with scaffolding and metacognition."""
        # Get basic teaching response
        response_data = self.generate_teaching_response(
            student_query=student_query,
            student_profile=student_profile,
            context=context
        )
        
        # Add scaffolded learning components
        scaffolding = self.generate_scaffolded_response(
            student_query=student_query,
            student_profile=student_profile,
            context=context
        )
        response_data.update({'scaffolding': scaffolding})
        
        # Add metacognitive components based on simulated student reaction
        if 'student_reaction' in response_data:
            metacognition = self.generate_metacognitive_prompts(
                student_response=response_data['student_reaction']['response'],
                student_profile=student_profile
            )
            response_data.update({'metacognition': metacognition})
        
        return response_data

# Add backward compatibility for code that imports LLMInterface
# This ensures that existing code referencing the old interface will continue to work
DSPyLLMInterface = EnhancedDSPyLLMInterface 

class PedagogicalLanguageProcessor:
    """
    A specialized language processor for educational interactions using DSPy.
    Handles teaching-specific language tasks and pedagogical adaptations.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the pedagogical language processor.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize DSPy components
        try:
            dspy.settings.configure(lm=model_name)
            self.logger.info(f"Initialized DSPy with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize DSPy: {str(e)}")
            raise
    
    def process_teaching_response(self, 
                                student_query: str,
                                context: Dict[str, Any],
                                teaching_style: Optional[str] = None) -> str:
        """
        Process a teaching response using pedagogical principles.
        
        Args:
            student_query: The student's question or input
            context: Teaching context including subject, grade level, etc.
            teaching_style: Optional specific teaching style to use
            
        Returns:
            Processed teaching response
        """
        try:
            # Define the teaching response signature
            class TeachingResponse(dspy.Signature):
                """Signature for generating teaching responses."""
                context = dspy.InputField(desc="Teaching context and student information")
                query = dspy.InputField(desc="Student's question or input")
                style = dspy.InputField(desc="Teaching style to use")
                response = dspy.OutputField(desc="Pedagogically appropriate response")
            
            # Create the DSPy program
            class TeachingProgram(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.gen = dspy.ChainOfThought(TeachingResponse)
                
                def forward(self, context, query, style):
                    pred = self.gen(context=context, query=query, style=style)
                    return pred.response
            
            # Initialize and run the program
            program = TeachingProgram()
            response = program(
                context=str(context),
                query=student_query,
                style=teaching_style or "adaptive"
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing teaching response: {str(e)}")
            raise
    
    def analyze_student_understanding(self,
                                    student_response: str,
                                    expected_concepts: List[str]) -> Dict[str, float]:
        """
        Analyze student's understanding based on their response.
        
        Args:
            student_response: The student's response to analyze
            expected_concepts: List of concepts that should be present
            
        Returns:
            Dictionary of concept understanding scores
        """
        try:
            # Define the analysis signature
            class UnderstandingAnalysis(dspy.Signature):
                """Signature for analyzing student understanding."""
                response = dspy.InputField(desc="Student's response")
                concepts = dspy.InputField(desc="Expected concepts to check for")
                scores = dspy.OutputField(desc="Understanding scores for each concept")
            
            # Create the DSPy program
            class AnalysisProgram(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.analyze = dspy.ChainOfThought(UnderstandingAnalysis)
                
                def forward(self, response, concepts):
                    pred = self.analyze(response=response, concepts=concepts)
                    return pred.scores
            
            # Initialize and run the program
            program = AnalysisProgram()
            scores = program(
                response=student_response,
                concepts=str(expected_concepts)
            )
            
            # Convert scores to dictionary
            understanding_scores = {}
            for concept, score in zip(expected_concepts, scores.split(",")):
                understanding_scores[concept] = float(score)
            
            return understanding_scores
            
        except Exception as e:
            self.logger.error(f"Error analyzing student understanding: {str(e)}")
            raise
    
    def generate_scaffolding(self,
                           concept: str,
                           current_understanding: float,
                           student_profile: Dict[str, Any]) -> List[str]:
        """
        Generate scaffolding steps based on student's current understanding.
        
        Args:
            concept: The concept to scaffold
            current_understanding: Current understanding score (0-1)
            student_profile: Student's learning profile
            
        Returns:
            List of scaffolding steps
        """
        try:
            # Define the scaffolding signature
            class ScaffoldingPlan(dspy.Signature):
                """Signature for generating scaffolding steps."""
                concept = dspy.InputField(desc="Concept to scaffold")
                understanding = dspy.InputField(desc="Current understanding level")
                profile = dspy.InputField(desc="Student's learning profile")
                steps = dspy.OutputField(desc="List of scaffolding steps")
            
            # Create the DSPy program
            class ScaffoldingProgram(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.plan = dspy.ChainOfThought(ScaffoldingPlan)
                
                def forward(self, concept, understanding, profile):
                    pred = self.plan(
                        concept=concept,
                        understanding=str(understanding),
                        profile=str(profile)
                    )
                    return pred.steps
            
            # Initialize and run the program
            program = ScaffoldingProgram()
            steps = program(
                concept=concept,
                understanding=current_understanding,
                profile=student_profile
            )
            
            # Convert steps string to list
            scaffolding_steps = [
                step.strip()
                for step in steps.split("\n")
                if step.strip()
            ]
            
            return scaffolding_steps
            
        except Exception as e:
            self.logger.error(f"Error generating scaffolding: {str(e)}")
            raise

    def _create_signatures(self) -> Dict[str, dspy.Signature]:
        """Create DSPy signatures for different tasks."""
        class GenerateSignature(dspy.Signature):
            """Signature for basic generation."""
            prompt: str
            response: str
        
        class ExplainSignature(dspy.Signature):
            """Signature for teaching explanations."""
            topic: str
            student_level: str
            context: str
            explanation: str
            examples: str
            questions: str
        
        class ScaffoldSignature(dspy.Signature):
            """Signature for scaffolding."""
            student_response: str
            learning_objective: str
            current_level: str
            feedback: str
            next_step: str
            support_needed: str
        
        return {
            'generate': GenerateSignature,
            'explain': ExplainSignature,
            'scaffold': ScaffoldSignature
        }

class DSPyLLMHandler:
    """
    A simplified wrapper for DSPy LLM handling functionality.
    This class serves as the main interface for applications 
    to interact with the DSPy-based LLM capabilities.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the DSPy LLM handler.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.interface = EnhancedDSPyLLMInterface(model_name=model_name)
        
    def generate(self, prompt):
        """
        Generate a response using the LLM.
        
        Args:
            prompt (str): The prompt to send to the LLM
            
        Returns:
            str: The generated response
        """
        # Convert simple prompt to messages format
        messages = [{"role": "user", "content": prompt}]
        return self.interface.get_llm_response(messages)
        
    def get_llm_response(self, messages, output_format=None):
        """
        Get a response from the LLM using a structured messages format.
        
        Args:
            messages (list): List of message dictionaries with role and content
            output_format (dict, optional): Expected output format
            
        Returns:
            str: The generated response
        """
        return self.interface.get_llm_response(messages, output_format)
        
    def generate_student_reaction(self, teacher_input, student_profile, scenario_context=""):
        """
        Generate a simulated student reaction to teacher input.
        
        Args:
            teacher_input (str): The teacher's input or question
            student_profile (dict): Profile of the student
            scenario_context (str): Additional context for the scenario
            
        Returns:
            str: The generated student reaction
        """
        return self.interface.generate_student_reaction(teacher_input, student_profile, scenario_context)
        
    def analyze_teaching_strategies(self, teacher_input, student_profile, scenario_context):
        """
        Analyze teaching strategies used in the teacher's input.
        
        Args:
            teacher_input (str): The teacher's input to analyze
            student_profile (dict): Profile of the student
            scenario_context (str): Additional context for the scenario
            
        Returns:
            dict: Analysis of the teaching strategies
        """
        return self.interface.analyze_teaching_strategies(teacher_input, student_profile, scenario_context) 