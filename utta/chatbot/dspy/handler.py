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

class StudentReactionGenerator(dspy.Module):
    """Module for generating realistic student reactions with conversation awareness"""
    
    def __init__(self):
        super().__init__()
        # Use the global DSPy configuration without reconfiguring
        if not dspy_config.is_configured:
            logging.warning("DSPy not configured in StudentReactionGenerator initialization")
            # Don't try to configure here, just warn about it
        
        # Set up the predictor
        self.generate_reaction = dspy.Predict(StudentResponse)
        
        # Log initialization
        logging.info("StudentReactionGenerator initialized")
    
    def forward(self, teacher_input, student_profile, scenario_context):
        """Generate a realistic student reaction to a teacher's input"""
        
        # Check if there's conversation history in the scenario context
        conversation_history = ""
        if isinstance(scenario_context, dict) and "conversation_history" in scenario_context:
            conversation_history = f"""
            ## CONVERSATION HISTORY:
            {scenario_context['conversation_history']}
            """
        
        # Check if there's a previous response to avoid repeating
        previous_response_instruction = ""
        if isinstance(scenario_context, dict) and "previous_response" in scenario_context:
            previous_response_instruction = f"""
            ## YOUR PREVIOUS RESPONSE:
            {scenario_context['previous_response']}
            
            IMPORTANT: Your new response MUST be different from your previous response.
            Do not repeat the same ideas or phrases. Continue the conversation naturally.
            """
        
        # Extract student name if available
        student_name = ""
        if isinstance(student_profile, dict) and "name" in student_profile:
            student_name = student_profile["name"]
        elif isinstance(student_profile, str):
            try:
                profile_dict = json.loads(student_profile)
                student_name = profile_dict.get("name", "")
            except json.JSONDecodeError:
                pass
        
        # Create a personalized intro if student name is available
        personalized_intro = f"You are a student named {student_name}." if student_name else "You are a student."
        
        # Create a detailed prompt for the LLM to simulate a student reaction
        prompt = f"""
        {personalized_intro} Based on the teacher's input, respond authentically as this student would.
        
        ## TEACHER'S MOST RECENT INPUT:
        {teacher_input}
        
        ## YOUR STUDENT PROFILE:
        {student_profile}
        
        ## SCENARIO CONTEXT:
        {scenario_context}
        
        {conversation_history}
        
        {previous_response_instruction}
        
        ## IMPORTANT INSTRUCTIONS:
        1. Respond ONLY as the student character - NEVER as an AI assistant.
        2. DO NOT describe yourself or your characteristics - BE the student in your response.
        3. Use language appropriate for your age/grade level.
        4. Your response should reflect your learning style, challenges, and strengths.
        5. Keep the response concise and natural - as a real student would speak.
        6. Never break character or explain your reasoning.
        7. Respond directly to the teacher's specific question or statement.
        8. Don't repeat back the entire question.
        9. Make sure your response is DIFFERENT from any previous responses.
        10. If you're feeling frustrated, confused, or excited about the subject, show it naturally.
        """
        
        try:
            result = self.generate_reaction(prompt=prompt)
            
            # Extract just the response content, removing any additional formatting
            student_response = result.response.strip()
            
            # Remove any prefixes like "Student:" or "Me:" that might appear
            prefixes_to_remove = ["student:", "me:", "student response:", "response:", f"{student_name}:"]
            for prefix in prefixes_to_remove:
                if student_response.lower().startswith(prefix.lower()):
                    student_response = student_response[len(prefix):].strip()
            
            # Create a new result with the cleaned response
            clean_result = dspy.Prediction(response=student_response)
            return clean_result
            
        except Exception as e:
            logging.error(f"Error in StudentReactionGenerator: {e}", exc_info=True)
            # Return a basic response in case of error
            return dspy.Prediction(response=f"I'm not sure how to respond to that. Can you explain it differently?")

class TeachingAnalyzer(dspy.Module):
    """Module for analyzing teaching approaches"""
    
    def __init__(self):
        super().__init__()
        self.analyze = Predict(TeachingAnalysis)
    
    def forward(self, teacher_input, student_profile, scenario_context):
        """Analyze teaching strategies used in the teacher's input"""
        prompt = f"""
        Analyze the following teaching approach:
        
        "{teacher_input}"
        
        STUDENT PROFILE:
        {student_profile}
        
        SCENARIO CONTEXT:
        {scenario_context}
        
        Identify the teaching strategies used and their potential effectiveness for this student.
        """
        return self.analyze(prompt=prompt)

class DSPyLLMInterface:
    """
    Base interface for LLM communication using DSPy.
    
    This class handles initialization of the DSPy language model and
    provides methods for generating responses and recommendations.
    
    Attributes:
        model_name (str): Name of the LLM model to use
        lm: DSPy language model instance
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the LLM interface.
        
        Args:
            model_name (str): Name of the LLM model to use
        """
        self.model_name = model_name
        
        # Check if DSPy is already configured with a language model
        if dspy_config.is_configured:
            logging.info(f"Using existing DSPy configuration with model: {model_name}")
            self.lm = dspy_config.lm
        else:
            # Initialize a new model but don't configure DSPy settings yet
            self.lm = self._initialize_model(model_name)
    
    def configure_dspy_settings(self):
        """
        Configure DSPy settings with the language model.
        This should be called once from the main thread.
        """
        # Use the singleton configuration manager to configure DSPy
        return dspy_config.configure_dspy_settings(self.model_name)
    
    def _initialize_model(self, model_name):
        """
        Initialize the appropriate language model based on the model name.
        
        Args:
            model_name (str): Name of the model to initialize
            
        Returns:
            DSPy language model instance
        """
        # Handle different model types
        if "gpt" in model_name.lower():
            logging.info(f"Initializing OpenAI model: {model_name}")
            # Import OpenAIProvider, but use it as the provider parameter to dspy.LM
            from dspy.clients.openai import OpenAIProvider
            return dspy.LM(model=model_name, provider=OpenAIProvider(), temperature=0.7)
            
        elif "claude" in model_name.lower():
            logging.info(f"Initializing Anthropic model: {model_name}")
            from dspy.clients.anthropic import AnthropicProvider
            return dspy.LM(model=model_name, provider=AnthropicProvider(), temperature=0.7)
            
        elif "llama-3" in model_name.lower() or "llama3" in model_name.lower():
            # Try to use Ollama for Llama 3 models
            try:
                logging.info("Attempting to initialize Llama 3 with Ollama...")
                # Convert model name format for Ollama (llama-3-8b → llama3:8b)
                ollama_model_name = "llama3:8b"
                if "70b" in model_name.lower():
                    ollama_model_name = "llama3:70b"
                    
                logging.info(f"Using Ollama model: {ollama_model_name}")
                
                # Try to use litellm directly
                try:
                    import litellm
                    logging.info("Initializing Ollama through litellm")
                    
                    # Create a custom provider that uses litellm
                    class LiteLLMProvider:
                        def __call__(self, prompt, **kwargs):
                            try:
                                response = litellm.completion(
                                    model="ollama/"+ollama_model_name,
                                    messages=[{"role": "user", "content": prompt}],
                                    temperature=0.7
                                )
                                return response.choices[0].message.content
                            except Exception as e:
                                logging.error(f"LiteLLM error: {e}")
                                return "Error generating response with Ollama."
                    
                    # Include the model parameter as required by dspy.LM
                    return dspy.LM(model=ollama_model_name, provider=LiteLLMProvider(), temperature=0.7)
                    
                except ImportError as e:
                    logging.warning(f"litellm not available: {e}")
                    # Fall back to OpenAI
                    logging.info("Falling back to default model")
                    from dspy.clients.openai import OpenAIProvider
                    return dspy.LM(model="gpt-3.5-turbo", provider=OpenAIProvider(), temperature=0.7)
                
            except Exception as e:
                logging.warning(f"Error initializing Ollama: {e}")
                logging.info("Falling back to default model")
                from dspy.clients.openai import OpenAIProvider
                return dspy.LM(model="gpt-3.5-turbo", provider=OpenAIProvider(), temperature=0.7)
        else:
            # Default to GPT-3.5-turbo
            logging.info("Defaulting to OpenAI GPT-3.5-turbo model")
            from dspy.clients.openai import OpenAIProvider
            return dspy.LM(model="gpt-3.5-turbo", provider=OpenAIProvider(), temperature=0.7)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def get_llm_response(self, messages, output_format=None):
        """
        Get a response from the language model with retry logic.
        
        Args:
            messages: List of message dictionaries with role and content
            output_format: Optional format specification (json, markdown, etc.)
            
        Returns:
            str: The model's response
        """
        try:
            # Convert message dictionaries to a prompt string
            prompt = self._messages_to_prompt(messages, output_format)
            
            # Call the LM object directly with the prompt
            result = self.lm(prompt)
            
            # Ensure we return a string (DSPy might return a list with one item)
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return result
            
        except Exception as e:
            logging.error(f"Failed to get LLM response: {e}")
            return f"Error: Unable to generate response. Please try again later."
    
    def _messages_to_prompt(self, messages, output_format=None):
        """Convert message dictionaries to a prompt string"""
        prompt_parts = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"Instructions: {content}\n")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}\n")
            else:  # Default to user
                prompt_parts.append(f"User: {content}\n")
        
        # Add output format instruction if needed
        if output_format == "json":
            prompt_parts.append("Return your response in valid JSON format.")
        
        return "\n".join(prompt_parts)
    
    def generate_teaching_recommendation(self, scenario, student_profile):
        """
        Generate teaching recommendations based on scenario and student profile.
        
        Args:
            scenario: Teaching scenario details
            student_profile: Student characteristics and needs
            
        Returns:
            dict: Teaching recommendations including strategies and approaches
        """
        prompt = f"""
        You are an expert education consultant providing teaching recommendations.
        
        SCENARIO:
        {json.dumps(scenario, indent=2)}
        
        STUDENT PROFILE:
        {json.dumps(student_profile, indent=2)}
        
        Please provide teaching recommendations appropriate for this scenario and student.
        Include specific strategies, approaches, and potential adaptations.
        Return your response as JSON with the following structure:
        {{
            "recommended_strategies": [list of strategies],
            "adaptations": [list of adaptations for student needs],
            "communication_approach": "description of effective communication",
            "assessment_methods": [list of appropriate assessment approaches]
        }}
        """
        
        response = self.get_llm_response([{"role": "user", "content": prompt}], output_format="json")
        
        try:
            return json.loads(response)
        except:
            # If JSON parsing fails, return a structured dictionary with the raw response
            return {
                "recommended_strategies": ["Error parsing recommendations"],
                "adaptations": [],
                "communication_approach": response,
                "assessment_methods": []
            }


class EnhancedDSPyLLMInterface(DSPyLLMInterface):
    """
    Extension of the DSPyLLMInterface with enhanced capabilities.
    Provides specialized modules for educational tasks.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo"):
        """
        Initialize the enhanced DSPy interface with specialized modules.
        
        Args:
            model_name: The name of the model to use
        """
        super().__init__(model_name)
        
        # Ensure DSPy configuration immediately upon initialization
        success = dspy_config.ensure_configuration(model_name)
        if not success:
            logging.warning("Failed to configure DSPy on initialization; will retry when needed")
            
        # Initialize specialized modules
        self.modules = {}
        self.ensure_modules_initialized()

    def ensure_modules_initialized(self):
        """Make sure specialized modules are initialized if DSPy is configured."""
        if dspy_config.is_configured and not self.modules:
            self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize specialized DSPy modules for educational tasks."""
        try:
            # Ensure DSPy configuration is in place first
            if not dspy_config.is_configured:
                logging.info("Configuring DSPy before initializing modules")
                success = dspy_config.configure_dspy_settings(self.model_name)
                if not success:
                    logging.error("Failed to configure DSPy before initializing modules")
                    self.modules = {}
                    return
            
            # Make sure DSPy is configured with our language model
            dspy.configure(lm=dspy_config.lm)
            
            logging.info("Initializing specialized DSPy modules")
            self.modules["teacher_responder"] = TeacherResponder()
            self.modules["student_reaction"] = StudentReactionGenerator()
            self.modules["teaching_analyzer"] = TeachingAnalyzer()
            logging.info("Successfully initialized specialized DSPy modules")
        except Exception as e:
            logging.error(f"Error initializing DSPy modules: {e}")
            # Initialize with empty dict to prevent repeated initialization attempts
            self.modules = {}
    
    def get_llm_response_with_context(self, messages, context_data, prompt_type="student_simulation"):
        """
        Get a response tailored to a specific educational context.
        
        Args:
            messages: List of message dictionaries
            context_data: Contextual information relevant to the request
            prompt_type: Type of prompt to use (student_simulation, teaching_analysis, etc.)
            
        Returns:
            str: Context-aware response from the LLM
        """
        # Make sure modules are initialized if DSPy is configured
        self.ensure_modules_initialized()
        
        # Check if DSPy is configured
        if not dspy_config.is_configured:
            logging.error("DSPy is not configured. Cannot generate contextual response.")
            return "Error: DSPy is not configured. Please initialize the LLM first."
        
        try:
            # Use standard DSPy prediction based on the modules
            if prompt_type == "student_simulation" and "student_reaction" in self.modules:
                return self.generate_student_response(
                    messages[-1]["content"], 
                    context_data.get("student_profile", {}),
                    context_data
                )
            elif prompt_type == "teaching_analysis" and "teaching_analyzer" in self.modules:
                analysis = self.analyze_teaching_strategies(
                    messages[-1]["content"],
                    context_data.get("student_profile", {}),
                    context_data
                )
                return json.dumps(analysis, indent=2)
            else:
                # Fallback to standard response
                return self.get_llm_response(messages)
        except Exception as e:
            logging.error(f"Error getting contextual response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_student_response(self, teacher_input, student_profile, scenario_context):
        """
        Generate a realistic student response to a teacher's input.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Student characteristics
            scenario_context: Additional context about the scenario
            
        Returns:
            str: A simulated student response
        """
        # Make sure modules are initialized if DSPy is configured
        self.ensure_modules_initialized()
        
        # Check if DSPy is configured, and configure it if not
        if not dspy_config.is_configured:
            success = dspy_config.ensure_configuration(self.model_name)
            if not success:
                logging.error("DSPy is not configured and configuration failed. Cannot generate student response.")
                return "I'm sorry, I can't respond right now. [DSPy configuration error]"
            
        try:
            # Convert inputs to strings if they're dictionaries
            student_profile_str = json.dumps(student_profile) if isinstance(student_profile, dict) else str(student_profile)
            scenario_context_str = json.dumps(scenario_context) if isinstance(scenario_context, dict) else str(scenario_context)
            
            if "student_reaction" not in self.modules:
                logging.error("Student reaction generator module not initialized.")
                self.ensure_modules_initialized()  # Try to initialize again
                if "student_reaction" not in self.modules:
                    return "I'm not sure how to respond to that. [Module initialization error]"
                
            result = self.modules["student_reaction"](
                teacher_input=teacher_input, 
                student_profile=student_profile_str, 
                scenario_context=scenario_context_str
            )
            
            return result.response
        except Exception as e:
            logging.error(f"Error generating student response: {e}")
            return "I'm not sure how to respond to that right now. [Error: " + str(e) + "]"
    
    def analyze_teaching_strategies(self, teacher_input, student_profile, scenario_context):
        """
        Analyze teaching strategies in the teacher's input.
        
        Args:
            teacher_input: The teacher's statement or approach
            student_profile: Student characteristics
            scenario_context: Additional context about the scenario
            
        Returns:
            dict: Analysis of teaching strategies
        """
        # Make sure modules are initialized if DSPy is configured
        self.ensure_modules_initialized()
        
        # Check if DSPy is configured
        if not dspy_config.is_configured:
            logging.error("DSPy is not configured. Cannot analyze teaching strategies.")
            return {
                "strengths": ["Unable to analyze - DSPy not configured"],
                "areas_for_improvement": ["Please initialize the LLM first"],
                "effectiveness_score": 0,
                "rationale": "DSPy is not configured yet. Please initialize the LLM first."
            }
            
        try:
            # Convert inputs to strings if they're dictionaries
            student_profile_str = json.dumps(student_profile) if isinstance(student_profile, dict) else str(student_profile)
            scenario_context_str = json.dumps(scenario_context) if isinstance(scenario_context, dict) else str(scenario_context)
            
            if "teaching_analyzer" not in self.modules:
                logging.error("Teaching analyzer module not initialized.")
                return {
                    "strengths": ["Unable to analyze - module not initialized"],
                    "areas_for_improvement": ["Please try again"],
                    "effectiveness_score": 0,
                    "rationale": "Teaching analyzer module not initialized."
                }
                
            result = self.modules["teaching_analyzer"](
                teacher_input=teacher_input,
                student_profile=student_profile_str,
                scenario_context=scenario_context_str
            )
            
            return {
                "strengths": result.strengths,
                "areas_for_improvement": result.areas_for_improvement,
                "effectiveness_score": result.effectiveness_score,
                "rationale": result.rationale
            }
        except Exception as e:
            logging.error(f"Error analyzing teaching strategies: {e}")
            return {
                "strengths": ["Error during analysis"],
                "areas_for_improvement": ["Unable to complete analysis"],
                "effectiveness_score": 0,
                "rationale": f"Error during analysis: {str(e)}"
            }

    def generate_student_reaction(self, teacher_input, student_profile, scenario_context=""):
        """
        Generate a realistic student reaction based on teacher input and student profile.
        
        Args:
            teacher_input: The teacher's question or statement
            student_profile: Dictionary or JSON string with student profile
            scenario_context: Dictionary or JSON string with scenario context
            
        Returns:
            str: The student's reaction
        """
        # Make sure DSPy is configured before proceeding
        if not dspy_config.is_configured:
            logging.error("DSPy is not configured yet. Cannot generate student reaction.")
            return "I don't know how to respond yet."
        
        try:
            # Ensure student_profile is a string for the generator
            if isinstance(student_profile, dict):
                student_profile = json.dumps(student_profile)
                
            # Ensure scenario_context is a string for the generator
            if isinstance(scenario_context, dict):
                scenario_context = json.dumps(scenario_context)
            
            # Generate student reaction
            result = self.modules["student_reaction"](
                teacher_input=teacher_input,
                student_profile=student_profile,
                scenario_context=scenario_context
            )
            
            return result.response
        except Exception as e:
            logging.error(f"Error generating student reaction: {e}")
            # Return a fallback response in case of error
            return "I'm not sure how to respond to that."

# Add backward compatibility for code that imports LLMInterface
# This ensures that existing code referencing the old interface will continue to work
DSPyLLMInterface = EnhancedDSPyLLMInterface 

class PedagogicalLanguageProcessor:
    """
    Process teaching language and generate educational responses.
    
    This class provides specialized language processing for educational scenarios,
    including teaching response analysis, student reaction generation, and
    scenario creation.
    """
    
    def __init__(self, model="gpt-3.5-turbo"):
        """
        Initialize the processor.
        
        Args:
            model: Name of the language model to use
        """
        logging.info(f"Initializing pedagogical processor with model: {model}")
        
        # Use the globally configured DSPy settings
        self.model_name = model
        
        # Initialize DSPy modules for specific educational tasks
        self.teacher_responder = TeacherResponder()
        self.student_reaction_generator = StudentReactionGenerator()
        self.teaching_analyzer = TeachingAnalyzer()
        
    def create_scenario(self, context):
        """
        Create a teaching scenario based on the provided context.
        
        Args:
            context: Dictionary containing scenario parameters
            
        Returns:
            dict: A structured teaching scenario
        """
        # Make sure DSPy is configured before proceeding
        if not dspy_config.is_configured:
            logging.error("DSPy is not configured yet. Cannot create scenario.")
            return {"error": "DSPy is not configured yet. Please initialize the LLM first."}
            
        # Create a prompt for scenario generation
        subject = context.get("subject", "mathematics")
        grade_level = context.get("grade_level", "3rd grade")
        
        prompt = f"""
        Create a realistic classroom teaching scenario for a {grade_level} {subject} class.
        The scenario should:
        1. Describe a specific teaching situation or challenge
        2. Include 1-2 students with specific learning needs or behaviors
        3. Provide background context on what was being taught
        4. Present a decision point where the teacher needs to respond
        """
        
        try:
            # Use the DSPy module directly
            result = self.teacher_responder(
                context=json.dumps(context), 
                student_profile="", 
                question=prompt
            )
            
            # Check if we got a structured response
            if hasattr(result, 'scenario_title'):
                # We have a structured scenario response
                logging.info("Received structured scenario response")
                return {
                    "scenario_title": result.scenario_title,
                    "scenario_description": result.scenario_description,
                    "learning_objectives": result.learning_objectives,
                    "student_background": result.student_background,
                    "teacher_challenge": result.teacher_challenge
                }
            
            # Otherwise handle the text response
            logging.info("Received text scenario response, attempting to parse JSON")
            response_text = result.response.strip()
            
            # Log the response for debugging
            logging.info(f"Raw scenario response: {response_text}")
            
            # Try to parse as JSON
            try:
                scenario_dict = json.loads(response_text)
                return scenario_dict
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}")
                
                # Try to find and extract JSON if it's embedded in other text
                try:
                    # Look for JSON object between curly braces
                    import re
                    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                        result = json.loads(json_text)
                        return result
                except:
                    pass
                
                # Fallback: create a structured response from the text
                return {
                    "scenario_title": "Classroom Challenge",
                    "scenario_description": response_text,
                    "learning_objectives": ["Understand the subject material"],
                    "student_background": "Students with various learning needs",
                    "teacher_challenge": "Addressing student needs effectively"
                }
                
        except Exception as e:
            logging.error(f"Error creating scenario: {e}")
            # Return a fallback scenario
            return {
                "scenario_title": "Error in Scenario Generation",
                "scenario_description": "There was an error generating the scenario. Please try again.",
                "learning_objectives": ["N/A"],
                "student_background": "N/A",
                "teacher_challenge": "N/A"
            }
            
    def analyze_teaching_response(self, teacher_input, context):
        """
        Analyze a teaching response for strengths and areas of improvement.
        
        Args:
            teacher_input: The teacher's statement or approach
            context: Contextual information about the scenario
            
        Returns:
            dict: Analysis of the teaching approach
        """
        # Make sure DSPy is configured before proceeding
        if not dspy_config.is_configured:
            logging.error("DSPy is not configured yet. Cannot analyze teaching response.")
            return {
                "strengths": ["Unable to analyze - DSPy not configured"],
                "areas_for_improvement": ["Please initialize the LLM first"],
                "effectiveness_score": 0,
                "rationale": "DSPy is not configured yet. Please initialize the LLM first."
            }
            
        try:
            # Use the DSPy module directly
            prediction = self.teaching_analyzer(teacher_input=teacher_input, 
                                              student_profile=json.dumps(context.get("student_profile", {})), 
                                              scenario_context=json.dumps(context))
            
            # Create the analysis dictionary
            analysis = {
                "strengths": prediction.strengths,
                "areas_for_improvement": prediction.areas_for_improvement,
                "effectiveness_score": prediction.effectiveness_score,
                "rationale": prediction.rationale
            }
            
            return analysis
        except Exception as e:
            logging.error(f"Error analyzing teaching response: {e}")
            return {
                "strengths": ["Error during analysis"],
                "areas_for_improvement": ["Unable to complete analysis"],
                "effectiveness_score": 0,
                "rationale": f"Error during analysis: {str(e)}"
            }
    
    def generate_student_reaction(self, teacher_input, student_profile, scenario_context=""):
        """
        Generate a realistic student reaction based on teacher input and student profile.
        
        Args:
            teacher_input: The teacher's question or statement
            student_profile: Dictionary or JSON string with student profile
            scenario_context: Dictionary or JSON string with scenario context
            
        Returns:
            str: The student's reaction
        """
        # Make sure DSPy is configured before proceeding
        if not dspy_config.is_configured:
            logging.error("DSPy is not configured yet. Cannot generate student reaction.")
            return "I don't know how to respond yet."
        
        try:
            # Ensure student_profile is a string for the generator
            if isinstance(student_profile, dict):
                student_profile = json.dumps(student_profile)
                
            # Ensure scenario_context is a string for the generator
            if isinstance(scenario_context, dict):
                scenario_context = json.dumps(scenario_context)
            
            # Generate student reaction
            result = self.student_reaction_generator(
                teacher_input=teacher_input,
                student_profile=student_profile,
                scenario_context=scenario_context
            )
            
            return result.response
        except Exception as e:
            logging.error(f"Error generating student reaction: {e}")
            # Return a fallback response in case of error
            return "I'm not sure how to respond to that." 

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