"""
DSPy Adapter Module

This adapter module provides compatibility between the existing application's
LangChain-based code and the new DSPy implementation. It ensures that the
application can use our enhanced DSPy features without requiring modifications
to the existing codebase.

The adapter maintains the same interface expected by the application while
delegating the actual processing to the DSPy implementation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Import the DSPy implementation
from .handler import EnhancedDSPyLLMInterface, PedagogicalLanguageProcessor
import dspy

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class LLMInterface:
    """
    Legacy LLMInterface that adapts to the new DSPy implementation.
    
    This class maintains the same interface as the original LLMInterface
    while delegating the actual work to our DSPy implementation.
    """
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.7):
        """Initialize the legacy interface with the DSPy implementation."""
        self.model_name = model_name
        self.temperature = temperature
        self.chat_model = None
        
        # Initialize the DSPy implementation
        logging.info(f"Initializing DSPy LLM Interface with model: {model_name}")
        self.dspy_interface = EnhancedDSPyLLMInterface(model_name=model_name)
    
    def _initialize_model(self, model_name):
        """
        Initialize the appropriate language model.
        This method exists for compatibility with the original interface.
        The actual initialization is handled by the DSPy implementation.
        """
        try:
            # We already initialized the model in __init__, so just log it
            logging.info(f"Model {model_name} already initialized in DSPy implementation")
            return self.dspy_interface.lm
        except Exception as e:
            logging.error(f"Error initializing model: {e}")
            raise e
    
    def get_chat_response(self, messages, stream=False):
        """
        Get a response from the chat model.
        
        Args:
            messages: List of message dictionaries with role and content
            stream: Whether to stream the response (ignored in this implementation)
            
        Returns:
            str: The model's response
        """
        try:
            # Use the DSPy implementation to get the response
            logging.info("Getting chat response using DSPy implementation")
            return self.dspy_interface.get_llm_response(messages)
        except Exception as e:
            logging.error(f"Error getting chat response: {e}")
            return "I'm sorry, I encountered an error while processing your request."
    
    def generate_with_retry(self, messages, max_retries=3):
        """
        Generate a response with retry logic.
        
        Args:
            messages: List of message dictionaries
            max_retries: Maximum number of retries
            
        Returns:
            str: The model's response
        """
        # DSPy implementation already includes retry logic
        return self.dspy_interface.get_llm_response(messages)


class EnhancedLLMInterface(LLMInterface):
    """Extension of the LLMInterface with enhanced capabilities."""
    
    def __init__(self, 
                 model_name: str = "gpt-3.5-turbo",
                 cache_dir: Optional[Path] = None):
        """
        Initialize the enhanced LLM interface.
        
        Args:
            model_name: Name of the LLM model to use
            cache_dir: Optional directory for caching responses
        """
        super().__init__(model_name, 0.7)
        self.logger = logging.getLogger(__name__)
        self.cache_dir = cache_dir or Path("cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize DSPy
        try:
            dspy.settings.configure(lm=model_name)
            self.logger.info(f"Initialized DSPy with model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize DSPy: {str(e)}")
            raise
        
        # Load cache if it exists
        self.response_cache = self.load_cache()
    
    def load_cache(self) -> Dict[str, Any]:
        """Load the response cache from disk."""
        cache_file = self.cache_dir / "response_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache: {str(e)}")
                return {}
        return {}
    
    def save_cache(self):
        """Save the response cache to disk."""
        cache_file = self.cache_dir / "response_cache.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(self.response_cache, f)
        except Exception as e:
            self.logger.error(f"Error saving cache: {str(e)}")
    
    def get_cache_key(self, messages: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request."""
        import hashlib
        request_str = json.dumps({"messages": messages, "context": context}, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def get_response(self, 
                    messages: List[Dict[str, str]], 
                    context: Optional[Dict[str, Any]] = None,
                    use_cache: bool = True) -> str:
        """
        Get a response from the LLM with optional caching.
        
        Args:
            messages: List of message dictionaries
            context: Optional context dictionary
            use_cache: Whether to use response caching
            
        Returns:
            The LLM's response
        """
        context = context or {}
        
        # Check cache if enabled
        if use_cache:
            cache_key = self.get_cache_key(messages, context)
            if cache_key in self.response_cache:
                self.logger.info("Using cached response")
                return self.response_cache[cache_key]
        
        try:
            # Define the response signature
            class ChatResponse(dspy.Signature):
                """Signature for chat responses."""
                messages = dspy.InputField(desc="Chat messages")
                context = dspy.InputField(desc="Additional context")
                response = dspy.OutputField(desc="Generated response")
            
            # Create the DSPy program
            class ChatProgram(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.gen = dspy.ChainOfThought(ChatResponse)
                
                def forward(self, messages, context):
                    pred = self.gen(messages=messages, context=context)
                    return pred.response
            
            # Initialize and run the program
            program = ChatProgram()
            response = program(
                messages=str(messages),
                context=str(context)
            )
            
            # Cache the response if enabled
            if use_cache:
                self.response_cache[cache_key] = response
                self.save_cache()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting response: {str(e)}")
            raise
    
    def set_pedagogical_processor(self, processor):
        """Set the pedagogical processor for educational responses."""
        self.pedagogical_processor = processor
        logging.info("Pedagogical processor set successfully")
    
    def get_streaming_response(self, messages):
        """
        Get a streaming response from the chat model.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generator yielding partial responses
        """
        # Use standard non-streaming response as a fallback
        response = self.dspy_interface.get_llm_response(messages)
        yield response
    
    def analyze_teaching_approach(self, teaching_input, context):
        """
        Analyze a teaching approach for effectiveness.
        
        Args:
            teaching_input: The teacher's statement or explanation
            context: Additional context and student information
            
        Returns:
            dict: Analysis of the teaching approach
        """
        if self.pedagogical_processor:
            return self.pedagogical_processor.analyze_teaching_approach(teaching_input, context)
        else:
            logging.warning("Pedagogical processor not available; using direct LLM call")
            prompt = f"""
            Analyze this teaching approach:
            
            Teaching input: {teaching_input}
            
            Context: {json.dumps(context)}
            
            Provide a detailed analysis of the effectiveness of this approach.
            """
            response = self.get_chat_response([{"role": "user", "content": prompt}])
            return {"analysis": response}
    
    def analyze_teaching_strategies(self, teacher_input, student_profile, scenario_context):
        """
        Analyze teaching strategies for a specific student.
        
        Args:
            teacher_input: The teacher's statement or question
            student_profile: Student characteristics and needs
            scenario_context: Additional context about the scenario
            
        Returns:
            dict: Analysis of the teaching strategies
        """
        # Use DSPy implementation directly
        try:
            # Ensure student profile and scenario context are properly formatted
            if isinstance(student_profile, str):
                try:
                    student_profile = json.loads(student_profile)
                except:
                    student_profile = {"description": student_profile}
                    
            if isinstance(scenario_context, str):
                try:
                    scenario_context = json.loads(scenario_context)
                except:
                    scenario_context = {"description": scenario_context}
            
            # Call the DSPy interface
            return self.dspy_interface.analyze_teaching_strategies(
                teacher_input=teacher_input,
                student_profile=student_profile,
                scenario_context=scenario_context
            )
        except Exception as e:
            logging.error(f"Error analyzing teaching strategies: {e}")
            return {"error": str(e)}
    
    def generate_teaching_recommendation(self, scenario, student_profile):
        """
        Generate teaching recommendations based on scenario and student profile.
        
        Args:
            scenario: Teaching scenario details
            student_profile: Student characteristics and needs
            
        Returns:
            dict: Teaching recommendations
        """
        # Use DSPy implementation directly
        try:
            return self.dspy_interface.generate_teaching_recommendation(
                scenario=scenario,
                student_profile=student_profile
            )
        except Exception as e:
            logging.error(f"Error generating teaching recommendation: {e}")
            return {"error": str(e)}
    
    def simulate_student_response(self, teacher_input, student_profile, scenario_context):
        """
        Simulate a student's response to a teacher's input, taking into account
        the student profile and scenario context.
        
        Args:
            teacher_input (str): The teacher's input
            student_profile (dict or str): The student profile, containing information about
                the student's learning style, strengths, challenges, etc.
            scenario_context (dict or str): The scenario context, containing information about
                the subject, difficulty, grade level, etc.
            
        Returns:
            str: The simulated student response
        """
        logging.info("Simulating student response")
        
        try:
            # Special handling for the student response
            # Include conversation history and personalized elements
            
            # Add conversation history if available
            conversation_history = ""
            if isinstance(scenario_context, dict) and "conversation_history" in scenario_context:
                conversation_history = f"\n\nCONVERSATION HISTORY:\n{scenario_context['conversation_history']}"
            
            # Add additional instructions to avoid repetition
            special_instructions = """
            Respond as a student would in this scenario. Your response should be:
            1. Brief and natural (1-3 sentences)
            2. Age-appropriate for your grade level
            3. Reflect your learning style, challenges, and strengths
            4. Directly responsive to what the teacher just said
            5. NOT a self-description (don't say "As a student with...")
            """
            
            # Add previous response if available to prevent repetition
            previous_response = ""
            if isinstance(scenario_context, dict) and "previous_response" in scenario_context:
                previous_response = f"\n\nYour previous response was: '{scenario_context['previous_response']}'\nMake sure your new response is DIFFERENT and moves the conversation forward."
            
            # Personalize the instructions if the student profile includes a name
            student_name = student_profile.get("name", "") if isinstance(student_profile, dict) else ""
            if student_name:
                special_instructions = f"You are {student_name}. {special_instructions}"
            
            # Enhance the teacher input with the special instructions
            enhanced_input = f"{teacher_input}\n\n---\nINSTRUCTIONS: {special_instructions}{conversation_history}{previous_response}"
            
            # Try to use the pedagogical processor if available
            if hasattr(self, 'pedagogical_processor') and self.pedagogical_processor:
                logging.info("Using pedagogical processor for student response")
                student_response = self.pedagogical_processor.generate_student_reaction(
                    enhanced_input,
                    json.dumps(student_profile) if isinstance(student_profile, dict) else student_profile,
                    json.dumps(scenario_context) if isinstance(scenario_context, dict) else scenario_context
                )
            else:
                # Fallback to direct LLM call instead of using the non-existent _get_student_response method
                logging.info("Using direct LLM call for student response")
                if hasattr(self, 'dspy_interface'):
                    student_response = self.dspy_interface.generate_student_response(
                        enhanced_input,
                        student_profile,
                        scenario_context
                    )
                else:
                    # Final fallback using the generate_student_response method
                    student_response = self.generate_student_response(
                        enhanced_input,
                        student_profile,
                        scenario_context
                    )
            
            # Save this response in the session to avoid repetition in future responses
            if isinstance(scenario_context, dict) and "session_state" in scenario_context:
                scenario_context["session_state"]["last_student_response"] = student_response
            
            return student_response
            
        except Exception as e:
            logging.error(f"Error simulating student response: {e}", exc_info=True)
            return f"I'm not sure how to respond to that. [Error: {str(e)}]"
            
    def _get_student_response(self, teacher_input, student_profile, scenario_context):
        """
        Internal method to get a student response when pedagogical processor is not available.
        This is a fallback method to avoid AttributeError.
        
        Args:
            teacher_input (str): The teacher's input
            student_profile (dict or str): The student profile
            scenario_context (dict or str): The scenario context
            
        Returns:
            str: The simulated student response
        """
        # Generate a student response using standard methods
        try:
            # If we have DSPy interface available, use it
            if hasattr(self, 'dspy_interface'):
                return self.dspy_interface.generate_student_response(
                    teacher_input,
                    student_profile,
                    scenario_context
                )
            
            # Otherwise use the base generate_student_response method
            return self.generate_student_response(
                teacher_input, 
                student_profile, 
                scenario_context
            )
        except Exception as e:
            logging.error(f"Error in _get_student_response: {e}", exc_info=True)
            return "I'm not sure how to answer that question."


# Factory function for creating the appropriate interface
def create_llm_interface(model_name: str = "gpt-3.5-turbo",
                        cache_dir: Optional[Path] = None) -> EnhancedLLMInterface:
    """
    Create an enhanced LLM interface.
    
    Args:
        model_name: Name of the LLM model to use
        cache_dir: Optional directory for caching responses
        
    Returns:
        An initialized EnhancedLLMInterface instance
    """
    return EnhancedLLMInterface(model_name=model_name, cache_dir=cache_dir) 