"""DSPy LLM interface for teaching responses."""

import os
import logging
import importlib.util
import openai
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if dspy is available
HAVE_DSPY = False
try:
    # First try regular import
    import dspy
    HAVE_DSPY = True
    logger.info("Successfully imported dspy")
except ImportError as e:
    logger.error(f"DSPy import error: {e}")
    # Create a fallback module for graceful degradation
    class FakeDSPy:
        def __init__(self):
            class FakeField:
                def __call__(self, *args, **kwargs):
                    return None
            self.InputField = FakeField()
            self.OutputField = FakeField()
            
            class FakeSignature:
                pass
            self.Signature = FakeSignature
            
            class FakePredict:
                def __call__(self, *args, **kwargs):
                    class Result:
                        response = "I'm sorry, but the DSPy module is not available. Please install dspy-ai package."
                    return Result()
            self.Predict = FakePredict()
    
    dspy = FakeDSPy()
    logger.warning("Using fake DSPy implementation for graceful degradation")

class EnhancedDSPyLLMInterface:
    """Enhanced DSPy interface for teaching responses."""
    
    def __init__(self):
        """Initialize the DSPy interface."""
        self.initialized = False
        
        try:
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment")
            
            # Initialize OpenAI client directly
            openai.api_key = api_key
            
            # Set up teaching signature
            if HAVE_DSPY:
                self._create_signatures()
                self.initialized = True
                logger.info("Initialized DSPy LLM interface with OpenAI")
            else:
                logger.warning("DSPy not available, using fallback mode")
                self._create_fallback_signatures()
                
        except Exception as e:
            logger.error(f"Failed to initialize DSPy interface: {e}")
            self._create_fallback_signatures()

    def _create_signatures(self):
        """Create DSPy signatures for different tasks."""
        try:
            # Basic response generation signature
            class TeachingSignature(dspy.Signature):
                """Signature for teaching response generation."""
                query = dspy.InputField()
                context = dspy.InputField(desc="Reference content to use")
                metadata = dspy.InputField(desc="Teaching context like grade level")
                response = dspy.OutputField(desc="Teaching response")
            
            self.signatures = {
                "teaching": TeachingSignature
            }
            logger.info("Created DSPy signatures")
            
        except Exception as e:
            logger.error(f"Failed to create signatures: {e}")
            self._create_fallback_signatures()

    def _create_fallback_signatures(self):
        """Create fallback signatures when DSPy is not available."""
        class FallbackSignature:
            pass
            
        self.signatures = {
            "teaching": FallbackSignature
        }
        logger.warning("Using fallback signatures")

    def generate_response(self, 
                       prompt: str,
                       context: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a teaching response using DSPy or direct OpenAI API.
        
        Args:
            prompt: The teaching prompt/question
            context: Optional reference content
            metadata: Optional teaching context
            
        Returns:
            Generated teaching response
        """
        if not self.initialized:
            return "I'm sorry, but the teaching assistant is not fully initialized. Please check the logs for more information."
        
        try:
            # If DSPy is not working properly, fall back to direct OpenAI API call
            model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
            max_tokens = int(os.getenv("MAX_TOKENS", "1000"))
            temperature = float(os.getenv("TEMPERATURE", "0.7"))
            
            # Create a system message with the context if available
            system_content = "You are a helpful teaching assistant, providing thorough and educational responses."
            if context:
                context_str = "\n\n".join(context)
                system_content += f"\n\nUse the following information as reference:\n{context_str}"
            
            if metadata:
                meta_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])
                system_content += f"\n\nTeaching context:\n{meta_str}"
            
            # Call OpenAI API directly
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"I'm sorry, but I encountered an error: {str(e)}" 