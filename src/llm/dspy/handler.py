"""DSPy LLM interface for teaching responses."""

import os
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import dspy
    HAVE_DSPY = True
except ImportError:
    logger.error("DSPy not available. Please install dspy-ai")
    HAVE_DSPY = False

class EnhancedDSPyLLMInterface:
    """Enhanced DSPy interface for teaching responses."""
    
    def __init__(self):
        """Initialize the DSPy interface."""
        try:
            # Initialize DSPy with OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment")
            
            # Configure DSPy with GPT-4
            self.lm = dspy.LM(model="gpt-4", api_key=api_key)
            dspy.configure(lm=self.lm)
            logger.info("Initialized DSPy with GPT-4")
            
            # Initialize signatures
            self._create_signatures()
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy interface: {e}")
            raise

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
            raise

    def generate_response(self,
                       prompt: str,
                       context: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a teaching response using DSPy.
        
        Args:
            prompt: The teaching prompt/question
            context: Optional reference content
            metadata: Optional teaching context
            
        Returns:
            Generated teaching response
        """
        try:
            if not HAVE_DSPY:
                raise ImportError("DSPy not available")
            
            # Create predictor with teaching signature
            predictor = dspy.Predict(self.signatures["teaching"])
            
            # Generate response
            result = predictor(
                query=prompt,
                context=context if context else [],
                metadata=metadata if metadata else {}
            )
            
            return result.response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise 