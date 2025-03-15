"""DSPy LLM interface for teaching responses."""

import os
import logging
import importlib.util
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
    # Try to find the package another way
    try:
        spec = importlib.util.find_spec("dspy")
        if spec is not None:
            dspy = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dspy)
            HAVE_DSPY = True
            logger.info("Successfully imported dspy via importlib")
    except Exception as e2:
        logger.error(f"Failed to import DSPy via importlib: {e2}")
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
                
            def configure(self, **kwargs):
                pass
                
            def LM(self, **kwargs):
                return None
        
        dspy = FakeDSPy()
        logger.warning("Using fake DSPy implementation for graceful degradation")

class EnhancedDSPyLLMInterface:
    """Enhanced DSPy interface for teaching responses."""
    
    def __init__(self):
        """Initialize the DSPy interface."""
        self.initialized = False
        try:
            # Initialize DSPy with OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not found in environment")
            
            if HAVE_DSPY:
                # Configure DSPy with GPT-4
                self.lm = dspy.LM(model="gpt-4", api_key=api_key)
                dspy.configure(lm=self.lm)
                logger.info("Initialized DSPy with GPT-4")
                
                # Initialize signatures
                self._create_signatures()
                self.initialized = True
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
        """Generate a teaching response using DSPy.
        
        Args:
            prompt: The teaching prompt/question
            context: Optional reference content
            metadata: Optional teaching context
            
        Returns:
            Generated teaching response
        """
        try:
            if not self.initialized:
                return "I'm sorry, but the teaching assistant is not fully initialized. Please check the logs for more information."
            
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
            return f"I'm sorry, but I encountered an error: {str(e)}" 