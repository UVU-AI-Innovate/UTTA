"""DSPy-based LLM interface for generating teaching responses."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import dependencies with error handling
try:
    import dspy
    from dotenv import load_dotenv
    HAVE_DSPY = True
except ImportError as e:
    logger.error(f"Failed to import DSPy: {e}")
    logger.error("Please install DSPy: pip install dspy-ai")
    HAVE_DSPY = False

# Load environment variables
load_dotenv()

class TeachingSignature(dspy.Signature):
    """Base signature for teaching-related tasks."""
    context: List[str] = dspy.InputField(default_factory=list)
    metadata: Dict[str, Any] = dspy.InputField(default_factory=dict)
    response: str = dspy.OutputField()

class GenerateSignature(TeachingSignature):
    """Signature for generating teaching responses."""
    prompt: str = dspy.InputField()

class ExplainSignature(TeachingSignature):
    """Signature for generating explanations."""
    topic: str = dspy.InputField()
    student_level: str = dspy.InputField()
    explanation: str = dspy.OutputField()
    examples: List[str] = dspy.OutputField(default_factory=list)
    questions: List[str] = dspy.OutputField(default_factory=list)

class ScaffoldSignature(TeachingSignature):
    """Signature for scaffolding responses."""
    concept: str = dspy.InputField()
    prerequisites: List[str] = dspy.OutputField(default_factory=list)
    steps: List[str] = dspy.OutputField(default_factory=list)
    practice: List[str] = dspy.OutputField(default_factory=list)

class EnhancedDSPyLLMInterface:
    """Enhanced DSPy-based LLM interface with caching and error handling."""
    
    def __init__(self, 
                model_name: str = "gpt-3.5-turbo",
                cache_dir: Optional[str] = None,
                temperature: float = 0.7):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the LLM model to use
            cache_dir: Directory for response caching
            temperature: Temperature for response generation
        """
        if not HAVE_DSPY:
            logger.error("DSPy not available. Interface will be limited.")
            return
            
        try:
            # Set up caching
            self.cache_dir = Path(cache_dir or "data/cache")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "responses.json"
            self.response_cache = self._load_cache()
            
            # Initialize DSPy
            dspy.settings.configure(lm=model_name, temperature=temperature)
            
            # Initialize modules
            self.generator = dspy.ChainOfThought(GenerateSignature)
            self.explainer = dspy.ChainOfThought(ExplainSignature)
            self.scaffolder = dspy.ChainOfThought(ScaffoldSignature)
            
            logger.info("Initialized DSPy LLM interface")
            
        except Exception as e:
            logger.error(f"Failed to initialize DSPy interface: {e}")
            raise
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load response cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save response cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _create_cache_key(self,
                       prompt: str,
                       context: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a unique cache key."""
        import hashlib
        key_data = {
            "prompt": prompt,
            "context": context or [],
            "metadata": metadata or {}
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def generate_response(self,
                       prompt: str,
                       context: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a teaching response.
        
        Args:
            prompt: The input prompt/question
            context: Optional list of context strings
            metadata: Optional metadata (grade level, subject, etc.)
            
        Returns:
            Generated response
        """
        if not HAVE_DSPY:
            return "DSPy not available. Please install required packages."
            
        try:
            # Check cache
            cache_key = self._create_cache_key(prompt, context, metadata)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]
            
            # Generate response
            result = self.generator(
                prompt=prompt,
                context=context or [],
                metadata=metadata or {},
                response=None
            )
            
            # Cache response
            self.response_cache[cache_key] = result.response
            self._save_cache()
            
            return result.response
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "I apologize, but I encountered an error generating a response. Please try again."
    
    def generate_explanation(self,
                         topic: str,
                         student_level: str,
                         context: Optional[List[str]] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a detailed explanation.
        
        Args:
            topic: The topic to explain
            student_level: Student's grade/comprehension level
            context: Optional list of context strings
            metadata: Optional metadata
            
        Returns:
            Dictionary containing explanation, examples, and practice questions
        """
        if not HAVE_DSPY:
            return {
                "explanation": "DSPy not available. Please install required packages.",
                "examples": [],
                "questions": []
            }
            
        try:
            result = self.explainer(
                topic=topic,
                student_level=student_level,
                context=context or [],
                metadata=metadata or {},
                explanation=None,
                examples=None,
                questions=None
            )
            
            return {
                "explanation": result.explanation,
                "examples": result.examples,
                "questions": result.questions
            }
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return {
                "explanation": "Error generating explanation",
                "examples": [],
                "questions": []
            }
    
    def generate_scaffold(self,
                       concept: str,
                       context: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate scaffolded learning steps.
        
        Args:
            concept: The concept to scaffold
            context: Optional list of context strings
            metadata: Optional metadata
            
        Returns:
            Dictionary containing prerequisites, steps, and practice items
        """
        if not HAVE_DSPY:
            return {
                "prerequisites": [],
                "steps": [],
                "practice": []
            }
            
        try:
            result = self.scaffolder(
                concept=concept,
                context=context or [],
                metadata=metadata or {},
                prerequisites=None,
                steps=None,
                practice=None
            )
            
            return {
                "prerequisites": result.prerequisites,
                "steps": result.steps,
                "practice": result.practice
            }
            
        except Exception as e:
            logger.error(f"Failed to generate scaffold: {e}")
            return {
                "prerequisites": [],
                "steps": [],
                "practice": []
            } 