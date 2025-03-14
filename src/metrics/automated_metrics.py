"""Automated metrics evaluator for teaching responses."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import textstat
    import spacy
    HAVE_DEPS = True
except ImportError:
    logger.warning("Some dependencies missing. Metrics will be limited.")
    HAVE_DEPS = False

@dataclass
class TeachingMetrics:
    """Metrics for evaluating teaching responses."""
    clarity: float = 0.0
    engagement: float = 0.0
    pedagogical_approach: float = 0.0
    emotional_support: float = 0.0
    content_accuracy: float = 0.0
    age_appropriateness: float = 0.0
    feedback: List[str] = None

    def __post_init__(self):
        if self.feedback is None:
            self.feedback = []

class AutomatedMetricsEvaluator:
    """Evaluates teaching responses using automated metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        try:
            if HAVE_DEPS:
                self.nlp = spacy.load("en_core_web_sm")
            
            # Pedagogical markers
            self.engagement_markers = {
                "question_words": ["what", "why", "how", "can you", "do you"],
                "interactive_phrases": ["let's", "try this", "imagine", "think about"]
            }
            
            self.scaffolding_markers = {
                "sequence": ["first", "then", "next", "finally"],
                "explanation": ["because", "therefore", "this means", "in other words"],
                "examples": ["for example", "such as", "like", "instance"]
            }
            
            self.emotional_support_markers = {
                "encouragement": ["great", "excellent", "well done", "good job"],
                "reassurance": ["don't worry", "it's okay", "take your time"],
                "empathy": ["understand", "see why", "makes sense"]
            }
            
            logger.info("Initialized AutomatedMetricsEvaluator")
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluator: {e}")
            raise

    def evaluate_response(self,
                       response: str,
                       prompt: str,
                       grade_level: int,
                       context: Optional[List[str]] = None) -> TeachingMetrics:
        """Evaluate a teaching response.
        
        Args:
            response: The teaching response to evaluate
            prompt: The original prompt/question
            grade_level: Student's grade level
            context: Optional reference content
            
        Returns:
            TeachingMetrics object with scores and feedback
        """
        try:
            metrics = TeachingMetrics()
            
            # Basic text analysis
            if HAVE_DEPS:
                doc = self.nlp(response)
                
                # Evaluate clarity
                metrics.clarity = self._evaluate_clarity(doc)
                
                # Evaluate engagement
                metrics.engagement = self._evaluate_engagement(doc)
                
                # Evaluate pedagogical approach
                metrics.pedagogical_approach = self._evaluate_pedagogical_approach(doc)
                
                # Evaluate emotional support
                metrics.emotional_support = self._evaluate_emotional_support(doc)
                
                # Evaluate age appropriateness
                metrics.age_appropriateness = self._evaluate_age_appropriateness(response, grade_level)
                
                # Evaluate content accuracy if context provided
                if context:
                    metrics.content_accuracy = self._evaluate_content_accuracy(response, context)
                
                # Generate feedback
                metrics.feedback = self._generate_feedback(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return TeachingMetrics()

    def _evaluate_clarity(self, doc) -> float:
        """Evaluate clarity of the response."""
        try:
            # Calculate readability
            text = doc.text
            readability_score = textstat.flesch_reading_ease(text) / 100
            
            # Analyze sentence structure
            sentence_lengths = [len(sent) for sent in doc.sents]
            avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            sentence_complexity = min(1.0, 20 / avg_sentence_length) if avg_sentence_length > 0 else 0
            
            # Combine metrics
            clarity_score = (readability_score + sentence_complexity) / 2
            return max(0.0, min(1.0, clarity_score))
            
        except Exception as e:
            logger.warning(f"Error evaluating clarity: {e}")
            return 0.5

    def _evaluate_engagement(self, doc) -> float:
        """Evaluate engagement level of the response."""
        try:
            text_lower = doc.text.lower()
            
            # Count questions
            question_count = sum(1 for sent in doc.sents if sent.text.strip().endswith("?"))
            
            # Check for engagement markers
            marker_counts = {
                "questions": sum(1 for word in self.engagement_markers["question_words"] if word in text_lower),
                "interactive": sum(1 for phrase in self.engagement_markers["interactive_phrases"] if phrase in text_lower)
            }
            
            # Calculate engagement score
            total_markers = sum(marker_counts.values()) + question_count
            engagement_score = min(1.0, total_markers / 5)  # Normalize to [0,1]
            
            return engagement_score
            
        except Exception as e:
            logger.warning(f"Error evaluating engagement: {e}")
            return 0.5

    def _evaluate_pedagogical_approach(self, doc) -> float:
        """Evaluate pedagogical approach of the response."""
        try:
            text_lower = doc.text.lower()
            
            # Count scaffolding elements
            scaffold_counts = {
                category: sum(1 for marker in markers if marker in text_lower)
                for category, markers in self.scaffolding_markers.items()
            }
            
            # Calculate scaffolding score
            total_scaffolds = sum(scaffold_counts.values())
            scaffold_score = min(1.0, total_scaffolds / 6)  # Normalize to [0,1]
            
            return scaffold_score
            
        except Exception as e:
            logger.warning(f"Error evaluating pedagogical approach: {e}")
            return 0.5

    def _evaluate_emotional_support(self, doc) -> float:
        """Evaluate emotional support in the response."""
        try:
            text_lower = doc.text.lower()
            
            # Count support markers
            support_counts = {
                category: sum(1 for phrase in phrases if phrase in text_lower)
                for category, phrases in self.emotional_support_markers.items()
            }
            
            # Calculate support score
            total_support = sum(support_counts.values())
            support_score = min(1.0, total_support / 4)  # Normalize to [0,1]
            
            return support_score
            
        except Exception as e:
            logger.warning(f"Error evaluating emotional support: {e}")
            return 0.5

    def _evaluate_content_accuracy(self, response: str, context: List[str]) -> float:
        """Evaluate content accuracy against provided context."""
        try:
            if not context:
                return 0.5
                
            # Simple word overlap for now
            response_words = set(response.lower().split())
            context_words = set(" ".join(context).lower().split())
            
            overlap = len(response_words & context_words)
            accuracy_score = overlap / len(context_words) if context_words else 0.5
            
            return max(0.0, min(1.0, accuracy_score))
            
        except Exception as e:
            logger.warning(f"Error evaluating content accuracy: {e}")
            return 0.5

    def _evaluate_age_appropriateness(self, text: str, grade_level: int) -> float:
        """Evaluate age appropriateness of the response."""
        try:
            # Calculate reading level
            reading_level = textstat.text_standard(text, float_output=True)
            
            # Compare with target grade level
            level_difference = abs(reading_level - grade_level)
            appropriateness_score = max(0.0, 1.0 - (level_difference / 5))
            
            return appropriateness_score
            
        except Exception as e:
            logger.warning(f"Error evaluating age appropriateness: {e}")
            return 0.5

    def _generate_feedback(self, metrics: TeachingMetrics) -> List[str]:
        """Generate feedback based on metrics."""
        feedback = []
        
        # Clarity feedback
        if metrics.clarity < 0.5:
            feedback.append("Consider simplifying the language and sentence structure.")
        elif metrics.clarity > 0.8:
            feedback.append("Good clarity and readability.")
            
        # Engagement feedback
        if metrics.engagement < 0.5:
            feedback.append("Try adding more interactive elements and questions.")
        elif metrics.engagement > 0.8:
            feedback.append("Strong engagement with good use of questions and interactive elements.")
            
        # Pedagogical approach feedback
        if metrics.pedagogical_approach < 0.5:
            feedback.append("Include more scaffolding and structured explanation.")
        elif metrics.pedagogical_approach > 0.8:
            feedback.append("Excellent use of pedagogical techniques.")
            
        # Emotional support feedback
        if metrics.emotional_support < 0.5:
            feedback.append("Consider adding more encouraging and supportive language.")
        elif metrics.emotional_support > 0.8:
            feedback.append("Great emotional support and encouragement.")
            
        return feedback 