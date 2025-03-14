"""Automated metrics for evaluating teaching responses."""

import logging
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import textstat
from pathlib import Path

class AutomatedMetricsEvaluator:
    """Evaluates teaching responses using automated metrics."""
    
    def __init__(self):
        """Initialize the evaluator."""
        # Load models
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_sm')
        
        # Define pedagogical markers
        self.markers = {
            'engagement': [
                'let\'s', 'try', 'imagine', 'think about',
                'what if', 'how would you', 'can you'
            ],
            'scaffolding': [
                'first', 'next', 'then', 'finally',
                'step by step', 'break this down'
            ],
            'examples': [
                'for example', 'like', 'such as',
                'similar to', 'imagine if'
            ],
            'checking': [
                'do you understand', 'does that make sense',
                'what do you think', 'can you explain'
            ]
        }
        
        logging.info("Initialized AutomatedMetricsEvaluator")
    
    def evaluate_response(self,
                        teaching_response: str,
                        student_profile: Dict[str, str],
                        context: str) -> Dict[str, Any]:
        """Evaluate a teaching response using multiple metrics.
        
        Args:
            teaching_response: The response to evaluate
            student_profile: Student characteristics
            context: Teaching context
            
        Returns:
            Dictionary containing evaluation metrics
        """
        try:
            # Calculate individual metrics
            metrics = {
                'clarity_score': self._evaluate_clarity(teaching_response),
                'engagement_score': self._evaluate_engagement(teaching_response),
                'context_score': self._evaluate_context_alignment(
                    teaching_response,
                    context
                ),
                'age_appropriate_score': self._evaluate_age_appropriateness(
                    teaching_response,
                    student_profile
                ),
                'pedagogical_score': self._evaluate_pedagogical_elements(
                    teaching_response
                )
            }
            
            # Calculate overall score
            metrics['overall_score'] = np.mean([
                metrics['clarity_score'],
                metrics['engagement_score'],
                metrics['context_score'],
                metrics['age_appropriate_score'],
                metrics['pedagogical_score']
            ])
            
            # Generate feedback
            metrics['feedback'] = self._generate_feedback(metrics)
            
            logging.info(f"Evaluated response with overall score: {metrics['overall_score']:.2f}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error evaluating response: {e}")
            return {
                'error': str(e),
                'overall_score': 0.0
            }
    
    def _evaluate_clarity(self, response: str) -> float:
        """Evaluate the clarity of the response."""
        try:
            # Calculate readability scores
            flesch_score = textstat.flesch_reading_ease(response) / 100
            
            # Analyze sentence structure
            doc = self.nlp(response)
            sentences = list(doc.sents)
            
            # Check average sentence length (aim for 15-25 words)
            avg_sentence_length = np.mean([len(sent) for sent in sentences])
            length_score = 1.0 - min(abs(20 - avg_sentence_length) / 20, 1.0)
            
            # Check for complex words
            complex_words = len([
                word for word in doc
                if len(word.text) > 3 and word.is_alpha
                and textstat.syllable_count(word.text) > 2
            ])
            complexity_score = 1.0 - min(complex_words / len(doc), 1.0)
            
            # Combine scores
            clarity_score = np.mean([
                flesch_score,
                length_score,
                complexity_score
            ])
            
            return min(max(clarity_score, 0.0), 1.0)
            
        except Exception as e:
            logging.error(f"Error evaluating clarity: {e}")
            return 0.0
    
    def _evaluate_engagement(self, response: str) -> float:
        """Evaluate the engagement level of the response."""
        try:
            response_lower = response.lower()
            
            # Count engagement markers
            engagement_count = sum(
                1 for marker in self.markers['engagement']
                if marker in response_lower
            )
            
            # Count questions
            doc = self.nlp(response)
            questions = len([
                sent for sent in doc.sents
                if sent.text.strip().endswith('?')
            ])
            
            # Calculate engagement score
            marker_score = min(engagement_count / 3, 1.0)
            question_score = min(questions / 2, 1.0)
            
            return np.mean([marker_score, question_score])
            
        except Exception as e:
            logging.error(f"Error evaluating engagement: {e}")
            return 0.0
    
    def _evaluate_context_alignment(self,
                                 response: str,
                                 context: str) -> float:
        """Evaluate how well the response aligns with the context."""
        try:
            # Generate embeddings
            response_embedding = self.encoder.encode(response)
            context_embedding = self.encoder.encode(context)
            
            # Calculate cosine similarity
            similarity = np.dot(response_embedding, context_embedding) / (
                np.linalg.norm(response_embedding) *
                np.linalg.norm(context_embedding)
            )
            
            return float(similarity)
            
        except Exception as e:
            logging.error(f"Error evaluating context alignment: {e}")
            return 0.0
    
    def _evaluate_age_appropriateness(self,
                                   response: str,
                                   student_profile: Dict[str, str]) -> float:
        """Evaluate if the response is age-appropriate."""
        try:
            # Get grade level
            grade_level = student_profile.get('grade_level', '5th')
            target_grade = int(''.join(filter(str.isdigit, grade_level)))
            
            # Calculate reading level
            reading_level = textstat.text_standard(response, float_output=True)
            
            # Score based on how close the reading level is to the target grade
            score = 1.0 - min(abs(reading_level - target_grade) / target_grade, 1.0)
            
            return score
            
        except Exception as e:
            logging.error(f"Error evaluating age appropriateness: {e}")
            return 0.0
    
    def _evaluate_pedagogical_elements(self, response: str) -> float:
        """Evaluate the presence of pedagogical elements."""
        try:
            response_lower = response.lower()
            scores = []
            
            # Check for each type of pedagogical marker
            for marker_type, markers in self.markers.items():
                count = sum(1 for marker in markers if marker in response_lower)
                scores.append(min(count / 2, 1.0))
            
            return np.mean(scores)
            
        except Exception as e:
            logging.error(f"Error evaluating pedagogical elements: {e}")
            return 0.0
    
    def _generate_feedback(self, metrics: Dict[str, float]) -> List[str]:
        """Generate feedback based on metrics."""
        feedback = []
        
        # Clarity feedback
        if metrics['clarity_score'] < 0.6:
            feedback.append(
                "Consider simplifying language and shortening sentences"
            )
        
        # Engagement feedback
        if metrics['engagement_score'] < 0.6:
            feedback.append(
                "Add more interactive elements like questions or thought experiments"
            )
        
        # Context feedback
        if metrics['context_score'] < 0.6:
            feedback.append(
                "Align the response more closely with the teaching context"
            )
        
        # Age appropriateness feedback
        if metrics['age_appropriate_score'] < 0.6:
            feedback.append(
                "Adjust the language level to better match the student's grade"
            )
        
        # Pedagogical elements feedback
        if metrics['pedagogical_score'] < 0.6:
            feedback.append(
                "Include more scaffolding, examples, or check for understanding"
            )
        
        return feedback 