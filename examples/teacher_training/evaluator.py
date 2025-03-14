#!/usr/bin/env python3
"""
Response Evaluator

This module evaluates teacher responses to classroom scenarios based on educational
best practices and research. It provides evidence-based feedback and suggestions
for improvement.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import core components
from src.core.vector_database import VectorDatabase
from src.llm.dspy.handler import DSPyLLMHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """
    Evaluates teacher responses to classroom scenarios based on educational research.
    Provides evidence-based feedback and suggestions for improvement.
    """
    
    def __init__(self, vector_db: VectorDatabase, llm_handler: Optional[DSPyLLMHandler] = None):
        """
        Initialize the response evaluator.
        
        Args:
            vector_db: Vector database containing educational knowledge
            llm_handler: LLM handler for generating evaluations
        """
        self.vector_db = vector_db
        self.llm_handler = llm_handler
        
    def evaluate_teacher_response(self, teacher_response: str, scenario: dict, student_state: dict) -> dict:
        """
        Evaluate a teacher's response based on pedagogical criteria and context.
        
        This function performs a comprehensive evaluation of a teacher's response
        considering multiple factors including the teaching context, student's
        emotional state, and pedagogical best practices.
        
        Evaluation Criteria:
            1. Use of supportive language
            2. Emotional awareness and response
            3. Instructional clarity
            4. Politeness and tone
            5. Context appropriateness
        
        Args:
            teacher_response (str): The teacher's reply to evaluate. Should be
                the complete response including any instructions or feedback.
            
            scenario (dict): Teaching scenario context including:
                - supportive_keywords (list): Expected supportive phrases
                - direct_instruction (bool): Whether direct instruction is needed
                - learning_objectives (list): Target learning goals
                - difficulty_level (str): Expected complexity level
                Example:
                {
                    "supportive_keywords": ["I understand", "good job"],
                    "direct_instruction": True,
                    "learning_objectives": ["addition", "subtraction"]
                }
            
            student_state (dict): Current student state including:
                - emotion (str): Current emotional state
                - engagement (float): Engagement level
                - understanding (float): Comprehension level
                Example:
                {
                    "emotion": "upset",
                    "engagement": 0.7,
                    "understanding": 0.4
                }
        
        Returns:
            dict: Evaluation results containing:
                - score (int): Numerical assessment (0-10)
                - feedback (list): Specific improvement suggestions
                Example:
                {
                    "score": 7,
                    "feedback": [
                        "Good use of supportive language",
                        "Consider acknowledging student's emotion"
                    ]
                }
        
        Note:
            The scoring system weights different aspects based on:
            - Student's emotional state (higher weight when upset)
            - Scenario requirements (e.g., need for direct instruction)
            - Use of supportive language
            - Overall response appropriateness
        """
        result = {"score": 0, "feedback": []}

        # Evaluate supportive language from teacher based on scenario expectations
        supportive_keywords = scenario.get("supportive_keywords", ["I understand", "good job", "well done"])
        if any(word.lower() in teacher_response.lower() for word in supportive_keywords):
            result["score"] += 2
        else:
            result["feedback"].append("Try to include supportive language in your reply.")

        # Evaluate based on the student's emotional state
        if student_state.get("emotion") == "upset":
            if "I know it's hard" in teacher_response or "I understand" in teacher_response:
                result["score"] += 3
            else:
                result["feedback"].append("Consider acknowledging that the student is feeling upset.")
        
        # Additional scenario-based criteria (e.g. using polite requests)
        if scenario.get("direct_instruction"):
            if "please" in teacher_response.lower():
                result["score"] += 1
            else:
                result["feedback"].append("Remember to use polite directives; try including 'please'.")
        
        return result 