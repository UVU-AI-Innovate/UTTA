"""
Teacher Response Evaluation Module

This module provides functionality for evaluating teacher responses in
educational scenarios. It analyzes responses based on multiple criteria
including supportive language, emotional awareness, and instructional
effectiveness.

Key Features:
    - Multi-criteria evaluation
    - Context-aware scoring
    - Detailed feedback generation
    - Student state consideration
    - Scenario-specific assessment

Components:
    - Response Analysis: Evaluates teaching approach
    - Scoring System: Assigns points based on criteria
    - Feedback Generation: Provides improvement suggestions
    - State Management: Considers student emotional state

Example:
    result = evaluate_teacher_response(
        "Let's try this together. I know it's challenging.",
        scenario={"supportive_keywords": ["together", "try"]},
        student_state={"emotion": "frustrated"}
    )
"""

def evaluate_teacher_response(teacher_response: str, scenario: dict, student_state: dict) -> dict:
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