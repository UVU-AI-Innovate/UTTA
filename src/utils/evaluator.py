def evaluate_teacher_response(teacher_response: str, scenario: dict, student_state: dict) -> dict:
    """
    Evaluate a teacher's response based on the current scenario and student's state.

    Args:
        teacher_response (str): The teacher's reply.
        scenario (dict): Contains scenario-specific evaluation criteria. For example:
                         { "supportive_keywords": ["I understand", "good job"],
                           "direct_instruction": True }
        student_state (dict): The current state of the student, e.g.:
                              { "emotion": "upset" }

    Returns:
        dict: A dictionary containing a "score" and a list of "feedback" messages.
    """
    result = {"score": 0, "feedback": []}

    # Evaluate supportive language from teacher based on scenario expectations.
    supportive_keywords = scenario.get("supportive_keywords", ["I understand", "good job", "well done"])
    if any(word.lower() in teacher_response.lower() for word in supportive_keywords):
        result["score"] += 2
    else:
        result["feedback"].append("Try to include supportive language in your reply.")

    # Evaluate based on the student's emotional state.
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