from typing import List, Dict, Any
import json
from datetime import datetime
from pathlib import Path

class HumanEvaluation:
    """Class for managing human evaluation of chatbot responses."""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_evaluation_form(self, 
                             conversation: List[Dict[str, str]], 
                             criteria: List[str] = None) -> Dict[str, Any]:
        """
        Create an evaluation form for human annotators.
        
        Args:
            conversation: List of conversation turns
            criteria: List of evaluation criteria. If None, uses default criteria
            
        Returns:
            Dictionary containing the evaluation form structure
        """
        if criteria is None:
            criteria = [
                "Response Relevance (1-5)",
                "Response Accuracy (1-5)",
                "Language Quality (1-5)",
                "Task Completion (1-5)",
                "Overall Quality (1-5)"
            ]
            
        form = {
            "conversation_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "conversation": conversation,
            "criteria": criteria,
            "evaluations": []
        }
        
        return form
    
    def save_evaluation(self, 
                       evaluator_id: str,
                       form_data: Dict[str, Any],
                       scores: Dict[str, int],
                       comments: str = "") -> None:
        """
        Save a completed evaluation.
        
        Args:
            evaluator_id: Unique identifier for the evaluator
            form_data: The evaluation form being completed
            scores: Dictionary mapping criteria to scores
            comments: Optional evaluator comments
        """
        evaluation = {
            "evaluator_id": evaluator_id,
            "timestamp": datetime.now().isoformat(),
            "scores": scores,
            "comments": comments
        }
        
        form_data["evaluations"].append(evaluation)
        
        # Save to file
        output_file = self.output_dir / f"evaluation_{form_data['conversation_id']}.json"
        with open(output_file, 'w') as f:
            json.dump(form_data, f, indent=2)
            
    def aggregate_evaluations(self, conversation_id: str) -> Dict[str, float]:
        """
        Aggregate scores from multiple evaluators for a conversation.
        
        Args:
            conversation_id: ID of the conversation to aggregate
            
        Returns:
            Dictionary containing mean scores for each criterion
        """
        eval_file = self.output_dir / f"evaluation_{conversation_id}.json"
        
        if not eval_file.exists():
            raise FileNotFoundError(f"No evaluation file found for conversation {conversation_id}")
            
        with open(eval_file, 'r') as f:
            form_data = json.load(f)
            
        # Calculate mean scores for each criterion
        all_scores = {}
        for criterion in form_data["criteria"]:
            scores = [eval_["scores"][criterion] for eval_ in form_data["evaluations"]]
            all_scores[criterion] = {
                "mean": sum(scores) / len(scores),
                "std": (sum((x - (sum(scores) / len(scores))) ** 2 for x in scores) / len(scores)) ** 0.5
            }
            
        return all_scores 