"""Manages the teacher training simulator session."""
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime

@dataclass
class SimulatorSession:
    start_time: datetime = field(default_factory=datetime.now)
    history: List[Dict] = field(default_factory=list)
    settings: Dict = field(default_factory=lambda: {
        "show_tips": True,
        "auto_clear": True,
        "show_examples": True,
        "debug_mode": False
    })
    
    def add_interaction(self, teacher_response: str, evaluation: Dict):
        """Add an interaction to the session history."""
        self.history.append({
            "timestamp": datetime.now(),
            "teacher_response": teacher_response,
            "score": evaluation["score"],
            "student_reaction": evaluation["student_reaction"],
            "feedback": evaluation["feedback"],
            "suggestions": evaluation["suggestions"]
        })
    
    def get_average_score(self) -> float:
        """Calculate average score for the session."""
        if not self.history:
            return 0.0
        return sum(h["score"] for h in self.history) / len(self.history)
    
    def get_best_response(self) -> Dict:
        """Get the best response from the session."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x["score"])
    
    def get_areas_for_improvement(self) -> List[str]:
        """Analyze common suggestions to identify areas for improvement."""
        if not self.history:
            return []
        all_suggestions = [s for h in self.history for s in h["suggestions"]]
        # Count occurrences of each suggestion type
        from collections import Counter
        return [item for item, count in Counter(all_suggestions).most_common(3)] 