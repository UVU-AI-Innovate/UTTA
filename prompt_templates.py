"""
Prompt templates for different teaching scenarios and interactions.
"""

class PromptTemplates:
    """Manages prompt templates for different teaching scenarios."""
    
    def __init__(self):
        """Initialize prompt templates."""
        self.templates = {
            "analysis": """
                Analyze this teacher's response to a second-grade student.
                Consider:
                1. Age-appropriateness
                2. Clarity of instruction
                3. Emotional support
                4. Teaching effectiveness
                
                Student Context:
                {student_context}
                
                Teacher's Response:
                {teacher_input}
                
                Provide analysis in the following format:
                - Effectiveness Score (0-1)
                - Identified Strengths
                - Areas for Improvement
                - Suggested Alternative Approaches
            """,
            
            "student_reaction": """
                Generate an age-appropriate student reaction.
                Consider:
                1. Student's current emotional state
                2. Learning style
                3. Previous interactions
                4. Teaching effectiveness score: {effectiveness}
                
                Student Context:
                {student_context}
                
                Generate a natural, second-grade level response that includes:
                1. Verbal response
                2. Behavioral cues (in *asterisks*)
            """,
            
            "scenario_creation": """
                Create a teaching scenario for a second-grade classroom.
                Include:
                1. Subject area: {subject}
                2. Specific learning challenge
                3. Student characteristics
                4. Environmental factors
                5. Learning objectives
                
                Parameters:
                {parameters}
                
                Format the scenario with:
                - Context description
                - Student background
                - Specific challenge
                - Teaching goals
            """
        }
    
    def get(self, template_name: str) -> str:
        """
        Get a specific prompt template.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            The prompt template string
        """
        return self.templates.get(
            template_name,
            "Template not found. Please check the template name."
        ) 