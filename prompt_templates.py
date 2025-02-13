"""
Educational Prompt Templates Module

This module provides a collection of structured prompts for generating
appropriate responses in various teaching scenarios. The templates are
designed to maintain educational context and pedagogical principles
while interacting with language models.

Key Components:
    - Analysis Templates: For evaluating teaching responses
    - Reaction Templates: For generating student reactions
    - Scenario Templates: For creating teaching situations
    - Template Management: For accessing and formatting prompts

Features:
    - Age-appropriate language
    - Pedagogically sound structure
    - Consistent formatting
    - Context preservation
    - Flexible parameter insertion

Usage:
    templates = PromptTemplates()
    analysis_prompt = templates.get("analysis")
    formatted_prompt = analysis_prompt.format(
        student_context="Visual learner, enjoys math",
        teacher_input="Let's draw this problem"
    )
"""

class PromptTemplates:
    """
    Manages prompt templates for different teaching scenarios.
    
    This class provides a centralized collection of carefully crafted
    prompts designed to generate appropriate responses for various
    teaching situations. Each template is structured to maintain
    educational context and produce consistent, useful outputs.
    
    Templates Categories:
        - Analysis: Evaluates teaching responses
        - Student Reaction: Generates age-appropriate reactions
        - Scenario Creation: Builds teaching scenarios
    
    Features:
        - Structured output formats
        - Context preservation
        - Age-appropriate language
        - Pedagogical considerations
        - Flexible parameter insertion
    
    Example:
        templates = PromptTemplates()
        prompt = templates.get("analysis")
        formatted = prompt.format(
            student_context="Second grade, visual learner",
            teacher_input="Let's solve this step by step"
        )
    """
    
    def __init__(self):
        """
        Initialize the collection of prompt templates.
        
        Sets up a dictionary of carefully structured prompts for:
            - Teaching response analysis
            - Student reaction generation
            - Scenario creation
            
        Each template is designed with:
            - Clear instructions
            - Context placeholders
            - Output structure
            - Educational considerations
        """
        self.templates = {
            # Template for analyzing teacher responses
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
            
            # Template for generating student reactions
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
            
            # Template for creating teaching scenarios
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
        Retrieve a specific prompt template by name.
        
        This method provides access to the pre-defined prompt templates,
        allowing for consistent prompt generation across the application.
        
        Args:
            template_name (str): The identifier for the desired template.
                Valid options:
                - "analysis": For evaluating teaching responses
                - "student_reaction": For generating student reactions
                - "scenario_creation": For creating teaching scenarios
            
        Returns:
            str: The requested prompt template, ready for parameter
                formatting. Returns a default message if template is
                not found.
        
        Example:
            template = templates.get("analysis")
            formatted = template.format(
                student_context="Visual learner",
                teacher_input="Let's draw it out"
            )
        """
        return self.templates.get(
            template_name,
            "Template not found. Please check the template name."
        ) 