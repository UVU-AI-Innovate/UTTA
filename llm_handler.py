"""
Pedagogical Language Processing Module

This module handles all natural language processing tasks related to teaching scenarios,
including response analysis, scenario creation, and student reaction generation.
It provides a structured interface to the underlying LLM while maintaining
educational context and pedagogical principles.

Key Components:
    - PedagogicalLanguageProcessor: Main class for processing educational language
    - TeachingContext: Data class for maintaining teaching context
    - LLM Integration: Handles communication with language models
    - Template Management: Manages educational prompt templates

Features:
    - Teaching response analysis
    - Student reaction generation
    - Scenario creation
    - Real-time feedback
    - Context-aware processing

Dependencies:
    - requests: For API communication
    - json: For data serialization
    - llm_interface: For language model interaction
    - prompt_templates: For educational prompts

Example:
    processor = PedagogicalLanguageProcessor()
    analysis = processor.analyze_teaching_response("Let's use blocks to count", context)
    reaction = processor.generate_student_reaction(context, effectiveness=0.8)
"""

import requests
import json
import time
import os
import signal
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from llm_interface import LLMInterface
from prompt_templates import PromptTemplates

@dataclass
class TeachingContext:
    """
    Data class for maintaining teaching context and state.
    
    This class encapsulates all relevant information about the current
    teaching situation, student characteristics, and interaction history.
    It serves as a central point for context management in the teaching
    simulation.
    
    Attributes:
        subject (str): The subject being taught (e.g., "math", "reading")
        grade_level (str): The grade level of the student (e.g., "2nd", "3rd")
        learning_objectives (List[str]): Specific goals for the lesson
        student_characteristics (Dict[str, Any]): Student traits including:
            - learning_style: Preferred learning method
            - attention_span: Attention capacity
            - strengths: Areas of proficiency
            - challenges: Areas needing support
        previous_interactions (Optional[List[Dict[str, str]]]): History of
            recent teacher-student interactions
    
    Example:
        context = TeachingContext(
            subject="mathematics",
            grade_level="2nd",
            learning_objectives=["Add two-digit numbers"],
            student_characteristics={
                "learning_style": "visual",
                "attention_span": "moderate"
            }
        )
    """
    subject: str
    grade_level: str
    learning_objectives: List[str]
    student_characteristics: Dict[str, Any]
    previous_interactions: Optional[List[Dict[str, str]]] = None

class PedagogicalLanguageProcessor:
    """
    Processes natural language in educational contexts using LLMs.
    
    This class serves as the main interface for all language-related
    operations in the teaching simulation. It maintains educational
    context while processing language, ensuring responses are
    pedagogically appropriate and age-suitable.
    
    Key Features:
        - Teaching response analysis
        - Student reaction generation
        - Scenario creation
        - Real-time feedback
        - Context maintenance
    
    Components:
        - LLM Interface: Handles communication with language models
        - Prompt Templates: Manages educational prompt patterns
        - Context Management: Maintains teaching situation state
        - Response Processing: Analyzes and generates responses
    
    Attributes:
        llm (LLMInterface): Interface to the language model
        templates (PromptTemplates): Collection of educational prompt templates
        
    Example:
        processor = PedagogicalLanguageProcessor()
        analysis = processor.analyze_teaching_response(
            "Let's solve this step by step",
            teaching_context
        )
    """
    
    def __init__(self):
        """Initialize the processor with LLM interface and prompt templates."""
        self.llm = LLMInterface()
        self.templates = PromptTemplates()
        
    def analyze_teaching_response(
        self, 
        teacher_input: str, 
        context: TeachingContext
    ) -> Dict[str, Any]:
        """
        Analyze the effectiveness of a teacher's response.
        
        Args:
            teacher_input: The teacher's response or action
            context: Current teaching context including student info
            
        Returns:
            Dictionary containing analysis results:
            - effectiveness_score: Float between 0 and 1
            - identified_strengths: List of effective elements
            - improvement_areas: List of areas for improvement
            - suggested_strategies: Alternative approaches
            
        Example:
            analysis = processor.analyze_teaching_response(
                "Let's break this problem into smaller steps",
                current_context
            )
        """
        prompt = self.templates.get("analysis")
        return self.llm.generate(
            prompt=prompt,
            context=context,
            input=teacher_input
        )
        
    def generate_student_reaction(
        self, 
        context: TeachingContext, 
        effectiveness: float
    ) -> str:
        """
        Generate an age-appropriate student reaction.
        
        Args:
            context: Current teaching context
            effectiveness: Float between 0 and 1 indicating teaching effectiveness
            
        Returns:
            A string containing the simulated student's reaction
            
        Example:
            reaction = processor.generate_student_reaction(
                context,
                effectiveness=0.85
            )
        """
        prompt = self.templates.get("student_reaction")
        return self.llm.generate(
            prompt=prompt,
            context=context,
            effectiveness=effectiveness
        )
        
    def create_scenario(
        self, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a teaching scenario based on given parameters.
        
        Args:
            parameters: Dictionary containing scenario parameters:
                - subject: Subject area
                - difficulty: Difficulty level
                - student_profile: Student characteristics
                
        Returns:
            Dictionary containing the complete scenario:
            - context: Teaching context
            - student_background: Relevant student information
            - learning_objectives: Specific goals
            - potential_challenges: Anticipated difficulties
            
        Example:
            scenario = processor.create_scenario({
                "subject": "mathematics",
                "difficulty": "intermediate",
                "student_profile": {...}
            })
        """
        prompt = self.templates.get("scenario_creation")
        return self.llm.generate(
            prompt=prompt,
            parameters=parameters
        )

    def ensure_server_running(self):
        """
        Ensure the Ollama server is running and ready to process requests.
        
        This method:
            1. Checks if the server is already running
            2. Attempts to start the server if it's not running
            3. Verifies the server is responsive
            4. Handles startup errors gracefully
        
        Returns:
            bool: True if server is running and responsive, False otherwise
        
        Note:
            - Waits up to 10 seconds for server to start
            - Provides feedback about server status
            - Falls back gracefully if server cannot be started
        """
        if not self.check_server():
            print("\nStarting Ollama server...")
            try:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                for _ in range(5):
                    time.sleep(2)
                    if self.check_server():
                        print("Ollama server started successfully!")
                        return True
                print("Failed to start Ollama server.")
                return False
            except Exception as e:
                print(f"Error starting Ollama server: {e}")
                return False
        return True

    def check_server(self):
        """
        Check if the Ollama server is running and responsive.
        
        Performs a health check by:
            1. Sending a GET request to the server's version endpoint
            2. Verifying the response status
            3. Handling connection timeouts
        
        Returns:
            bool: True if server is responsive, False otherwise
        
        Note:
            Uses a 2-second timeout to avoid hanging on unresponsive server
        """
        try:
            response = requests.get(f"{self.base_url}/version", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def analyze_teaching_response(self, prompt: str) -> dict:
        """Analyze teacher's response using local LLM."""
        if not self.ensure_server_running():
            return self._get_fallback_evaluation()

        try:
            # Create a safe version of the prompt with escaped quotes
            safe_prompt = prompt.replace('"', '\\"')
            
            evaluation_prompt = (
                'Please analyze this teacher\'s response to a second-grade student.\n\n'
                f'The teacher\'s response was: "{safe_prompt}"\n\n'
                'Evaluate the response and provide feedback using this exact JSON structure:\n'
                '{\n'
                '    "score": 0.75,\n'
                '    "strengths": ["Example strength 1", "Example strength 2"],\n'
                '    "suggestions": ["Example suggestion 1", "Example suggestion 2"],\n'
                '    "explanation": "Example explanation"\n'
                '}\n\n'
                'Your evaluation should focus on:\n'
                '1. Clarity and age-appropriateness\n'
                '2. Teaching effectiveness\n'
                '3. Student engagement\n'
                '4. Emotional support\n\n'
                'Return only the JSON object with no additional text or formatting.'
            )

            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": evaluation_prompt,
                    "system": "You are a teaching expert. Return only a valid JSON object with no additional text.",
                    "stream": False
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()["response"].strip()
            
            try:
                # Clean the response
                result = result.replace('```json', '').replace('```', '').strip()
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                result = result.replace('\\"', '"')
                
                # Try to extract JSON object if there's extra text
                import re
                json_matches = re.findall(r'(\{[\s\S]*\})', result)
                if json_matches:
                    result = json_matches[0]
                
                # Parse the JSON
                parsed_result = json.loads(result)
                
                # Create a valid result structure
                valid_result = {
                    "score": 0.5,
                    "strengths": ["Basic teaching approach"],
                    "suggestions": ["Consider adding more specific strategies"],
                    "explanation": "Basic evaluation provided"
                }
                
                # Update with parsed values if they exist and are valid
                if isinstance(parsed_result.get("score"), (int, float)):
                    valid_result["score"] = float(parsed_result["score"])
                
                if isinstance(parsed_result.get("strengths"), list):
                    valid_result["strengths"] = [str(s) for s in parsed_result["strengths"]]
                elif isinstance(parsed_result.get("strengths"), str):
                    valid_result["strengths"] = [parsed_result["strengths"]]
                
                if isinstance(parsed_result.get("suggestions"), list):
                    valid_result["suggestions"] = [str(s) for s in parsed_result["suggestions"]]
                elif isinstance(parsed_result.get("suggestions"), str):
                    valid_result["suggestions"] = [parsed_result["suggestions"]]
                
                if isinstance(parsed_result.get("explanation"), str):
                    valid_result["explanation"] = parsed_result["explanation"]
                
                return valid_result
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw response: {result}")
                return self._structure_free_text_response(result)
                
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return self._get_fallback_evaluation()

    def _structure_free_text_response(self, text: str) -> dict:
        """
        Structure free-form text response into a standardized format.
        
        This utility method processes raw text output from the LLM and
        converts it into a structured dictionary format suitable for
        the application.
        
        Args:
            text (str): Raw text response from the language model
        
        Returns:
            dict: Structured response containing:
                - main_points: Key points extracted from text
                - suggestions: Teaching suggestions
                - analysis: Detailed analysis if present
                - metadata: Additional contextual information
        
        Note:
            Handles various text formats and attempts to extract meaningful
            structure even from inconsistent responses
        """
        try:
            # Remove any markdown formatting
            text = text.replace("```", "").strip()
            
            # Simple heuristic to estimate score based on positive/negative words
            positive_words = ['good', 'great', 'excellent', 'effective', 'well', 'clear', 'appropriate']
            negative_words = ['improve', 'consider', 'should', 'could', 'better', 'unclear', 'missing']
            
            text_lower = text.lower()
            score = 0.5  # Default score
            
            # Adjust score based on positive/negative words
            score += sum(0.1 for word in positive_words if word in text_lower)
            score -= sum(0.1 for word in negative_words if word in text_lower)
            score = max(0.1, min(1.0, score))  # Keep score between 0.1 and 1.0
            
            # Split text into strengths and suggestions
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            strengths = []
            suggestions = []
            
            for line in lines:
                line_lower = line.lower()
                if any(word in line_lower for word in positive_words):
                    strengths.append(line)
                elif any(word in line_lower for word in negative_words):
                    suggestions.append(line)
                elif 'strength' in line_lower or 'positive' in line_lower:
                    strengths.append(line)
                elif 'suggest' in line_lower or 'improve' in line_lower:
                    suggestions.append(line)
            
            return {
                "score": score,
                "strengths": strengths or ["Basic teaching elements present"],
                "suggestions": suggestions or ["Consider adding more specific strategies"],
                "explanation": text
            }
        except Exception as e:
            print(f"Error structuring response: {e}")
            return self._get_fallback_evaluation()

    def _get_fallback_evaluation(self) -> dict:
        """
        Provide fallback evaluation when LLM processing fails.
        
        This method serves as a safety mechanism when:
            - The LLM server is unavailable
            - Response processing fails
            - Network issues occur
        
        Returns:
            dict: A basic evaluation structure containing:
                - score: Default effectiveness score
                - strengths: Basic identified strengths
                - suggestions: Generic improvement suggestions
                - explanation: Simple explanation of the evaluation
        
        Note:
            This ensures the system remains functional even when
            LLM processing is unavailable
        """
        return {
            "score": 0.5,
            "strengths": ["Response provided basic teaching interaction"],
            "suggestions": ["Consider adding more specific teaching strategies"],
            "explanation": "Basic evaluation provided due to processing limitations"
        }

    def generate_student_reaction(
        self,
        context: dict,
        teacher_response: str,
        effectiveness: float
    ) -> str:
        """
        Generate a realistic student reaction to a teacher's response.
        
        This method creates contextually appropriate student reactions
        by considering multiple factors from the teaching context.
        
        Args:
            context (dict): Teaching context including:
                - subject: Subject being taught
                - grade_level: Student's grade level
                - student_profile: Student characteristics
            teacher_response (str): The teacher's action or statement
            effectiveness (float): Effectiveness score (0.0 to 1.0) of
                the teacher's response
        
        Returns:
            str: A natural language student reaction that reflects:
                - Understanding level
                - Emotional state
                - Engagement level
                - Age-appropriate language
                - Learning style preferences
        
        Note:
            The reaction is influenced by:
            - Previous interactions
            - Student's personality
            - Subject matter
            - Time of day
            - Current engagement level
        """
        if not self.ensure_server_running():
            return "*looks uncertain* Okay..."

        prompt = f"""
        As a second-grade student, generate a realistic reaction to the teacher's response.
        
        Context:
        - Subject: {context['subject']}
        - Current Challenge: {context['difficulty']}
        - Student's State: {context['behavioral_context']['type']}
        - Response Effectiveness: {effectiveness}
        
        Teacher's Response: "{teacher_response}"
        
        Generate a natural, age-appropriate reaction that:
        1. Reflects the effectiveness of the teacher's approach
        2. Shows realistic second-grade behavior
        3. Maintains consistency with the student's current state
        4. Includes both verbal response and behavioral cues (in *asterisks*)
        
        Response should be a single line of dialogue/action.
        """

        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": "You are a second-grade student in a classroom situation.",
                    "stream": False
                },
                timeout=10
            )
            response.raise_for_status()
            return response.json()["response"].strip()
            
        except Exception as e:
            print(f"Error generating reaction: {e}")
            return "*looks uncertain* Okay..." 