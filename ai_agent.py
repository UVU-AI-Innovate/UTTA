"""
Teacher Training Agent Module

This module implements the core AI agent responsible for simulating and managing
teacher training scenarios. It coordinates between the LLM interface, knowledge management,
and pedagogical processing components.

Key Components:
    - TeacherTrainingAgent: Main class that orchestrates the training simulation
    - Knowledge Base Integration: Loads and manages teaching strategies and student behaviors
    - LLM Integration: Handles communication with the language model for response generation
    - Scenario Management: Creates and manages teaching scenarios
    - Evaluation System: Provides feedback on teaching responses

Dependencies:
    - llm_handler: Handles language model interactions
    - knowledge_base: Contains teaching strategies and student characteristics
    - evaluator: Evaluates teacher responses
    - llm_interface: Provides interface to the language model

Example:
    agent = TeacherTrainingAgent()
    scenario = agent.create_teaching_scenario("mathematics", "intermediate", student_profile)
    feedback = agent.evaluate_teaching_response(teacher_input, scenario)
"""

import random
import json
from llm_handler import PedagogicalLanguageProcessor
from knowledge_base import SECOND_GRADE_CHARACTERISTICS, TEACHING_STRATEGIES
from evaluator import evaluate_teacher_response
from llm_interface import LLMInterface
from typing import Dict, Any

class TeacherTrainingAgent:
    """
    Main agent class for managing teacher training simulations.
    
    This class orchestrates the interaction between different components and manages
    the entire teaching simulation lifecycle, including scenario creation, response
    evaluation, and feedback generation.
    
    Key Features:
        - Dynamic scenario generation based on subject and difficulty
        - Realistic student behavior simulation
        - Comprehensive teaching response evaluation
        - Personalized feedback generation
        - Interactive session management
    
    Attributes:
        llm (LLMInterface): Interface to the language model for response generation
        processor (PedagogicalLanguageProcessor): Processor for pedagogical analysis
        learning_style (str): Current student's learning style preference
        current_challenges (list): List of current student's learning challenges
        teacher_profile (dict): Profile containing teacher characteristics
        personality (dict): Simulated student's personality traits and state
    """
    
    def __init__(self):
        """
        Initialize the TeacherTrainingAgent with required components.
        
        Sets up:
            - LLM interface for AI model communication
            - Language processor for response analysis
            - Default learning style and challenges
            - Empty teacher profile
            - Initial student state with randomized personality traits
        """
        self.llm = LLMInterface()
        self.processor = PedagogicalLanguageProcessor()
        self.learning_style = "visual"  # Default learning style
        self.current_challenges = ["staying focused", "asking for help"]  # Default challenges
        self.teacher_profile = {"name": "", "experience_level": "", "grade_level_interest": "", "subject_preferences": [], "teaching_style": "", "areas_for_growth": []}
        self._initialize_student_state()
        
    def create_teaching_scenario(
        self, 
        subject: str, 
        difficulty: str, 
        student_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a teaching scenario based on given parameters.
        
        This method generates a complete teaching scenario by combining the subject matter,
        difficulty level, and student characteristics. It uses the language processor
        to create a contextually appropriate teaching situation.
        
        Args:
            subject (str): The subject area for the scenario (e.g., "mathematics", "reading")
            difficulty (str): The difficulty level (e.g., "beginner", "intermediate", "advanced")
            student_profile (Dict[str, Any]): Dictionary containing:
                - learning_style: Student's preferred learning method
                - grade_level: Current grade level
                - strengths: Areas where the student excels
                - challenges: Areas needing improvement
            
        Returns:
            Dict[str, Any]: A comprehensive scenario including:
                - context: The teaching situation and environment
                - objectives: Specific learning goals
                - student_background: Relevant student information
                - suggested_approaches: Initial teaching strategy suggestions
                - potential_challenges: Anticipated difficulties
        
        Example:
            scenario = agent.create_teaching_scenario(
                "mathematics",
                "intermediate",
                {
                    "learning_style": "visual",
                    "grade_level": "3rd",
                    "strengths": ["pattern recognition"],
                    "challenges": ["word problems"]
                }
            )
        """
        context = {
            "subject": subject,
            "difficulty": difficulty,
            "student_profile": student_profile
        }
        return self.processor.create_scenario(context)
        
    def evaluate_teaching_response(
        self, 
        teacher_input: str, 
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a teacher's response and generate detailed feedback.
        
        This method analyzes the teacher's response in the context of the current
        scenario, considering factors such as pedagogical approach, student needs,
        and teaching effectiveness.
        
        Args:
            teacher_input (str): The teacher's response or action in the scenario
            scenario (Dict[str, Any]): The current teaching scenario containing:
                - context: Teaching situation
                - objectives: Learning goals
                - student_background: Student information
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation results including:
                - effectiveness: Overall effectiveness score (0-1)
                - strengths: List of identified strong points
                - areas_for_improvement: Specific suggestions for improvement
                - alternative_approaches: Other teaching strategies to consider
                - alignment: How well the response aligns with objectives
                - student_impact: Predicted impact on student learning
            
        Example:
            feedback = agent.evaluate_teaching_response(
                "I would use visual aids to demonstrate fraction concepts and 
                 connect them to real-world examples like pizza slices",
                current_scenario
            )
        """
        analysis = self.processor.analyze_teaching_response(
            teacher_input=teacher_input,
            context=scenario
        )
        return self.generate_feedback(analysis)
        
    def generate_feedback(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured feedback based on teaching response analysis.
        
        Args:
            analysis: Analysis results from the language processor
            
        Returns:
            Structured feedback dictionary with specific recommendations
            and improvement suggestions
        """
        # Get relevant teaching strategies
        strategies = TEACHING_STRATEGIES.get(analysis["context"], [])
        
        # Combine analysis with teaching strategies
        return {
            "effectiveness": analysis["effectiveness_score"],
            "strengths": analysis["identified_strengths"],
            "areas_for_improvement": analysis["improvement_areas"],
            "alternative_approaches": strategies
        }

    def _load_teaching_strategies(self):
        """
        Load teaching strategies from knowledge base files.
        
        Attempts to load teaching strategies from the JSON file in the knowledge base.
        If the file is not found, falls back to default strategies.
        
        Returns:
            dict: A dictionary containing teaching strategies organized by:
                - Time of day (morning, afternoon, etc.)
                - Learning styles (visual, auditory, kinesthetic)
                - Subject matter (math, reading, etc.)
                - Student needs (attention, motivation, etc.)
        
        Note:
            Falls back to default strategies if the file is not found
            to ensure the system remains functional.
        """
        try:
            with open('knowledge_base/teaching_strategies.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("! Teaching strategies file not found")
            return self._get_default_teaching_strategies()

    def _load_student_behaviors(self):
        """
        Load student behavior patterns from knowledge base files.
        
        Loads predefined patterns of student behavior and appropriate
        responses for different situations and challenges.
        
        Returns:
            dict: A dictionary containing:
                - Common behaviors and their triggers
                - Appropriate teaching responses
                - Behavioral indicators
                - Intervention strategies
        
        Note:
            Uses default behaviors if the knowledge base file is not found.
        """
        try:
            with open('knowledge_base/student_behaviors.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("! Student behaviors file not found")
            return self._get_default_student_behaviors()

    def _load_subject_content(self):
        """
        Load subject-specific content and teaching strategies.
        
        Retrieves detailed information about different subjects including:
            - Core concepts and progression
            - Common misconceptions
            - Effective teaching approaches
            - Assessment strategies
        
        Returns:
            dict: Subject-specific information organized by:
                - Subject area (math, reading, science, etc.)
                - Grade level appropriateness
                - Teaching methodologies
                - Common challenges and solutions
        """
        try:
            with open('knowledge_base/subject_content.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("! Subject content file not found")
            return self._get_default_subject_content()

    def _get_fallback_knowledge(self):
        """
        Provide basic fallback knowledge if files can't be loaded.
        
        This is a safety mechanism that ensures the system has basic
        functional knowledge even if the knowledge base files are
        inaccessible or corrupted.
        
        Returns:
            dict: A comprehensive set of default knowledge including:
                - Basic teaching strategies
                - Common student behaviors
                - Core subject content
                - Essential pedagogical approaches
        
        Note:
            This knowledge is more generic than the full knowledge base
            but ensures basic functionality of the system.
        """
        return {
            "teaching_strategies": self._get_default_teaching_strategies(),
            "student_behaviors": self._get_default_student_behaviors(),
            "subject_content": self._get_default_subject_content()
        }

    def _get_default_teaching_strategies(self):
        """Default teaching strategies if file loading fails."""
        return {
            "time_strategies": {
                "morning": {
                    "strategies": ["morning routine", "clear expectations", "structured start"],
                    "explanation": "Morning is ideal for setting clear expectations and structured activities when students are fresh."
                },
                "after lunch": {
                    "strategies": ["movement break", "active learning", "energy release"],
                    "explanation": "After lunch, students need movement and engaging activities to maintain focus."
                },
                "late afternoon": {
                    "strategies": ["short tasks", "varied activities", "brain breaks"],
                    "explanation": "Late afternoon requires shorter, varied tasks due to decreased attention spans."
                }
            },
            "learning_styles": {
                "visual": {
                    "strategies": ["look at", "watch", "see", "show", "draw"],
                    "explanation": "Visual learners need to see concepts represented through diagrams, demonstrations, or written instructions."
                },
                "auditory": {
                    "strategies": ["listen", "hear", "tell", "say", "sound"],
                    "explanation": "Auditory learners benefit from verbal instructions, discussions, and sound-based learning."
                },
                "kinesthetic": {
                    "strategies": ["try", "move", "touch", "build", "practice"],
                    "explanation": "Kinesthetic learners need hands-on activities and physical movement to engage with learning."
                }
            }
        }

    def _get_default_student_behaviors(self):
        """Default student behaviors if file loading fails."""
        return {
            "attention": {
                "strategies": ["let's focus", "watch carefully", "look at this"],
                "explanation": "Students with attention challenges need clear, direct prompts to maintain focus."
            },
            "frustration": {
                "strategies": ["you can do this", "let's try together", "take your time"],
                "explanation": "When frustrated, students need encouragement and support to rebuild confidence."
            }
        }

    def _get_default_subject_content(self):
        """Default subject content if file loading fails."""
        return {
            "math": {
                "strategies": ["break down", "step by step", "use manipulatives", "draw it out"],
                "explanation": "Mathematical concepts need concrete representations and step-by-step guidance."
            },
            "reading": {
                "strategies": ["sound it out", "look for clues", "picture walk", "sight words"],
                "explanation": "Reading skills develop through phonics, comprehension strategies, and visual supports."
            }
        }

    def _initialize_student_state(self):
        """
        Initialize student personality and state with randomized characteristics.
        
        Creates a realistic student profile with:
            - Base personality traits (attention span, learning style, etc.)
            - Current emotional and cognitive state
            - Subject preferences and challenges
            - Social and academic characteristics
        
        The state is dynamic and can change during the simulation based on:
            - Teacher interactions
            - Time of day
            - Activity type
            - Previous experiences
        """
        self.personality = {
            "base_traits": {
                "attention_span": random.uniform(0.4, 0.8),
                "learning_style": random.choice(["visual", "auditory", "kinesthetic"]),
                "social_confidence": random.uniform(0.3, 0.8),
                "favorite_subjects": random.sample(["math", "reading", "art", "science"], 2),
                "challenges": random.sample([
                    "reading comprehension",
                    "number sense",
                    "staying focused",
                    "asking for help"
                ], 2)
            },
            "current_state": {
                "engagement": 0.5,
                "understanding": 0.5,
                "mood": 0.5,
                "energy": 0.8
            }
        }

    def setup_teacher_profile(self):
        """Initial setup to gather teacher information."""
        print("\n=== Teacher Profile Setup ===")
        print("\nLet's get to know you better to personalize your training experience.")
        print("(Press Enter to use default values for quick testing)")
        
        name = input("\nWhat's your name? [Test Teacher]: ").strip()
        self.teacher_profile["name"] = name if name else "Test Teacher"
        
        print("\nWhat's your teaching experience level?")
        print("1. Novice (Student or New Teacher)")
        print("2. Intermediate (1-3 years)")
        print("3. Experienced (3+ years)")
        experience_choice = input("Choose (1-3) [1]: ").strip()
        self.teacher_profile["experience_level"] = {
            "1": "novice",
            "2": "intermediate",
            "3": "experienced"
        }.get(experience_choice, "novice")

        print("\nWhich grade levels are you most interested in teaching?")
        print("1. Lower Elementary (K-2)")
        print("2. Upper Elementary (3-5)")
        print("3. Middle School (6-8)")
        grade_choice = input("Choose (1-3) [1]: ").strip()
        self.teacher_profile["grade_level_interest"] = {
            "1": "lower_elementary",
            "2": "upper_elementary",
            "3": "middle_school"
        }.get(grade_choice, "lower_elementary")

        print("\nWhich subjects are you most interested in teaching? (Enter numbers, separated by commas)")
        print("1. Math")
        print("2. Reading/Language Arts")
        print("3. Science")
        print("4. Social Studies")
        subjects = input("Choose (e.g., 1,2) [1,2]: ").strip()
        subject_map = {
            "1": "math",
            "2": "reading",
            "3": "science",
            "4": "social_studies"
        }
        if subjects:
            self.teacher_profile["subject_preferences"] = [
                subject_map[s.strip()] for s in subjects.split(",") if s.strip() in subject_map
            ]
        # Default to math and reading if no input
        if not self.teacher_profile["subject_preferences"]:
            self.teacher_profile["subject_preferences"] = ["math", "reading"]

        print("\nWhat's your preferred teaching style?")
        print("1. Interactive/Hands-on")
        print("2. Traditional/Structured")
        print("3. Mixed Approach")
        style_choice = input("Choose (1-3) [1]: ").strip()
        self.teacher_profile["teaching_style"] = {
            "1": "interactive",
            "2": "traditional",
            "3": "mixed"
        }.get(style_choice, "interactive")

        print("\nWhat areas would you like to improve? (Enter numbers, separated by commas)")
        print("1. Classroom Management")
        print("2. Student Engagement")
        print("3. Differentiated Instruction")
        print("4. Behavior Management")
        print("5. Assessment Strategies")
        areas = input("Choose (e.g., 1,2) [1,2]: ").strip()
        area_map = {
            "1": "classroom_management",
            "2": "student_engagement",
            "3": "differentiated_instruction",
            "4": "behavior_management",
            "5": "assessment_strategies"
        }
        if areas:
            self.teacher_profile["areas_for_growth"] = [
                area_map[a.strip()] for a in areas.split(",") if a.strip() in area_map
            ]
        # Default areas if no input
        if not self.teacher_profile["areas_for_growth"]:
            self.teacher_profile["areas_for_growth"] = ["classroom_management", "student_engagement"]

        print("\nThank you! Your profile has been set up.")
        self._show_profile_summary()

    def _show_profile_summary(self):
        """Display a summary of the teacher's profile."""
        print("\n=== Your Teaching Profile ===")
        print(f"Name: {self.teacher_profile['name']}")
        print(f"Experience Level: {self.teacher_profile['experience_level'].title()}")
        print(f"Grade Level Interest: {self.teacher_profile['grade_level_interest'].replace('_', ' ').title()}")
        print(f"Preferred Subjects: {', '.join(s.title() for s in self.teacher_profile['subject_preferences'])}")
        print(f"Teaching Style: {self.teacher_profile['teaching_style'].title()}")
        print(f"Areas for Growth: {', '.join(a.replace('_', ' ').title() for a in self.teacher_profile['areas_for_growth'])}")

    def generate_scenario(self, subject=None):
        """Generate a detailed teaching scenario with student and classroom context."""
        if not subject:
            subject = random.choice(["math", "reading"])
        
        # Student context
        student_context = {
            "learning_style": self.learning_style,
            "attention_span": self.personality["base_traits"]["attention_span"],
            "social_confidence": self.personality["base_traits"]["social_confidence"],
            "current_challenges": self.current_challenges,
            "seating": random.choice(["front row", "middle row", "back row"]),
            "peer_interactions": random.choice([
                "works well in groups",
                "prefers working alone",
                "easily distracted by peers",
                "shy in group settings"
            ])
        }
        
        # Classroom context
        time_of_day = random.choice(["morning", "after lunch", "late afternoon"])
        class_energy = {
            "morning": "Students are generally alert but may need time to settle",
            "after lunch": "Energy levels are varied, some students restless",
            "late afternoon": "Attention spans are shorter, more frequent breaks needed"
        }
        
        # Get subject-specific details
        subject_skills = SECOND_GRADE_CHARACTERISTICS["cognitive"]["academic_skills"][subject]
        difficulty = random.choice(subject_skills["challenging"])
        
        # Get behavioral context
        behavior_type = random.choice(["attention", "frustration"])
        behavior_info = SECOND_GRADE_CHARACTERISTICS["behavioral_scenarios"][behavior_type]
        trigger = random.choice(behavior_info["triggers"])
        manifestation = random.choice(behavior_info["manifestations"])
        
        # Build detailed scenario description
        scenario_description = (
            f"Time: {time_of_day}. {class_energy[time_of_day]}.\n\n"
            f"Student Profile:\n"
            f"- Learning style: {student_context['learning_style']}\n"
            f"- Seating: {student_context['seating']}\n"
            f"- Peer interaction: {student_context['peer_interactions']}\n"
            f"- Current challenges: {', '.join(student_context['current_challenges'])}\n\n"
            f"Situation:\n"
            f"During {subject} class, while working on {difficulty}, "
            f"the student is {manifestation} after {trigger}."
        )
        
        return {
            "subject": subject,
            "description": scenario_description,
            "difficulty": difficulty,
            "time_of_day": time_of_day,
            "student_context": student_context,
            "behavioral_context": {
                "type": behavior_type,
                "trigger": trigger,
                "manifestation": manifestation,
                "effective_interventions": behavior_info["effective_interventions"]
            },
            "learning_objectives": [
                f"Master {difficulty}",
                f"Develop {behavior_type} management strategies",
                "Build confidence through successful experiences"
            ],
            "student_state": self.personality["current_state"].copy()
        }

    def _generate_student_state(self, trigger):
        """
        Generate an updated student state based on triggers and context.
        
        This internal method updates the student's emotional and cognitive
        state based on various triggers and the current context.
        
        Args:
            trigger: Event or action that may affect student state
        
        Updates:
            - Engagement level
            - Understanding
            - Emotional state
            - Energy level
            - Focus and attention
        
        Note:
            State changes are influenced by:
            - Previous state
            - Time of day
            - Recent interactions
            - Environmental factors
        """
        # Adjust current state based on trigger and personality
        self.personality["current_state"]["engagement"] += random.uniform(-0.2, 0.2)
        self.personality["current_state"]["mood"] += random.uniform(-0.1, 0.1)
        
        # Ensure values stay within bounds
        for state in self.personality["current_state"]:
            self.personality["current_state"][state] = max(0.0, min(1.0, self.personality["current_state"][state]))
        
        return self.personality["current_state"].copy()

    def simulate_student_response(self, teacher_response: str, scenario: dict) -> str:
        """
        Simulate a realistic student response to a teacher's action.
        
        This method generates contextually appropriate student responses by considering:
            - Current student state (mood, energy, engagement)
            - Teacher's approach and effectiveness
            - Subject matter and difficulty
            - Student's personality traits
            - Time of day and previous interactions
        
        Args:
            teacher_response (str): The teacher's action or statement
            scenario (dict): Current teaching scenario context including:
                - Subject matter
                - Learning objectives
                - Previous interactions
                - Environmental factors
        
        Returns:
            str: A realistic student response that reflects:
                - Understanding level
                - Emotional state
                - Engagement level
                - Learning style preferences
        """
        try:
            return self.llm.generate_response(teacher_response, scenario)
        except Exception as e:
            print(f"Error generating LLM response: {e}")
            # Fallback to template responses based on behavioral context
            behavior_type = scenario["behavioral_context"]["type"]
            if behavior_type == "attention":
                return random.choice([
                    "*tries to focus* Okay...",
                    "*looks back at the work* Can you show me again?",
                    "*fidgets less* I'll try to pay attention"
                ])
            elif behavior_type == "frustration":
                return random.choice([
                    "*takes a deep breath* This is hard...",
                    "*looks less tense* Can you help me?",
                    "Maybe I can try one more time..."
                ])
            else:
                return random.choice([
                    "Okay...",
                    "*nods quietly*",
                    "Can you explain again?"
                ])

    def evaluate_response(self, teacher_response: str, scenario: dict) -> dict:
        """Evaluate teacher's response with enhanced criteria."""
        # Calculate base score and feedback
        score = 0.0
        feedback = []
        suggestions = []
        explanations = []
        
        # Extract context
        time_of_day = scenario["time_of_day"]
        student_context = scenario["student_context"]
        behavior_type = scenario["behavioral_context"]["type"]
        subject = scenario["subject"]
        
        # 1. Time-appropriate strategies (20% of score)
        time_strategies = TEACHING_STRATEGIES["engagement"]
        if any(strategy.lower() in teacher_response.lower() for strategy in time_strategies):
            score += 0.2
            feedback.append(f"✓ Good use of engaging strategy")
        else:
            suggestions.append(f"Consider engagement strategies like: {time_strategies[0]}")
            explanations.append(f"Different times of day require different approaches")
        
        # 2. Learning style alignment (20% of score)
        style_strategies = TEACHING_STRATEGIES["differentiation"]
        if any(strategy.lower() in teacher_response.lower() for strategy in style_strategies):
            score += 0.2
            feedback.append(f"✓ Good alignment with {student_context['learning_style']} learning style")
        else:
            suggestions.append(f"Include differentiated learning approaches")
            explanations.append(f"{student_context['learning_style']} learners benefit from specific strategies")
        
        # 3. Behavioral response (30% of score)
        behavior_strategies = TEACHING_STRATEGIES["support"]
        if any(strategy.lower() in teacher_response.lower() for strategy in behavior_strategies):
            score += 0.3
            feedback.append("✓ Good behavioral management approach")
        else:
            suggestions.append(f"Try supportive strategies for {behavior_type} behavior")
            explanations.append(f"Students showing {behavior_type} need specific support")
        
        # Generate student reaction based on score
        if score >= 0.8:
            reaction = self._generate_positive_reaction()
        elif score >= 0.4:
            reaction = self._generate_neutral_reaction()
        else:
            reaction = self._generate_negative_reaction()
        
        return {
            'score': score,
            'feedback': feedback,
            'suggestions': suggestions,
            'explanations': explanations,
            'student_reaction': reaction,
            'state_changes': {
                'engagement': score/2,
                'understanding': score/2,
                'mood': score/2
            }
        }

    def _generate_positive_reaction(self) -> str:
        """Generate a positive student reaction."""
        return random.choice([
            "*sits up straighter* Oh, I think I get it now!",
            "*nods enthusiastically* Can I try the next one?",
            "*smiles* That makes it easier to understand!",
            "*looks more confident* I want to try it myself!",
            "*focuses on the work* This isn't as hard as I thought!"
        ])

    def _generate_neutral_reaction(self) -> str:
        """Generate a neutral student reaction."""
        return random.choice([
            "*listens carefully* Okay, I'll try...",
            "*thinks about it* Maybe... can you show me again?",
            "*nods slowly* I think I'm starting to understand...",
            "*looks uncertain* Should I try it this way?",
            "*pays more attention* Can you explain that part again?"
        ])

    def _generate_negative_reaction(self) -> str:
        """Generate a negative student reaction."""
        return random.choice([
            "*still looks confused* I don't understand...",
            "*slumps in chair* This is too hard...",
            "*seems discouraged* Everyone else gets it except me...",
            "*fidgets more* Can we do something else?",
            "*avoids eye contact* I'm not good at this..."
        ])

    def start_interactive_session(self, category=None):
        """
        Start an interactive teaching simulation session.
        
        This method initiates a dynamic teaching scenario where the agent:
            - Sets up the initial teaching context
            - Manages the flow of interaction
            - Provides real-time feedback
            - Adapts to teaching approaches
            - Simulates student responses
        
        Args:
            category (str, optional): Specific category or subject focus.
                Defaults to None for general teaching scenarios.
        
        The session progresses through several phases:
            1. Initial scenario setup
            2. Teacher response collection
            3. Student behavior simulation
            4. Real-time feedback generation
            5. Scenario adaptation based on interaction
        
        Note:
            The session continues until explicitly ended or learning
            objectives are met.
        """
        # Check if profile is set up
        if not self.teacher_profile["name"]:
            self.setup_teacher_profile()

        # Generate initial scenario based on teacher's profile
        scenario = self.generate_scenario(
            subject=random.choice(self.teacher_profile["subject_preferences"]) if self.teacher_profile["subject_preferences"] else None
        )
        session_history = []
        
        print(f"\n=== Interactive Teaching Session for {self.teacher_profile['name']} ===")
        print("\nScenario:", scenario['description'])
        print("\nTeaching Context:")
        print(f"- Subject: {scenario['subject'].title()}")
        print(f"- Specific Challenge: {scenario['difficulty']}")
        print(f"- Recommended Strategy: {scenario['recommended_strategy']}")
        print("\nLearning Objectives:")
        for i, objective in enumerate(scenario['learning_objectives'], 1):
            print(f"{i}. {objective}")
        
        print("\nStudent's Initial State:")
        for state, value in scenario['student_state'].items():
            print(f"- {state}: {value:.2f}")
        
        # Generate initial student reaction based on scenario
        behavior_type = scenario["behavioral_context"]["type"]
        manifestation = scenario["behavioral_context"]["manifestation"]
        initial_reactions = {
            "attention": {
                "fidgeting": "*fidgets in chair* Is it time to go home yet?",
                "looking around": "*looks out the window* What's happening outside?",
                "doodling": "*continues drawing* Math is boring...",
                "asking off-topic questions": "Can we have recess now?"
            },
            "frustration": {
                "giving up quickly": "*puts head down* I can't do this...",
                "saying 'I can't do it'": "This is too hard! I'm not good at math.",
                "becoming withdrawn": "*stares at paper silently*",
                "acting out": "*pushes worksheet away* I don't want to do this!"
            }
        }
        
        # Get initial reaction
        initial_reaction = initial_reactions.get(behavior_type, {}).get(
            manifestation, 
            random.choice(SECOND_GRADE_CHARACTERISTICS["social_emotional"]["common_expressions"])
        )
        
        print("\nStudent:", initial_reaction)
        
        while True:
            print("\n" + "="*50)
            print("\nWhat would you like to do?")
            print("1. Respond to the student")
            print("2. View student's current state")
            print("3. View session history")
            print("4. View teaching context")
            print("5. Start new scenario")
            print("6. End session")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print(f"\n[{self.teacher_profile['name']}]")
                teacher_response = input("Your response: ").strip()
                if not teacher_response:
                    print("Please enter a valid response.")
                    continue
                
                # Evaluate response and get student's reaction
                result = self.evaluate_response(teacher_response, scenario)
                
                # Store interaction in session history
                session_history.append({
                    "teacher_response": teacher_response,
                    "student_reaction": result["student_reaction"],
                    "feedback": result["feedback"],
                    "score": result["score"]
                })
                
                # Display results
                print("\nStudent:", result["student_reaction"])
                if result["feedback"]:
                    print("\nFeedback:", " | ".join(result["feedback"]))
                print(f"Response Score: {result['score']:.2f}")
                
            elif choice == "2":
                print("\nCurrent Student State:")
                for state, value in self.personality["current_state"].items():
                    print(f"- {state}: {value:.2f}")
                    
            elif choice == "3":
                if not session_history:
                    print("\nNo interactions yet.")
                else:
                    print("\n=== Session History ===")
                    for i, interaction in enumerate(session_history, 1):
                        print(f"\nInteraction {i}:")
                        print(f"Teacher: {interaction['teacher_response']}")
                        print(f"Student: {interaction['student_reaction']}")
                        print(f"Score: {interaction['score']:.2f}")
                        print(f"Feedback: {interaction['feedback']}")
                        
            elif choice == "4":
                print("\nTeaching Context:")
                print(f"- Subject: {scenario['subject'].title()}")
                print(f"- Specific Challenge: {scenario['difficulty']}")
                print(f"- Recommended Strategy: {scenario['recommended_strategy']}")
                print("\nLearning Objectives:")
                for i, objective in enumerate(scenario['learning_objectives'], 1):
                    print(f"{i}. {objective}")
                    
            elif choice == "5":
                scenario = self.generate_scenario(
                    subject=random.choice(self.teacher_profile["subject_preferences"]) if self.teacher_profile["subject_preferences"] else None
                )
                session_history = []
                print("\nNew Scenario:", scenario['description'])
                print("\nTeaching Context:")
                print(f"- Subject: {scenario['subject'].title()}")
                print(f"- Specific Challenge: {scenario['difficulty']}")
                print(f"- Recommended Strategy: {scenario['recommended_strategy']}")
                print("\nLearning Objectives:")
                for i, objective in enumerate(scenario['learning_objectives'], 1):
                    print(f"{i}. {objective}")
            
            elif choice == "6":
                if session_history:
                    print(f"\n=== Session Summary for {self.teacher_profile['name']} ===")
                    avg_score = sum(i["score"] for i in session_history) / len(session_history)
                    print(f"Average Response Score: {avg_score:.2f}")
                    print(f"Total Interactions: {len(session_history)}")
                print("\nThank you for participating in this training session!")
                break
                
            else:
                print("\nInvalid choice. Please enter a number between 1 and 6.")

    def get_initial_reaction(self, scenario: dict) -> str:
        """Generate initial student reaction based on scenario."""
        behavior_type = scenario["behavioral_context"]["type"]
        manifestation = scenario["behavioral_context"]["manifestation"]
        
        # Get emotional state
        engagement = scenario["student_state"]["engagement"]
        understanding = scenario["student_state"]["understanding"]
        mood = scenario["student_state"]["mood"]
        
        initial_reactions = {
            "attention": {
                "fidgeting": [
                    "*fidgets in chair* Is it time to go home yet?",
                    "*can't sit still* This is taking forever...",
                    "*moves around restlessly* When's recess?"
                ],
                "looking around": [
                    "*looks out the window* What's happening outside?",
                    "*glances around the room* The clock is so slow today...",
                    "*distracted* Did you see that bird?"
                ],
                "doodling": [
                    "*continues drawing* Math is boring...",
                    "*focused on doodling* I like drawing better than math.",
                    "*adds details to drawing* Can I color instead?"
                ]
            },
            "frustration": {
                "giving up quickly": [
                    "*puts head down* I can't do this...",
                    "*pushes paper away* It's too hard!",
                    "*slumps in chair* I'm not good at math..."
                ],
                "saying 'I can't do it'": [
                    "This is too hard! I'm not good at math.",
                    "*looks discouraged* I'll never understand this.",
                    "*close to tears* Everyone else gets it except me..."
                ],
                "becoming withdrawn": [
                    "*stares at paper silently*",
                    "*avoids eye contact* ...",
                    "*hunches over desk* I don't know how to start..."
                ]
            }
        }
        
        # Get appropriate reactions for the behavior
        reactions = initial_reactions.get(behavior_type, {}).get(manifestation, [
            "I don't know what to do...",
            "*looks confused* This is hard.",
            "Can we do something else?"
        ])
        
        # Choose reaction based on emotional state
        if understanding < 0.4:
            reactions.extend([
                "I don't get any of this...",
                "*looks lost* What are we supposed to do?",
                "This doesn't make sense..."
            ])
        
        if mood < 0.4:
            reactions.extend([
                "*sighs heavily* I hate math...",
                "Why do we have to do this?",
                "*frowns* This is the worst..."
            ])
        
        return random.choice(reactions)

def process_teacher_response(teacher_response, current_scenario, student_state):
    """
    Process and evaluate a teacher's response in the current context.
    
    This method performs a comprehensive analysis of the teacher's
    response considering multiple factors and contexts.
    
    Args:
        teacher_response (str): The teacher's action or statement
        current_scenario (dict): Current teaching context
        student_state (dict): Current student state and characteristics
    
    Evaluation aspects:
        - Pedagogical appropriateness
        - Student needs alignment
        - Learning objective progress
        - Engagement effectiveness
        - Classroom management
    
    Returns:
        dict: Detailed analysis including:
            - Effectiveness score
            - Specific strengths
            - Areas for improvement
            - Alternative approaches
            - Impact prediction
    """
    # Evaluate teacher response based on scenario context and student state.
    evaluation = evaluate_teacher_response(teacher_response, current_scenario, student_state)
    
    # Use f-strings so that teacher_response is printed safely without manual sanitation.
    print(f"Teacher response evaluated: {teacher_response}")
    print(f"Evaluation Score: {evaluation['score']}")
    if evaluation["feedback"]:
        print("Feedback:")
        for feedback in evaluation["feedback"]:
            print(f"- {feedback}")
    else:
        print("Great job!") 