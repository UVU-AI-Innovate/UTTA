"""
Teacher Training Agent Module

This module implements the core AI agent responsible for simulating and managing
teacher training scenarios. It coordinates between the LLM interface, knowledge management,
and pedagogical processing components.

Example:
    agent = TeacherTrainingAgent()
    scenario = agent.create_teaching_scenario("mathematics", "intermediate", student_profile)
    feedback = agent.evaluate_teaching_response(teacher_input, scenario)
"""

import random
import json
from llm_handler import LLMHandler
from knowledge_base import SECOND_GRADE_CHARACTERISTICS, TEACHING_STRATEGIES
from evaluator import evaluate_teacher_response
from knowledge_manager import KnowledgeManager
from typing import Dict, Any
from .llm_handler import PedagogicalLanguageProcessor
from ..models.knowledge_manager import PedagogicalKnowledgeManager
from ..utils.llm_interface import LLMInterface

class TeacherTrainingAgent:
    """
    Main agent class for managing teacher training simulations.
    
    This class orchestrates the interaction between different components:
    - LLM Interface for AI model interactions
    - Knowledge Manager for accessing teaching strategies
    - Language Processor for analyzing responses
    
    Attributes:
        llm (LLMInterface): Interface to the language model
        knowledge (PedagogicalKnowledgeManager): Manager for teaching knowledge
        processor (PedagogicalLanguageProcessor): Processor for language analysis
    """
    
    def __init__(self):
        """Initialize the TeacherTrainingAgent with required components."""
        self.llm = LLMInterface()
        self.knowledge = PedagogicalKnowledgeManager()
        self.processor = PedagogicalLanguageProcessor()
        
    def create_teaching_scenario(
        self, 
        subject: str, 
        difficulty: str, 
        student_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a teaching scenario based on given parameters.
        
        Args:
            subject: The subject area for the scenario (e.g., "mathematics", "reading")
            difficulty: The difficulty level (e.g., "beginner", "intermediate", "advanced")
            student_profile: Dictionary containing student characteristics and learning style
            
        Returns:
            Dictionary containing the generated scenario details including:
            - context: The teaching context
            - objectives: Learning objectives
            - student_background: Relevant student information
            - suggested_approaches: Initial teaching strategy suggestions
        
        Example:
            scenario = agent.create_teaching_scenario(
                "mathematics",
                "intermediate",
                {"learning_style": "visual", "grade_level": "3rd"}
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
        
        Args:
            teacher_input: The teacher's response or action in the scenario
            scenario: The current teaching scenario context
            
        Returns:
            Dictionary containing the evaluation results including:
            - effectiveness: Overall effectiveness score
            - strengths: Identified strong points
            - areas_for_improvement: Suggested improvements
            - alternative_approaches: Other teaching strategies to consider
            
        Example:
            feedback = agent.evaluate_teaching_response(
                "I would use visual aids to demonstrate fraction concepts",
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
        strategies = self.knowledge.get_teaching_strategies(analysis["context"])
        
        # Combine analysis with teaching strategies
        return {
            "effectiveness": analysis["effectiveness_score"],
            "strengths": analysis["identified_strengths"],
            "areas_for_improvement": analysis["improvement_areas"],
            "alternative_approaches": strategies
        }

    def _load_teaching_strategies(self):
        """Load teaching strategies from knowledge base files."""
        try:
            with open('knowledge_base/teaching_strategies.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("! Teaching strategies file not found")
            return self._get_default_teaching_strategies()

    def _load_student_behaviors(self):
        """Load student behavior patterns from knowledge base files."""
        try:
            with open('knowledge_base/student_behaviors.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("! Student behaviors file not found")
            return self._get_default_student_behaviors()

    def _load_subject_content(self):
        """Load subject-specific content and strategies."""
        try:
            with open('knowledge_base/subject_content.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("! Subject content file not found")
            return self._get_default_subject_content()

    def _get_fallback_knowledge(self):
        """Provide basic fallback knowledge if files can't be loaded."""
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
        """Initialize student personality and state."""
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
        """Generate student's emotional and behavioral state based on trigger."""
        # Adjust current state based on trigger and personality
        self.personality["current_state"]["engagement"] += random.uniform(-0.2, 0.2)
        self.personality["current_state"]["mood"] += random.uniform(-0.1, 0.1)
        
        # Ensure values stay within bounds
        for state in self.personality["current_state"]:
            self.personality["current_state"][state] = max(0.0, min(1.0, self.personality["current_state"][state]))
        
        return self.personality["current_state"].copy()

    def simulate_student_response(self, teacher_response: str, scenario: dict) -> str:
        """Generate more contextual student responses."""
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
        behavior_trigger = scenario["behavioral_context"]["trigger"]
        subject = scenario["subject"]
        
        # Get relevant knowledge from vector store
        time_query = f"teaching strategies for {time_of_day}"
        style_query = f"teaching methods for {student_context['learning_style']} learners"
        behavior_query = f"handling {behavior_type} behavior triggered by {behavior_trigger}"
        subject_query = f"teaching strategies for {subject}"
        
        # Search knowledge base
        time_strategies = self.knowledge_manager.search(time_query)
        style_strategies = self.knowledge_manager.search(style_query)
        behavior_strategies = self.knowledge_manager.search(behavior_query)
        subject_strategies = self.knowledge_manager.search(subject_query)
        
        # 1. Time-appropriate strategies (20% of score)
        if any(strategy['text'].lower() in teacher_response.lower() for strategy in time_strategies):
            score += 0.2
            feedback.append(f"✓ Good use of {time_of_day} appropriate strategy")
        else:
            suggestions.append(f"Consider {time_of_day} strategies like: {time_strategies[0]['text'] if time_strategies else 'structured activities'}")
            explanations.append(f"Different times of day require different approaches")
        
        # 2. Learning style alignment (20% of score)
        if any(strategy['text'].lower() in teacher_response.lower() for strategy in style_strategies):
            score += 0.2
            feedback.append(f"✓ Good alignment with {student_context['learning_style']} learning style")
        else:
            suggestions.append(f"Include {student_context['learning_style']} learning approaches")
            explanations.append(f"{student_context['learning_style']} learners benefit from specific strategies")
        
        # 3. Behavioral response (30% of score)
        if any(strategy['text'].lower() in teacher_response.lower() for strategy in behavior_strategies):
            score += 0.3
            feedback.append("✓ Good behavioral management approach")
        else:
            suggestions.append(f"Try strategies for {behavior_type} behavior")
            explanations.append(f"Students showing {behavior_type} need specific support")
        
        # 4. Subject-specific support (30% of score)
        if any(strategy['text'].lower() in teacher_response.lower() for strategy in subject_strategies):
            score += 0.3
            feedback.append(f"✓ Good {subject}-specific support")
        else:
            suggestions.append(f"Include {subject} strategies")
            explanations.append(f"{subject} concepts need specific teaching approaches")
        
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
        """Start an interactive teaching session."""
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