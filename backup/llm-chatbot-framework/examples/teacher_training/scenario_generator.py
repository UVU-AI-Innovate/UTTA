#!/usr/bin/env python3
"""
Classroom Scenario Generator

This module generates realistic classroom scenarios based on educational knowledge
retrieved from the vector database. It creates contextually appropriate situations
that teachers might encounter in elementary education settings.
"""

import os
import sys
import json
import random
import logging
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import core components
from src.core.vector_database import VectorDatabase
from src.llm.dspy.handler import DSPyLLMHandler

# Constants
KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data')
STUDENT_PROFILES_PATH = os.path.join(KNOWLEDGE_DIR, 'student_profiles.json')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassroomScenarioGenerator:
    """
    Generate realistic classroom scenarios based on educational research.
    
    This class leverages the vector database of educational knowledge to create
    evidence-based teaching scenarios that realistically model classroom
    interactions, student behaviors, and teaching challenges.
    """
    
    def __init__(self, vector_db, llm=None):
        """
        Initialize the scenario generator.
        
        Args:
            vector_db: Vector database containing educational knowledge
            llm: Optional language model for advanced scenario generation
        """
        self.vector_db = vector_db
        self.llm = llm
        self.knowledge_base_dir = os.path.join("knowledge_base")
        self.student_profiles = self._load_student_profiles()
        
        # Grade level characteristics
        self.grade_level_characteristics = {
            "elementary": {
                "ages": "6-10",
                "subjects": ["reading", "math", "science", "social studies"],
                "challenges": ["attention span", "basic classroom routines", "foundational skills", "socialization"]
            },
            "middle": {
                "ages": "11-13",
                "subjects": ["language arts", "algebra", "history", "life science"],
                "challenges": ["peer relationships", "identity development", "academic pressure", "organization"]
            },
            "high": {
                "ages": "14-18",
                "subjects": ["literature", "advanced math", "chemistry", "psychology"],
                "challenges": ["motivation", "future planning", "complex content", "independence"]
            }
        }
    
    def _load_student_profiles(self) -> List[Dict[str, Any]]:
        """
        Load student profiles from JSON file.
        
        Returns:
            List of student profile dictionaries
        """
        profiles_path = os.path.join(self.knowledge_base_dir, "student_profiles.json")
        try:
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    data = json.load(f)
                    return data.get("profiles", [])
            else:
                logger.warning(f"Student profiles file not found: {profiles_path}")
                return []
        except Exception as e:
            logger.error(f"Error loading student profiles: {e}")
            return []
    
    def _get_relevant_knowledge(self, topic: str, category: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant knowledge chunks from the vector database.
        
        Args:
            topic: Topic to search for
            category: Knowledge category to search in
            count: Number of knowledge chunks to retrieve
            
        Returns:
            List of relevant knowledge chunks
        """
        results = self.vector_db.search(topic, top_k=count, category=category)
        return results
    
    def _get_subject_specific_knowledge(self, subject: str) -> Dict[str, Any]:
        """
        Get subject-specific knowledge from CSV files.
        
        Args:
            subject: The subject area to get knowledge for
            
        Returns:
            Dictionary with subject-specific knowledge
        """
        try:
            import csv
            
            # Look for subject-specific CSV file
            subject_file = os.path.join(
                self.knowledge_base_dir, 
                "default_strategies", 
                f"{subject}_strategies.csv"
            )
            
            if not os.path.exists(subject_file):
                logger.warning(f"Subject file not found: {subject_file}")
                return {"topics": {}}
            
            topics = {}
            with open(subject_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "topic" in row:
                        topic = row["topic"]
                        topics[topic] = {
                            "common_challenges": row.get("common_challenges", ""),
                            "strategies": row.get("strategies", ""),
                            "examples": row.get("examples", "")
                        }
            
            return {"topics": topics}
        except Exception as e:
            logger.error(f"Error loading subject knowledge: {e}")
            return {"topics": {}}
    
    def _get_behavior_knowledge(self) -> Dict[str, Any]:
        """
        Get behavior management knowledge from CSV file.
        
        Returns:
            Dictionary with behavior management strategies
        """
        try:
            import csv
            
            behavior_file = os.path.join(
                self.knowledge_base_dir, 
                "default_strategies", 
                "behavior_management.csv"
            )
            
            if not os.path.exists(behavior_file):
                logger.warning(f"Behavior management file not found: {behavior_file}")
                return {}
            
            behaviors = {}
            with open(behavior_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    behavior_type = row.get("behavior_type")
                    if behavior_type:
                        if behavior_type not in behaviors:
                            behaviors[behavior_type] = {
                                "triggers": [],
                                "manifestations": [],
                                "strategies": [],
                                "rationale": row.get("rationale", "")
                            }
                        
                        for field in ["triggers", "manifestations", "strategies"]:
                            if field in row and row[field]:
                                if row[field] not in behaviors[behavior_type][field]:
                                    behaviors[behavior_type][field].append(row[field])
            
            return behaviors
        except Exception as e:
            logger.error(f"Error loading behavior knowledge: {e}")
            return {}
    
    def _select_student_profile(self, grade_level: str, profile_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Select a student profile for the scenario.
        
        Args:
            grade_level: Educational level 
            profile_id: Optional specific profile ID to use
            
        Returns:
            Selected student profile
        """
        matching_profiles = []
        
        # If specific profile requested, try to find it
        if profile_id and self.student_profiles:
            for profile in self.student_profiles:
                if profile.get("id") == profile_id:
                    return profile
        
        # Otherwise filter by grade level
        if self.student_profiles:
            if grade_level == "elementary":
                grade_range = [1, 2, 3, 4, 5]
            elif grade_level == "middle":
                grade_range = [6, 7, 8]
            else:  # high school
                grade_range = [9, 10, 11, 12]
                
            for profile in self.student_profiles:
                try:
                    grade = int(profile.get("grade", 0))
                    if grade in grade_range:
                        matching_profiles.append(profile)
                except (ValueError, TypeError):
                    # If grade isn't a valid integer, skip this profile
                    continue
        
        if matching_profiles:
            return random.choice(matching_profiles)
        else:
            # Create a basic profile if no matching profiles found
            return self._generate_fallback_profile(grade_level)
    
    def _generate_fallback_profile(self, grade_level: str) -> Dict[str, Any]:
        """
        Generate a fallback student profile when none are available.
        
        Args:
            grade_level: Educational level
            
        Returns:
            Generated student profile
        """
        # Names for random generation
        first_names = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie"]
        
        # Generate appropriate grade for level
        if grade_level == "elementary":
            grade = random.randint(1, 5)
        elif grade_level == "middle":
            grade = random.randint(6, 8)
        else:  # high school
            grade = random.randint(9, 12)
        
        # Generate random learning style
        learning_styles = ["visual", "auditory", "kinesthetic", "reading/writing"]
        
        return {
            "id": f"gen_{random.randint(1000, 9999)}",
            "name": random.choice(first_names),
            "grade": grade,
            "age": grade + 5,  # Approximate age based on grade
            "learning_style": random.choice(learning_styles),
            "strengths": ["Generated profile - strengths not specified"],
            "challenges": ["Generated profile - challenges not specified"]
        }
    
    def generate_scenario(
        self, 
        grade_level: str = "elementary", 
        subject: str = "math", 
        challenge_type: Optional[str] = None,
        student_profile_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a classroom scenario.
        
        Args:
            grade_level: Educational level (elementary, middle, high)
            subject: Subject area
            challenge_type: Type of classroom challenge to focus on
            student_profile_id: ID of a specific student profile to use
            
        Returns:
            Dictionary with the generated scenario
        """
        # 1. Get relevant academic content knowledge
        subject_knowledge = self._get_subject_specific_knowledge(subject)
        
        # 2. Get behavior management knowledge
        behavior_knowledge = self._get_behavior_knowledge()
        
        # 3. Select student profile
        student_profile = self._select_student_profile(grade_level, student_profile_id)
        
        # 4. Select specific challenge type if not provided
        if not challenge_type and behavior_knowledge:
            challenge_type = random.choice(list(behavior_knowledge.keys()))
        
        # 5. Get specific content topic
        available_topics = list(subject_knowledge.get("topics", {}).keys())
        topic = random.choice(available_topics) if available_topics else subject
        
        # 6. Get subject-specific challenge
        subject_challenges = subject_knowledge.get("topics", {}).get(topic, {}).get("common_challenges", "")
        
        # 7. Get behavior-specific information
        behavior_info = behavior_knowledge.get(challenge_type, {})
        triggers = behavior_info.get("triggers", ["Unspecified trigger"])
        manifestations = behavior_info.get("manifestations", ["Unspecified behavior"])
        strategies = behavior_info.get("strategies", ["No specific strategies available"])
        
        # 8. Generate scenario content
        scenario_content = self._build_scenario_content(
            grade_level=grade_level,
            subject=subject,
            topic=topic,
            student_profile=student_profile,
            subject_challenges=subject_challenges,
            behavior_type=challenge_type,
            triggers=triggers,
            manifestations=manifestations
        )
        
        # 9. Compile scenario data
        scenario = {
            "title": f"{grade_level.title()} {subject.title()} Lesson with {student_profile['name']}",
            "grade_level": grade_level,
            "subject": subject,
            "topic": topic,
            "challenge_type": challenge_type,
            "student_profile": student_profile,
            "scenario_content": scenario_content,
            "recommended_strategies": strategies[:3] if len(strategies) >= 3 else strategies,
            "knowledge_sources": [
                {
                    "category": "subject_content",
                    "source": f"{subject}_strategies.csv", 
                    "content": subject_knowledge.get("topics", {}).get(topic, {})
                },
                {
                    "category": "behavior_management",
                    "source": "behavior_management.csv",
                    "content": behavior_info
                }
            ]
        }
        
        # 10. Enhance with vector database knowledge if available
        self._enhance_with_vector_knowledge(scenario)
        
        return scenario
    
    def _build_scenario_content(
        self,
        grade_level: str,
        subject: str,
        topic: str,
        student_profile: Dict[str, Any],
        subject_challenges: str,
        behavior_type: str,
        triggers: List[str],
        manifestations: List[str]
    ) -> str:
        """
        Build detailed scenario content.
        
        Args:
            Various scenario parameters
            
        Returns:
            Formatted scenario content as a string
        """
        # Get classroom context based on grade level
        grade_info = self.grade_level_characteristics.get(grade_level, {})
        age_range = grade_info.get("ages", "unknown")
        
        # Select a trigger and manifestation
        trigger = random.choice(triggers) if triggers else "unclear trigger"
        manifestation = random.choice(manifestations) if manifestations else "unspecified behavior"
        
        # Format student name and pronouns
        student_name = student_profile.get("name", "Student")
        # Default pronouns - would be better to include these in the profile
        pronouns = {"they": "they", "them": "them", "their": "their"}
        
        # Build the scenario content
        content = [
            f"You are teaching a {grade_level} {subject} class focusing on {topic}.",
            f"Your students are approximately {age_range} years old."
        ]
        
        # Add specific subject challenges
        if subject_challenges:
            content.append(f"For this subject and topic, students commonly struggle with: {subject_challenges}")
        
        # Add student-specific context
        content.append(f"\nIn your class is {student_name}, who:")
        if "learning_style" in student_profile:
            content.append(f"- Is primarily a {student_profile['learning_style']} learner")
        if "strengths" in student_profile and student_profile["strengths"]:
            strengths = ", ".join(student_profile["strengths"][:3])
            content.append(f"- Has strengths in: {strengths}")
        if "challenges" in student_profile and student_profile["challenges"]:
            challenges = ", ".join(student_profile["challenges"][:3])
            content.append(f"- Struggles with: {challenges}")
        
        # Describe the specific situation
        content.append(f"\nDuring your lesson on {topic}:")
        content.append(f"- The trigger occurs: {trigger}")
        content.append(f"- {student_name} responds by: {manifestation}")
        content.append(f"- Other students are beginning to notice {student_name}'s behavior")
        content.append(f"- You need to address this situation while maintaining the flow of the lesson")
        
        # Add closing prompt
        content.append(f"\nHow would you respond to this situation to support {student_name} while keeping the rest of the class engaged?")
        
        return "\n".join(content)
    
    def _enhance_with_vector_knowledge(self, scenario: Dict[str, Any]) -> None:
        """
        Enhance scenario with knowledge from the vector database.
        
        Args:
            scenario: Scenario to enhance
            
        Modifies the scenario in place to add relevant knowledge.
        """
        if not hasattr(self.vector_db, 'search'):
            return
            
        try:
            # Build search query from scenario elements
            query_parts = [
                f"{scenario['grade_level']} education",
                f"{scenario['subject']} teaching",
                f"{scenario['challenge_type']} behavior",
                f"{scenario['topic']} instruction"
            ]
            query = " ".join(query_parts)
            
            # Search for relevant knowledge
            results = self.vector_db.search(query, top_k=2)
            
            if results:
                # Add to knowledge sources
                for i, result in enumerate(results):
                    scenario["knowledge_sources"].append({
                        "category": result.get("category", "general_education"),
                        "source": result.get("metadata", {}).get("source", "Unknown"),
                        "content": result.get("text", ""),
                        "relevance": result.get("similarity", 0.0)
                    })
                    
                    # Record that this chunk was used
                    if hasattr(self.vector_db, 'record_chunk_used') and "id" in result:
                        self.vector_db.record_chunk_used(result["id"])
        except Exception as e:
            logger.error(f"Error enhancing scenario with vector knowledge: {e}")
    
    def generate_enhanced_scenario(self, **kwargs) -> Dict[str, Any]:
        """
        Generate an enhanced scenario using the LLM if available.
        
        This method acts as a wrapper around generate_scenario, but uses
        the LLM to enhance the scenario with more natural language and
        realistic details if a language model is available.
        
        Args:
            **kwargs: Same parameters as generate_scenario
            
        Returns:
            Enhanced scenario dictionary
        """
        # Generate the base scenario
        scenario = self.generate_scenario(**kwargs)
        
        # If no LLM available, return the basic scenario
        if self.llm is None:
            return scenario
        
        try:
            # Enhance the scenario content using the LLM
            prompt = f"""
            Please enhance this classroom scenario to be more realistic and detailed:
            
            {scenario['scenario_content']}
            
            Make it sound like a real classroom situation, adding sensory details, 
            realistic dialogue, and classroom context. Keep the same educational 
            challenge but make the language more natural and engaging.
            """
            
            enhanced_content = self.llm.generate(prompt)
            if enhanced_content:
                scenario['scenario_content'] = enhanced_content
                scenario['llm_enhanced'] = True
        except Exception as e:
            logger.error(f"Error enhancing scenario with LLM: {e}")
            scenario['llm_enhanced'] = False
        
        return scenario


if __name__ == "__main__":
    # Example usage (without actual vector database)
    class DummyVectorDB:
        def search(self, *args, **kwargs):
            return []
    
    generator = ClassroomScenarioGenerator(DummyVectorDB())
    scenario = generator.generate_scenario(
        grade_level="elementary",
        subject="math",
        challenge_type="inattention"
    )
    
    print("\n" + "="*50)
    print("GENERATED CLASSROOM SCENARIO")
    print("="*50)
    print(f"Title: {scenario['title']}")
    print(f"Grade: {scenario['grade_level']}, Subject: {scenario['subject']}")
    print(f"Topic: {scenario['topic']}")
    print(f"Challenge: {scenario['challenge_type']}")
    print("\nScenario:")
    print(scenario['scenario_content'])
    print("\nRecommended Strategies:")
    for strategy in scenario['recommended_strategies']:
        print(f"- {strategy}")
    print("="*50) 