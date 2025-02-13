"""
Educational Knowledge Base Module

This module contains comprehensive data structures that define teaching strategies,
student characteristics, and behavioral patterns typically observed in educational
settings. It serves as a reference for the AI teaching simulation system.

Key Components:
    - Student Characteristics: Age-appropriate academic and behavioral traits
    - Teaching Strategies: Effective pedagogical approaches
    - Behavioral Scenarios: Common situations and appropriate interventions
    - Social-Emotional Patterns: Typical student expressions and reactions

Usage:
    This knowledge base is used by the TeacherTrainingAgent to:
    - Generate realistic student behaviors
    - Evaluate teaching responses
    - Suggest appropriate interventions
    - Maintain age-appropriate interactions

Note:
    The data structures are specifically tailored for second-grade level
    education but can be adapted for other grade levels.
"""

# Comprehensive characteristics of second-grade students including
# academic abilities, behavioral patterns, and social-emotional traits
SECOND_GRADE_CHARACTERISTICS = {
    # Cognitive and academic capabilities typical for second grade
    "cognitive": {
        # Subject-specific academic skills organized by difficulty level
        "academic_skills": {
            # Mathematics skills progression
            "math": {
                "basic": ["counting to 100", "basic addition", "basic subtraction"],
                "intermediate": ["skip counting", "place value", "mental math"],
                "challenging": ["word problems", "beginning multiplication", "money math"]
            },
            # Reading skills progression
            "reading": {
                "basic": ["sight words", "phonics", "basic comprehension"],
                "intermediate": ["fluency", "vocabulary", "main idea"],
                "challenging": ["inference", "context clues", "summarizing"]
            }
        }
    },
    # Common behavioral scenarios with triggers and appropriate responses
    "behavioral_scenarios": {
        # Attention-related behaviors and interventions
        "attention": {
            # Situations that may cause attention issues
            "triggers": [
                "long tasks",
                "complex instructions",
                "distracting environment",
                "abstract concepts"
            ],
            # How attention issues typically manifest
            "manifestations": [
                "fidgeting",
                "looking around",
                "doodling",
                "asking off-topic questions"
            ],
            # Research-based intervention strategies
            "effective_interventions": [
                "break tasks into smaller parts",
                "use visual aids",
                "incorporate movement",
                "provide frequent breaks"
            ]
        },
        # Frustration-related behaviors and interventions
        "frustration": {
            # Situations that may cause frustration
            "triggers": [
                "difficult problems",
                "repeated mistakes",
                "peer comparison",
                "time pressure"
            ],
            # How frustration typically manifests
            "manifestations": [
                "giving up quickly",
                "saying 'I can't do it'",
                "becoming withdrawn",
                "acting out"
            ],
            # Effective strategies for managing frustration
            "effective_interventions": [
                "provide encouragement",
                "scaffold learning",
                "celebrate small successes",
                "offer alternative approaches"
            ]
        }
    },
    # Typical social and emotional expressions
    "social_emotional": {
        # Common phrases used by second-grade students
        "common_expressions": [
            "This is too hard!",
            "I don't get it...",
            "Can you help me?",
            "I'm trying my best"
        ]
    }
}

# Comprehensive collection of effective teaching strategies
# organized by pedagogical purpose
TEACHING_STRATEGIES = {
    # Strategies for maintaining student engagement
    "engagement": [
        "Use hands-on materials",
        "Incorporate movement",
        "Connect to real life",
        "Use visual aids"
    ],
    # Approaches for differentiating instruction
    "differentiation": [
        "Provide multiple examples",
        "Offer choice in activities",
        "Adjust difficulty levels",
        "Use varied approaches"
    ],
    # Methods for providing student support
    "support": [
        "Give specific praise",
        "Break tasks into steps",
        "Model thinking process",
        "Provide scaffolding"
    ]
} 