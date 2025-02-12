"""
Knowledge base containing teaching strategies and student characteristics.
"""

SECOND_GRADE_CHARACTERISTICS = {
    "cognitive": {
        "academic_skills": {
            "math": {
                "basic": ["counting to 100", "basic addition", "basic subtraction"],
                "intermediate": ["skip counting", "place value", "mental math"],
                "challenging": ["word problems", "beginning multiplication", "money math"]
            },
            "reading": {
                "basic": ["sight words", "phonics", "basic comprehension"],
                "intermediate": ["fluency", "vocabulary", "main idea"],
                "challenging": ["inference", "context clues", "summarizing"]
            }
        }
    },
    "behavioral_scenarios": {
        "attention": {
            "triggers": [
                "long tasks",
                "complex instructions",
                "distracting environment",
                "abstract concepts"
            ],
            "manifestations": [
                "fidgeting",
                "looking around",
                "doodling",
                "asking off-topic questions"
            ],
            "effective_interventions": [
                "break tasks into smaller parts",
                "use visual aids",
                "incorporate movement",
                "provide frequent breaks"
            ]
        },
        "frustration": {
            "triggers": [
                "difficult problems",
                "repeated mistakes",
                "peer comparison",
                "time pressure"
            ],
            "manifestations": [
                "giving up quickly",
                "saying 'I can't do it'",
                "becoming withdrawn",
                "acting out"
            ],
            "effective_interventions": [
                "provide encouragement",
                "scaffold learning",
                "celebrate small successes",
                "offer alternative approaches"
            ]
        }
    },
    "social_emotional": {
        "common_expressions": [
            "This is too hard!",
            "I don't get it...",
            "Can you help me?",
            "I'm trying my best"
        ]
    }
}

TEACHING_STRATEGIES = {
    "engagement": [
        "Use hands-on materials",
        "Incorporate movement",
        "Connect to real life",
        "Use visual aids"
    ],
    "differentiation": [
        "Provide multiple examples",
        "Offer choice in activities",
        "Adjust difficulty levels",
        "Use varied approaches"
    ],
    "support": [
        "Give specific praise",
        "Break tasks into steps",
        "Model thinking process",
        "Provide scaffolding"
    ]
} 