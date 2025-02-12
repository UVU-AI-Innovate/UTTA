"""Knowledge base for second-grade student simulation."""

SECOND_GRADE_CHARACTERISTICS = {
    "cognitive": {
        "attention_span": {
            "typical_duration": "15-20 minutes for focused activities",
            "factors": [
                "time of day (better in morning)",
                "interest level in subject",
                "physical comfort",
                "hunger/fatigue",
                "classroom environment"
            ],
            "signs_of_fatigue": [
                "increased fidgeting",
                "looking around room",
                "asking unrelated questions",
                "slumping in chair",
                "playing with objects"
            ]
        },
        "learning_patterns": {
            "memory": "Short-term memory improving, needs repetition",
            "processing_speed": "Variable, often needs time to think",
            "abstract_thinking": "Developing but needs concrete examples",
            "sequencing": "Can follow 2-3 step instructions",
            "problem_solving": "Beginning to use multiple strategies"
        },
        "academic_skills": {
            "math": {
                "mastered": [
                    "counting to 100",
                    "basic addition facts",
                    "simple patterns",
                    "identifying coins"
                ],
                "developing": [
                    "two-digit addition",
                    "beginning subtraction",
                    "skip counting",
                    "telling time",
                    "counting money"
                ],
                "challenging": [
                    "word problems",
                    "regrouping",
                    "mental math",
                    "explaining thinking"
                ]
            },
            "reading": {
                "mastered": [
                    "letter sounds",
                    "sight words",
                    "basic decoding",
                    "simple sentences"
                ],
                "developing": [
                    "reading fluency",
                    "comprehension strategies",
                    "predicting outcomes",
                    "main idea identification"
                ],
                "challenging": [
                    "complex vocabulary",
                    "inference making",
                    "summarizing",
                    "reading expression"
                ]
            }
        }
    },

    "social_emotional": {
        "developmental_stage": {
            "self_awareness": "Developing sense of competence",
            "peer_relations": "Growing importance of friendships",
            "adult_relations": "Seeks approval and validation",
            "emotional_control": "Learning to manage feelings"
        },
        "common_behaviors": {
            "positive": [
                "eagerness to please",
                "showing empathy",
                "helping others",
                "sharing experiences",
                "following routines"
            ],
            "challenging": [
                "comparing self to peers",
                "fear of making mistakes",
                "difficulty with criticism",
                "wanting immediate attention",
                "emotional sensitivity"
            ]
        },
        "emotional_expressions": {
            "happy": [
                "*sits up straight with bright eyes*",
                "*raises hand excitedly*",
                "*smiles proudly*",
                "I did it!",
                "Can I show you something?"
            ],
            "frustrated": [
                "*slumps in chair*",
                "*crosses arms*",
                "*furrows brow*",
                "This is too hard!",
                "I can't do it!"
            ],
            "anxious": [
                "*fidgets with pencil*",
                "*looks at others' work*",
                "*asks repeated questions*",
                "Is this right?",
                "What if I make a mistake?"
            ],
            "disengaged": [
                "*stares out window*",
                "*doodles on paper*",
                "*plays with supplies*",
                "When is recess?",
                "I'm tired..."
            ]
        }
    },

    "learning_styles": {
        "visual": {
            "preferences": [
                "pictures and diagrams",
                "color coding",
                "watching demonstrations",
                "written instructions",
                "graphic organizers"
            ],
            "signs_of_engagement": [
                "looking intently at visuals",
                "drawing to explain ideas",
                "organizing materials neatly",
                "noticing visual details"
            ],
            "effective_strategies": [
                "using charts and graphs",
                "highlighting key information",
                "providing visual examples",
                "using gesture and movement"
            ]
        },
        "auditory": {
            "preferences": [
                "verbal explanations",
                "reading aloud",
                "discussions",
                "songs and rhymes",
                "sound patterns"
            ],
            "signs_of_engagement": [
                "mouthing words while reading",
                "repeating instructions",
                "enjoying class discussions",
                "learning through songs"
            ],
            "effective_strategies": [
                "think-pair-share",
                "verbal step-by-step instructions",
                "using rhythm and music",
                "encouraging self-talk"
            ]
        },
        "kinesthetic": {
            "preferences": [
                "hands-on activities",
                "movement-based learning",
                "manipulatives",
                "role-playing",
                "physical demonstrations"
            ],
            "signs_of_engagement": [
                "using fingers to count",
                "wanting to demonstrate",
                "moving while thinking",
                "touching materials"
            ],
            "effective_strategies": [
                "using manipulatives",
                "incorporating movement",
                "hands-on experiments",
                "active learning games"
            ]
        }
    },

    "behavioral_scenarios": {
        "attention": {
            "triggers": [
                "complex instructions",
                "long periods of sitting",
                "abstract concepts",
                "distracting environment",
                "tiredness or hunger",
                "after recess excitement",
                "end of day fatigue"
            ],
            "manifestations": [
                "fidgeting in seat",
                "looking around room",
                "playing with objects",
                "asking off-topic questions",
                "getting up frequently",
                "daydreaming"
            ],
            "effective_interventions": [
                "movement breaks",
                "chunking information",
                "visual aids",
                "proximity control",
                "clear routines",
                "varied activities"
            ]
        },
        "frustration": {
            "triggers": [
                "challenging tasks",
                "repeated mistakes",
                "peer comparison",
                "time pressure",
                "unclear expectations",
                "fear of failure",
                "perfectionism"
            ],
            "manifestations": [
                "giving up quickly",
                "negative self-talk",
                "avoiding work",
                "acting out",
                "becoming withdrawn",
                "crying or anger"
            ],
            "effective_interventions": [
                "breaking tasks into steps",
                "providing choice",
                "positive reinforcement",
                "growth mindset language",
                "acknowledging feelings",
                "offering help"
            ]
        }
    }
}

TEACHING_STRATEGIES = {
    "engagement": {
        "attention_grabbers": [
            "mystery box or bag",
            "special signals",
            "voice variation",
            "surprise elements",
            "interactive countdowns"
        ],
        "transitions": [
            "musical cues",
            "movement activities",
            "cleanup games",
            "routine signals",
            "clear warnings"
        ],
        "motivation": {
            "intrinsic": [
                "student choice",
                "relevant topics",
                "success opportunities",
                "curiosity sparking",
                "goal setting"
            ],
            "extrinsic": [
                "specific praise",
                "progress tracking",
                "earned privileges",
                "class rewards",
                "recognition"
            ]
        }
    },

    "instructional": {
        "direct_teaching": [
            "clear objectives",
            "step-by-step instruction",
            "modeling",
            "guided practice",
            "independent work"
        ],
        "differentiation": {
            "content": [
                "varied reading levels",
                "modified assignments",
                "choice boards",
                "interest-based tasks"
            ],
            "process": [
                "flexible grouping",
                "varied time frames",
                "multiple strategies",
                "scaffolded support"
            ],
            "product": [
                "multiple formats",
                "choice in presentation",
                "varied complexity",
                "different tools"
            ]
        }
    },

    "behavioral": {
        "preventive": [
            "clear expectations",
            "consistent routines",
            "structured environment",
            "positive relationships",
            "engaging lessons"
        ],
        "responsive": {
            "redirection": [
                "proximity control",
                "nonverbal cues",
                "quiet reminders",
                "attention signals"
            ],
            "reinforcement": [
                "specific praise",
                "earned privileges",
                "class rewards",
                "positive notes"
            ],
            "consequences": [
                "logical outcomes",
                "loss of privileges",
                "time to reflect",
                "parent contact"
            ]
        }
    }
} 