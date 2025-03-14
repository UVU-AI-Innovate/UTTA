"""
Test suite for the Pedagogical Language Processor.
"""

import pytest
import dspy
from src.llm.dspy.handler import (
    PedagogicalLanguageProcessor,
    TeachingContext,
    TeachingResponse,
    StudentReactionGenerator,
    DSPyLLMHandler
)

@pytest.fixture
def lm():
    """Create a DSPy language model instance for testing."""
    return DSPyLLMHandler("gpt-3.5-turbo")

@pytest.fixture
def processor(lm):
    """Create a PedagogicalLanguageProcessor instance."""
    return PedagogicalLanguageProcessor(lm)

@pytest.fixture
def reaction_generator(lm):
    """Create a StudentReactionGenerator instance."""
    return StudentReactionGenerator(lm)

def test_basic_teaching_response(processor):
    """Test basic teaching response generation."""
    query = "What is photosynthesis?"
    response = processor.get_teaching_response(query)
    
    assert isinstance(response, TeachingResponse)
    assert len(response.content) > 0
    assert len(response.follow_up_questions) > 0
    assert len(response.suggested_activities) > 0

def test_adaptive_teaching(processor):
    """Test adaptive teaching for different student levels."""
    topic = "multiplication"
    levels = ["beginner", "intermediate", "advanced"]
    
    responses = [
        processor.get_adaptive_teaching_response(
            topic=topic,
            student_level=level,
            context=f"Teaching multiplication to {level} student"
        )
        for level in levels
    ]
    
    # Verify different responses for different levels
    assert len(set(responses)) == len(levels)
    for response in responses:
        assert len(response) > 0

def test_student_response_analysis(processor):
    """Test analysis of student responses."""
    context = TeachingContext(
        student_level="intermediate",
        topic="algebra",
        previous_interactions=[],
        learning_objectives=["Understand basic algebraic expressions"]
    )
    
    student_response = "I think x + 5 = 10 means I should subtract 5 from both sides"
    analysis = processor.analyze_student_response(student_response, context)
    
    assert 'understanding_level' in analysis
    assert 'misconceptions' in analysis
    assert 'improvement_suggestions' in analysis
    assert 0 <= analysis['understanding_level'] <= 1

def test_scaffolding_levels(processor):
    """Test different scaffolding levels in responses."""
    context = TeachingContext(
        student_level="beginner",
        topic="fractions",
        previous_interactions=[],
        learning_objectives=["Understand fraction addition"]
    )
    
    response = processor.get_teaching_response("How do I add fractions?", context)
    assert response.scaffolding_level >= 0
    assert len(response.engagement_elements) > 0

def test_student_reaction_generation(reaction_generator):
    """Test generation of simulated student reactions."""
    teaching_response = """
    To add fractions, we need to:
    1. Find a common denominator
    2. Convert fractions to equivalent fractions
    3. Add the numerators
    4. Simplify if possible
    """
    
    student_profile = {
        "level": "beginner",
        "learning_style": "visual",
        "common_difficulties": ["fraction conversion"]
    }
    
    reaction = reaction_generator.generate_reaction(
        teaching_response=teaching_response,
        student_profile=student_profile,
        understanding_level=0.7
    )
    
    assert 'response' in reaction
    assert 'questions' in reaction
    assert 'misconceptions' in reaction
    assert len(reaction['response']) > 0

def test_context_handling(processor):
    """Test handling of teaching context in responses."""
    previous_interactions = [
        {"role": "student", "content": "I don't understand negative numbers"},
        {"role": "teacher", "content": "Think of them as numbers below zero"}
    ]
    
    context = TeachingContext(
        student_level="beginner",
        topic="negative numbers",
        previous_interactions=previous_interactions,
        learning_objectives=["Understand negative number operations"]
    )
    
    response = processor.get_teaching_response(
        "How do I add negative numbers?",
        context
    )
    
    assert isinstance(response, TeachingResponse)
    assert len(response.content) > 0
    assert response.scaffolding_level >= 0

def test_error_handling(processor):
    """Test error handling in the processor."""
    with pytest.raises(ValueError):
        processor.get_teaching_response("")
    
    with pytest.raises(ValueError):
        processor.analyze_student_response("", None)

if __name__ == "__main__":
    pytest.main([__file__]) 