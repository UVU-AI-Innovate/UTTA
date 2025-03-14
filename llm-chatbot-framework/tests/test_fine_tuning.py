"""
Test suite for the fine-tuning components.
"""

import pytest
import dspy
import json
from pathlib import Path
from src.fine_tuning.trainer import PedagogicalTrainer, TrainingExample

@pytest.fixture
def base_model():
    """Create a base model for testing."""
    return dspy.LM("gpt-3.5-turbo")

@pytest.fixture
def trainer(base_model, tmp_path):
    """Create a trainer instance with temporary directory."""
    return PedagogicalTrainer(base_model, data_path=str(tmp_path))

@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    return [
        TrainingExample(
            query="How do I teach addition to a 6-year-old?",
            response="""
            Start with concrete objects like blocks or candies.
            First, show them how to count individual items.
            Then, demonstrate combining groups: "If you have 2 blocks and get 3 more,
            let's count how many you have altogether."
            Use visual aids and encourage hands-on practice.
            """,
            metadata={
                "student_level": "beginner",
                "topic": "mathematics",
                "subtopic": "addition"
            },
            labels=["elementary_math", "concrete_learning"]
        ),
        TrainingExample(
            query="Explain photosynthesis to middle school students",
            response="""
            Think of plants as tiny factories that make their own food.
            They use sunlight as energy, CO2 from the air, and water from the soil.
            Just like we need energy from food, plants create their energy through
            this process called photosynthesis.
            For example, leaves are like solar panels capturing sunlight.
            """,
            metadata={
                "student_level": "intermediate",
                "topic": "science",
                "subtopic": "biology"
            },
            labels=["life_science", "energy_conversion"]
        )
    ]

def test_data_preparation(trainer, sample_training_data):
    """Test training data preparation."""
    trainer.prepare_training_data(sample_training_data)
    
    # Check if data file was created
    data_file = Path(trainer.data_path) / "training_data.json"
    assert data_file.exists()
    
    # Verify data format
    with open(data_file, 'r') as f:
        saved_data = json.load(f)
    
    assert len(saved_data) == len(sample_training_data)
    assert 'input' in saved_data[0]
    assert 'output' in saved_data[0]
    assert 'query' in saved_data[0]['input']
    assert 'metadata' in saved_data[0]['input']

def test_training_signature(trainer):
    """Test creation of training signature."""
    signature = trainer.create_training_signature()
    
    assert isinstance(signature, dspy.Signature)
    assert 'query' in signature.inputs
    assert 'metadata' in signature.inputs
    assert 'response' in signature.outputs
    assert 'labels' in signature.outputs

def test_model_training(trainer, sample_training_data):
    """Test model training process."""
    trainer.prepare_training_data(sample_training_data)
    
    # Test with minimal epochs for quick testing
    trained_module = trainer.train(num_epochs=1, batch_size=2)
    assert isinstance(trained_module, dspy.Module)

def test_model_evaluation(trainer, sample_training_data):
    """Test model evaluation."""
    # Prepare and train model
    trainer.prepare_training_data(sample_training_data)
    trained_module = trainer.train(num_epochs=1, batch_size=2)
    
    # Evaluate on test examples
    test_examples = [
        TrainingExample(
            query="How do I explain multiplication to beginners?",
            response="Use repeated addition and groups of objects to demonstrate multiplication",
            metadata={"student_level": "beginner"},
            labels=["elementary_math"]
        )
    ]
    
    results = trainer.evaluate(test_examples, trained_module)
    
    assert 'accuracy' in results
    assert 'relevance_score' in results
    assert 'pedagogical_score' in results
    assert all(0 <= score <= 1 for score in results.values())

def test_metric_calculations(trainer):
    """Test individual metric calculations."""
    # Test accuracy calculation
    accuracy = trainer._calculate_accuracy(
        "The cat sat on the mat",
        "The cat sat on the rug"
    )
    assert 0 <= accuracy <= 1
    
    # Test relevance calculation
    relevance = trainer._calculate_relevance(
        "Addition is combining numbers together",
        "How do I teach addition?"
    )
    assert 0 <= relevance <= 1
    
    # Test pedagogical score calculation
    ped_score = trainer._calculate_pedagogical_score(
        "For example, think of addition as combining groups. Because when we add, "
        "we're putting things together. What would happen if you had 2 apples and "
        "got 3 more?",
        {"student_level": "beginner"}
    )
    assert 0 <= ped_score <= 1

def test_error_handling(trainer):
    """Test error handling in trainer."""
    # Test training without prepared data
    with pytest.raises(ValueError):
        trainer.train()
    
    # Test with invalid examples
    with pytest.raises(TypeError):
        trainer.prepare_training_data(["invalid data"])
    
    # Test with empty examples
    trainer.prepare_training_data([])
    with pytest.raises(ValueError):
        trainer.train()

if __name__ == "__main__":
    pytest.main([__file__]) 