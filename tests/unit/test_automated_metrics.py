"""Tests for automated metrics evaluation."""

import unittest
import numpy as np
from src.evaluation.metrics.automated_metrics import AutomatedMetricsEvaluator

class TestAutomatedMetricsEvaluator(unittest.TestCase):
    """Test cases for AutomatedMetricsEvaluator."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.evaluator = AutomatedMetricsEvaluator()
        
        # Sample data
        cls.student_profile = {
            'grade_level': '5th',
            'learning_style': 'visual',
            'interests': ['science', 'art']
        }
        
        cls.context = """
        Teaching the concept of photosynthesis to a 5th grade student.
        The student has shown interest in plants and how they grow.
        Previous lessons covered basic plant parts.
        """
        
        cls.good_response = """
        Let's explore how plants make their own food! Think about plants like tiny 
        factories. First, they take in sunlight through their leaves. Then, they 
        combine this sunlight with water from their roots and carbon dioxide from 
        the air. Can you imagine this process happening in the leaves?
        
        For example, it's similar to how you might mix ingredients to make cookies,
        but plants mix sunlight, water, and air to make sugar! What do you think 
        happens to this sugar? Does this help explain why plants need sunlight to grow?
        """
        
        cls.poor_response = """
        Photosynthesis is the process by which plants convert light energy into 
        chemical energy that can later be released to fuel the organism's activities.
        This process occurs in the chloroplasts, specifically using chlorophyll
        pigments in the thylakoids of the chloroplasts.
        """
    
    def test_evaluate_good_response(self):
        """Test evaluation of a well-crafted response."""
        metrics = self.evaluator.evaluate_response(
            self.good_response,
            self.student_profile,
            self.context
        )
        
        self.assertGreater(metrics['overall_score'], 0.7)
        self.assertGreater(metrics['engagement_score'], 0.7)
        self.assertGreater(metrics['clarity_score'], 0.7)
        self.assertTrue(len(metrics['feedback']) < 2)
    
    def test_evaluate_poor_response(self):
        """Test evaluation of a poorly crafted response."""
        metrics = self.evaluator.evaluate_response(
            self.poor_response,
            self.student_profile,
            self.context
        )
        
        self.assertLess(metrics['overall_score'], 0.7)
        self.assertLess(metrics['engagement_score'], 0.5)
        self.assertLess(metrics['age_appropriate_score'], 0.6)
        self.assertTrue(len(metrics['feedback']) >= 2)
    
    def test_clarity_evaluation(self):
        """Test clarity metric calculation."""
        clear_text = "The sun gives plants energy to grow. Plants use this energy to make food."
        complex_text = "The photosynthetic process facilitates the transformation of electromagnetic radiation into glucose via chloroplastic organelles."
        
        clear_score = self.evaluator._evaluate_clarity(clear_text)
        complex_score = self.evaluator._evaluate_clarity(complex_text)
        
        self.assertGreater(clear_score, complex_score)
        self.assertGreater(clear_score, 0.7)
        self.assertLess(complex_score, 0.5)
    
    def test_engagement_evaluation(self):
        """Test engagement metric calculation."""
        engaging_text = "Can you imagine how plants grow? Let's explore this together! What do you think plants need?"
        flat_text = "Plants require sunlight, water, and carbon dioxide for photosynthesis."
        
        engaging_score = self.evaluator._evaluate_engagement(engaging_text)
        flat_score = self.evaluator._evaluate_engagement(flat_text)
        
        self.assertGreater(engaging_score, flat_score)
        self.assertGreater(engaging_score, 0.7)
        self.assertLess(flat_score, 0.3)
    
    def test_context_alignment(self):
        """Test context alignment metric."""
        aligned_text = "Since we learned about plant parts, let's see how leaves help plants make food through photosynthesis."
        unaligned_text = "The molecular structure of chlorophyll includes a magnesium ion at its center."
        
        aligned_score = self.evaluator._evaluate_context_alignment(
            aligned_text,
            self.context
        )
        unaligned_score = self.evaluator._evaluate_context_alignment(
            unaligned_text,
            self.context
        )
        
        self.assertGreater(aligned_score, unaligned_score)
        self.assertGreater(aligned_score, 0.6)
    
    def test_age_appropriateness(self):
        """Test age appropriateness metric."""
        appropriate_text = "Plants are like tiny factories that make their own food using sunlight."
        inappropriate_text = "The light-dependent reactions of photosynthesis occur in the thylakoid membrane."
        
        appropriate_score = self.evaluator._evaluate_age_appropriateness(
            appropriate_text,
            self.student_profile
        )
        inappropriate_score = self.evaluator._evaluate_age_appropriateness(
            inappropriate_text,
            self.student_profile
        )
        
        self.assertGreater(appropriate_score, inappropriate_score)
        self.assertGreater(appropriate_score, 0.7)
        self.assertLess(inappropriate_score, 0.5)
    
    def test_feedback_generation(self):
        """Test feedback generation."""
        metrics = {
            'clarity_score': 0.4,
            'engagement_score': 0.8,
            'context_score': 0.3,
            'age_appropriate_score': 0.9,
            'pedagogical_score': 0.5
        }
        
        feedback = self.evaluator._generate_feedback(metrics)
        
        self.assertTrue(any('simplifying' in f.lower() for f in feedback))
        self.assertTrue(any('context' in f.lower() for f in feedback))
        self.assertFalse(any('grade' in f.lower() for f in feedback))

if __name__ == '__main__':
    unittest.main() 