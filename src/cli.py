"""Command-line interface for the teaching assistant."""

import os
import sys
import logging
from pathlib import Path
import json
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

try:
    from dotenv import load_dotenv
    from llm.dspy.handler import EnhancedDSPyLLMInterface
    from knowledge_base.document_indexer import DocumentIndexer
    from metrics.automated_metrics import AutomatedMetricsEvaluator
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

# Load environment variables
load_dotenv()

class TeachingAssistantCLI:
    """Command-line interface for the teaching assistant."""
    
    def __init__(self):
        """Initialize the CLI interface."""
        try:
            self.llm = EnhancedDSPyLLMInterface()
            self.indexer = DocumentIndexer()
            self.evaluator = AutomatedMetricsEvaluator()
            
            # Default settings
            self.settings = {
                "grade_level": 5,
                "subject": "General",
                "learning_style": "Visual"
            }
            
            logger.info("Initialized Teaching Assistant CLI")
            
        except Exception as e:
            logger.error(f"Failed to initialize CLI: {e}")
            raise

    def configure_settings(self):
        """Configure teaching settings."""
        print("\nCurrent Settings:")
        for key, value in self.settings.items():
            print(f"{key}: {value}")
        
        print("\nUpdate settings (press Enter to keep current value):")
        
        # Grade level
        grade = input("Grade level (1-12): ").strip()
        if grade and grade.isdigit():
            self.settings["grade_level"] = int(grade)
        
        # Subject
        subject = input("Subject: ").strip()
        if subject:
            self.settings["subject"] = subject
        
        # Learning style
        style = input("Learning style (Visual/Auditory/Reading/Kinesthetic): ").strip()
        if style:
            self.settings["learning_style"] = style

    def add_document(self):
        """Add a document to the knowledge base."""
        print("\nAdd Document to Knowledge Base")
        
        # Get file path
        while True:
            path = input("Enter file path (or 'q' to cancel): ").strip()
            if path.lower() == 'q':
                return
                
            if os.path.exists(path):
                break
            print("File not found. Please try again.")
        
        try:
            # Read file
            with open(path, 'r') as f:
                content = f.read()
            
            # Get metadata
            print("\nDocument Metadata:")
            subject = input("Subject: ").strip() or self.settings["subject"]
            grade = input(f"Grade level (default: {self.settings['grade_level']}): ").strip()
            grade_level = int(grade) if grade.isdigit() else self.settings["grade_level"]
            
            # Add to index
            doc_id = self.indexer.add_document(
                content,
                metadata={
                    "subject": subject,
                    "grade_level": grade_level,
                    "type": Path(path).suffix[1:]  # File extension without dot
                }
            )
            
            print(f"\nDocument added successfully (ID: {doc_id})")
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            print("Failed to add document. Please check the logs.")

    def chat(self):
        """Start a chat session."""
        print("\nChat Session Started")
        print("Type 'q' to quit, 'c' to configure settings")
        
        while True:
            try:
                # Get input
                prompt = input("\nYou: ").strip()
                if not prompt:
                    continue
                    
                if prompt.lower() == 'q':
                    break
                elif prompt.lower() == 'c':
                    self.configure_settings()
                    continue
                
                # Get relevant context
                context = []
                try:
                    results = self.indexer.search(
                        prompt,
                        filters={"subject": self.settings["subject"]}
                    )
                    context = [r["content"] for r in results]
                except Exception as e:
                    logger.warning(f"Failed to search knowledge base: {e}")
                
                # Generate response
                response = self.llm.generate_response(
                    prompt,
                    context=context,
                    metadata=self.settings
                )
                
                # Evaluate response
                try:
                    metrics = self.evaluator.evaluate_response(
                        response,
                        prompt,
                        self.settings["grade_level"],
                        context
                    )
                    
                    # Display response and metrics
                    print(f"\nAssistant: {response}\n")
                    print("Response Evaluation:")
                    print(f"Clarity: {metrics.clarity:.2f}")
                    print(f"Engagement: {metrics.engagement:.2f}")
                    print(f"Pedagogical Approach: {metrics.pedagogical_approach:.2f}")
                    print(f"Emotional Support: {metrics.emotional_support:.2f}")
                    print(f"Content Accuracy: {metrics.content_accuracy:.2f}")
                    print(f"Age Appropriateness: {metrics.age_appropriateness:.2f}")
                    
                    if metrics.feedback:
                        print("\nFeedback:")
                        for item in metrics.feedback:
                            print(f"- {item}")
                            
                except Exception as e:
                    logger.warning(f"Failed to evaluate response: {e}")
                    print(f"\nAssistant: {response}")
                
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print("An error occurred. Please try again.")

    def run(self):
        """Run the CLI interface."""
        print("Welcome to the Teaching Assistant CLI")
        print("====================================")
        
        while True:
            print("\nOptions:")
            print("1. Start Chat")
            print("2. Configure Settings")
            print("3. Add Document")
            print("4. Quit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            try:
                if choice == '1':
                    self.chat()
                elif choice == '2':
                    self.configure_settings()
                elif choice == '3':
                    self.add_document()
                elif choice == '4':
                    break
                else:
                    print("Invalid choice. Please try again.")
                    
            except Exception as e:
                logger.error(f"Error handling choice: {e}")
                print("An error occurred. Please try again.")

def main():
    """Main entry point."""
    try:
        cli = TeachingAssistantCLI()
        cli.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 