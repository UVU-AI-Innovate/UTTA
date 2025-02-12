"""Terminal interface for the teacher training simulator."""
from ai_agent import AIAgent
from termcolor import colored
import os
from datetime import datetime
from typing import Dict
from pathlib import Path
import json
from textwrap import dedent
from tqdm import tqdm
import time
import sys

class TerminalInterface:
    def __init__(self):
        """Initialize the terminal interface."""
        self.chatbot = AIAgent()
        self.current_scenario = None
        self.session_start = datetime.now()
        self.interaction_history = []
        self.knowledge_base_path = Path("knowledge_base")
        
        # Ensure knowledge base directory exists
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # Process knowledge base on startup
        self.process_knowledge_base()

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')

    def display_header(self):
        """Display the application header."""
        print("\n" + "="*70)
        print(colored("🎓 Teacher Training Simulator", "cyan", attrs=["bold"]).center(70))
        duration = datetime.now() - self.session_start
        print(colored(f"Session Time: {duration.seconds//60}m {duration.seconds%60}s", "cyan").center(70))
        print("="*70 + "\n")

    def display_scenario(self, scenario: Dict):
        """Display the current teaching scenario."""
        print(colored("\n📚 Current Teaching Scenario", "blue", attrs=["bold"]))
        print("─" * 50)

        # Context
        print(colored("Context:", "blue"))
        print(f"• Subject: {scenario['subject'].title()}")
        print(f"• Time: {scenario['time_of_day'].replace('_', ' ').title()}")

        # Student Information
        print(colored("\nStudent:", "blue"))
        student = scenario["student_context"]
        print(f"• Learning Style: {student['learning_style'].title()}")
        print(f"• Current State: {student['peer_interactions']}")
        
        # Current Situation
        print(colored("\nSituation:", "blue"))
        behavior = scenario["behavioral_context"]
        print(f"• {behavior['manifestation'].capitalize()}")
        print(f"• Triggered by: {behavior['trigger']}")

        print("\n" + "─" * 50)

    def load_knowledge_base(self):
        """Load and process knowledge base files."""
        print(colored("\n📚 Loading Knowledge Base...", "yellow"))
        
        if not self.knowledge_base_path.exists():
            print(colored("Creating knowledge base directory...", "yellow"))
            self.knowledge_base_path.mkdir(exist_ok=True)
            self._create_default_files()
        
        self.knowledge_files = {
            "teaching_strategies": self.knowledge_base_path / "teaching_strategies.json",
            "student_behaviors": self.knowledge_base_path / "student_behaviors.json",
            "subject_content": self.knowledge_base_path / "subject_content.json"
        }
        
        # Load each knowledge file
        self.knowledge_base = {}
        for name, filepath in self.knowledge_files.items():
            if filepath.exists():
                with open(filepath, 'r') as f:
                    self.knowledge_base[name] = json.load(f)
                print(colored(f"✓ Loaded {name}", "green"))
            else:
                print(colored(f"! Missing {name}", "red"))

    def display_knowledge_base_info(self):
        """Display information about the knowledge base structure."""
        print(colored("\n📖 Knowledge Base Information", "blue", attrs=["bold"]))
        print("="*70)
        
        print(colored("\nDirectory Structure:", "cyan"))
        print(dedent("""
            knowledge_base/           # All knowledge files
            ├── documents/           # Original files
            │   ├── teaching/       # Teaching strategies
            │   ├── behavior/       # Student behaviors
            │   └── content/        # Subject content
            │
            └── processed/          # Processed data
                ├── knowledge.db    # Vector store
                └── metadata.json   # File information
        """))
        
        print(colored("\nSupported File Types:", "cyan"))
        print(dedent("""
            • PDF Documents (.pdf)
            • Word Documents (.docx)
            • Text Files (.txt)
            • Markdown Files (.md)
            • JSON Files (.json)
            • YAML Files (.yml, .yaml)
            • CSV Files (.csv)
        """))
        
        # Show current knowledge base stats
        self.summarize_knowledge_base()

    def summarize_knowledge_base(self):
        """Provide a summary of the loaded knowledge base."""
        print("\n" + "─"*70)
        
        # Get all documents from knowledge store
        all_docs = self.chatbot.knowledge_manager.knowledge_store.get_all_documents()
        
        # Organize documents by type
        doc_types = {}
        for doc in all_docs:
            doc_type = doc['metadata'].get('type', 'unknown')
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(doc)
        
        # Display summary
        print(colored("\n📚 Knowledge Base Summary:", "yellow"))
        for doc_type, docs in doc_types.items():
            print(f"\n• {doc_type.title()} Documents:")
            print(f"  - Total chunks: {len(docs)}")
            
            # Group by source
            sources = {}
            for doc in docs:
                source = doc['metadata'].get('source', 'unknown')
                if source not in sources:
                    sources[source] = 0
                sources[source] += 1
            
            for source, count in sources.items():
                print(f"  - {source}: {count} chunks")
        
        # Show sample content
        print(colored("\n📖 Sample Content:", "yellow"))
        samples = self.chatbot.knowledge_manager.search("teaching strategies", k=3)
        for i, sample in enumerate(samples, 1):
            print(f"\n{i}. From {sample['metadata'].get('source', 'unknown')}:")
            print(f"   {sample['text'][:100]}...")
            print(f"   Relevance: {sample['relevance']:.2f}")
        
        print("\n" + "─"*70)

    def display_knowledge_structure(self):
        """Display knowledge base structure and how to use it."""
        print(colored("\n📂 Knowledge Base Structure", "blue", attrs=["bold"]))
        print("="*70)
        
        print(colored("\nSupported File Types:", "cyan"))
        print(dedent("""
            • PDF Documents (.pdf)
            • Word Documents (.docx)
            • Text Files (.txt)
            • Markdown Files (.md)
            • JSON Files (.json)
            • YAML Files (.yml, .yaml)
            • CSV Files (.csv)
            • Structured Data Files
        """))
        
        print(colored("\nHow to Add Files:", "yellow"))
        print(dedent("""
            1. Place any supported file in the 'knowledge_base' directory
            2. Files are automatically processed into chunks
            3. Content is embedded and added to vector store
            4. Original files are preserved for reference
        """))
        
        print(colored("\nCurrent Knowledge Base:", "green"))
        self._show_knowledge_base_stats()
        
        # Show sample content
        if not list(self.knowledge_base_path.glob('*.*')):
            print("\nNo files found. Would you like to load sample content? (y/n)")
            if input().lower().strip() == 'y':
                self.load_sample_content()
                print("\nUpdated Knowledge Base:")
                self._show_knowledge_base_stats()

    def _show_knowledge_base_stats(self):
        """Show statistics about the current knowledge base."""
        if not self.knowledge_base_path.exists():
            print("No knowledge base directory found")
            return

        files = list(self.knowledge_base_path.glob('*.*'))
        if not files:
            print("No files in knowledge base")
            return

        print("\nFiles in knowledge base:")
        for file in files:
            docs = self.chatbot.knowledge_manager.knowledge_store.get_documents_by_source(file.name)
            content_types = {}
            for doc in docs:
                content_type = doc['chunk_type']
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            print(f"\n• {file.name}:")
            print(f"  Total extracted pieces: {len(docs)}")
            for content_type, count in content_types.items():
                if content_type == 'structured':
                    print(f"  - Teaching concepts: {count}")
                elif content_type == 'txt':
                    print(f"  - Paragraphs: {count}")
                elif content_type == 'csv':
                    print(f"  - Data rows: {count}")
                elif content_type == 'markdown':
                    print(f"  - Strategy sections: {count}")
                elif content_type == 'pdf':
                    print(f"  - Document sections: {count}")
                else:
                    print(f"  - {content_type.title()} sections: {count}")

    def display_menu(self):
        """Display the main menu."""
        print(colored("\n📋 Options:", "green"))
        options = [
            "Respond to student",
            "View current scenario",
            "View session history",
            "Start new scenario",
            "View knowledge base info",
            "View knowledge structure",
            "View knowledge impact",
            "Exit session"
        ]
        for i, option in enumerate(options, 1):
            print(colored(f"{i}.", "green"), option)
        return input(colored("\nChoose an option (1-8): ", "green")).strip()

    def display_evaluation(self, evaluation: Dict):
        """Display the evaluation results."""
        print("\n" + "─"*50)
        
        # Overall Score
        score = evaluation['score']
        score_color = "red" if score < 0.4 else "yellow" if score < 0.8 else "green"
        print(colored(f"Response Score: {score*100:.0f}%", score_color, attrs=["bold"]))

        # Strengths
        if evaluation['feedback']:
            print(colored("\nStrengths:", "green"))
            for strength in evaluation['feedback']:
                print(colored(f"✓ {strength}", "green"))

        # Suggestions
        if evaluation['suggestions']:
            print(colored("\nSuggestions:", "yellow"))
            for suggestion in evaluation['suggestions']:
                print(colored(f"→ {suggestion}", "yellow"))

        # Student Reaction
        print(colored("\nStudent:", "cyan"))
        print(colored(evaluation['student_reaction'], "cyan"))
        
        print("─"*50)

    def handle_response(self):
        """Handle teacher's response to the scenario."""
        print(colored("\nWhat would you say to the student?", "cyan"))
        response = input("> ").strip()
        
        if response.lower() in ['exit', 'quit']:
            return False
            
        evaluation = self.chatbot.evaluate_response(response, self.current_scenario)
        self.display_evaluation(evaluation)
        
        # Store interaction
        self.interaction_history.append({
            'timestamp': datetime.now(),
            'response': response,
            'evaluation': evaluation
        })
        
        return True

    def display_history(self):
        """Display the session history."""
        if not self.interaction_history:
            print(colored("\nNo interactions yet.", "yellow"))
            return

        print(colored("\n📖 Session History", "blue", attrs=["bold"]))
        print("─" * 50)
        
        for i, interaction in enumerate(self.interaction_history, 1):
            print(colored(f"\nInteraction {i}:", "blue"))
            print(f"Teacher: {interaction['response']}")
            print(colored(f"Score: {interaction['evaluation']['score']*100:.0f}%", "cyan"))
            print(colored(f"Student: {interaction['evaluation']['student_reaction']}", "cyan"))
            print("─" * 30)

    def run(self):
        """Run the terminal interface."""
        self.clear_screen()
        self.display_header()
        
        # Show initial knowledge base summary
        self.summarize_knowledge_base()
        
        # Generate initial scenario
        self.current_scenario = self.chatbot.generate_scenario()
        self.display_scenario(self.current_scenario)
        
        while True:
            choice = self.display_menu()
            
            if choice == "1":
                if not self.handle_response():
                    break
            elif choice == "2":
                self.display_scenario(self.current_scenario)
            elif choice == "3":
                self.display_history()
            elif choice == "4":
                self.current_scenario = self.chatbot.generate_scenario()
                self.clear_screen()
                self.display_header()
                self.display_scenario(self.current_scenario)
            elif choice == "5":
                self.display_knowledge_base_info()
            elif choice == "6":
                self.display_knowledge_structure()
            elif choice == "7":
                self.display_knowledge_impact()
            elif choice == "8":
                self.display_session_summary()
                break
            else:
                print(colored("\n❌ Invalid choice. Please choose 1-8.", "red"))

    def display_session_summary(self):
        """Display the session summary."""
        print("\n" + "="*70)
        print(colored("📊 Session Summary", "magenta", attrs=["bold"]))
        print("="*70)
        
        # Calculate statistics
        total_interactions = len(self.interaction_history)
        if total_interactions > 0:
            avg_score = sum(i['evaluation']['score'] for i in self.interaction_history) / total_interactions
            best_score = max(i['evaluation']['score'] for i in self.interaction_history)
            
            print(f"\nTotal Interactions: {total_interactions}")
            print(colored(f"Average Score: {avg_score*100:.1f}%", "cyan"))
            print(colored(f"Best Score: {best_score*100:.1f}%", "green"))
        
        print(colored("\nThank you for using the Teacher Training Simulator! 👋", "cyan"))
        print("\n" + "="*70)

    def load_sample_content(self):
        """Load sample content in different formats."""
        samples = {
            "classroom_management.pdf": """
            Classroom Management Strategies
            1. Establishing Routines...
            """,
            "math_activities.docx": """
            Second Grade Math Activities
            Activity 1: Number Line Jumps...
            """,
            "reading_strategies.md": """
            # Reading Strategies for Second Grade
            ## Decoding Skills...
            """,
            "student_profiles.yml": """
            learning_styles:
              visual:
                characteristics:...
            """,
            "behavior_interventions.csv": """
            behavior,trigger,intervention,effectiveness
            fidgeting,long sitting,movement break,high
            """
        }
        
        for filename, content in samples.items():
            file_path = self.knowledge_base_path / filename
            with open(file_path, 'w') as f:
                f.write(content)
            print(f"Created sample file: {filename}")
            
            # Process the file
            self.chatbot.knowledge_manager.add_to_knowledge_base(str(file_path))

    def _analyze_knowledge_impact(self):
        """Analyze how the knowledge base improves the chatbot's capabilities."""
        all_docs = self.chatbot.knowledge_manager.knowledge_store.get_all_documents()
        
        # Categorize knowledge by educational aspects
        knowledge_categories = {
            'teaching_strategies': [],
            'student_behaviors': [],
            'classroom_management': [],
            'subject_content': [],
            'learning_styles': [],
            'interventions': []
        }
        
        # Analyze each document
        for doc in all_docs:
            content = doc['content'].lower()
            if any(word in content for word in ['teach', 'strategy', 'method']):
                knowledge_categories['teaching_strategies'].append(doc)
            if any(word in content for word in ['behavior', 'reaction', 'respond']):
                knowledge_categories['student_behaviors'].append(doc)
            if any(word in content for word in ['manage', 'routine', 'classroom']):
                knowledge_categories['classroom_management'].append(doc)
            if any(word in content for word in ['math', 'reading', 'subject']):
                knowledge_categories['subject_content'].append(doc)
            if any(word in content for word in ['visual', 'auditory', 'kinesthetic']):
                knowledge_categories['learning_styles'].append(doc)
            if any(word in content for word in ['intervention', 'support', 'help']):
                knowledge_categories['interventions'].append(doc)

        print(colored("\n🧠 Knowledge Impact Analysis", "blue", attrs=["bold"]))
        print("="*70)
        
        # Show knowledge distribution
        print(colored("\nKnowledge Distribution:", "yellow"))
        for category, docs in knowledge_categories.items():
            if docs:
                print(f"\n• {category.replace('_', ' ').title()}:")
                print(f"  Total entries: {len(docs)}")
                
                # Show key concepts
                concepts = set()
                for doc in docs[:5]:  # Sample from first 5 docs
                    words = doc['content'].lower().split()
                    concepts.update([word for word in words if len(word) > 4])
                print(f"  Key concepts: {', '.join(list(concepts)[:5])}")
                
                # Show a sample entry
                if docs:
                    print(f"  Sample: {docs[0]['content'][:100]}...")

        print(colored("\n🎯 Simulation Capabilities:", "yellow"))
        print("\nThis knowledge enables the chatbot to:")
        
        capabilities = [
            ("Scenario Generation", [
                "• Create realistic classroom situations",
                "• Vary student behaviors and triggers",
                "• Incorporate subject-specific challenges"
            ]),
            ("Student Responses", [
                "• Generate contextual student reactions",
                "• Simulate different learning styles",
                "• Show realistic behavioral patterns"
            ]),
            ("Response Evaluation", [
                "• Assess teaching strategy effectiveness",
                "• Consider timing and context",
                "• Provide evidence-based feedback"
            ])
        ]
        
        for capability, details in capabilities:
            print(f"\n{capability}:")
            for detail in details:
                print(detail)
        
        # Show coverage analysis
        print(colored("\n📊 Knowledge Coverage:", "yellow"))
        total_docs = len(all_docs)
        coverage = {
            'Time Periods': ['morning', 'afternoon', 'after lunch'],
            'Learning Styles': ['visual', 'auditory', 'kinesthetic'],
            'Subjects': ['math', 'reading', 'writing'],
            'Behaviors': ['attention', 'frustration', 'engagement']
        }
        
        for aspect, keywords in coverage.items():
            covered = sum(1 for doc in all_docs 
                         if any(keyword in doc['content'].lower() 
                         for keyword in keywords))
            percentage = (covered / total_docs) * 100
            print(f"\n• {aspect}:")
            print(f"  Coverage: {percentage:.1f}%")
            print(f"  Keywords: {', '.join(keywords)}")

    def display_knowledge_impact(self):
        """Display detailed analysis of knowledge base impact."""
        print(colored("\n🧠 Knowledge Base Impact Analysis", "blue", attrs=["bold"]))
        print("="*70)
        
        all_docs = self.chatbot.knowledge_manager.knowledge_store.get_all_documents()
        
        # Categorize knowledge
        knowledge_categories = {
            'teaching_strategies': [],
            'student_behaviors': [],
            'classroom_management': [],
            'subject_content': [],
            'learning_styles': [],
            'interventions': []
        }
        
        # Analyze each document
        for doc in all_docs:
            content = doc['content'].lower()
            if any(word in content for word in ['teach', 'strategy', 'method']):
                knowledge_categories['teaching_strategies'].append(doc)
            if any(word in content for word in ['behavior', 'reaction', 'respond']):
                knowledge_categories['student_behaviors'].append(doc)
            if any(word in content for word in ['manage', 'routine', 'classroom']):
                knowledge_categories['classroom_management'].append(doc)
            if any(word in content for word in ['math', 'reading', 'subject']):
                knowledge_categories['subject_content'].append(doc)
            if any(word in content for word in ['visual', 'auditory', 'kinesthetic']):
                knowledge_categories['learning_styles'].append(doc)
            if any(word in content for word in ['intervention', 'support', 'help']):
                knowledge_categories['interventions'].append(doc)

        # 1. Knowledge Distribution
        print(colored("\n📊 Knowledge Distribution:", "yellow"))
        for category, docs in knowledge_categories.items():
            if docs:
                print(f"\n• {category.replace('_', ' ').title()}:")
                print(f"  Total entries: {len(docs)}")
                
                # Extract and show key concepts
                concepts = set()
                for doc in docs[:5]:
                    words = doc['content'].lower().split()
                    concepts.update([word for word in words if len(word) > 4])
                print(f"  Key concepts: {', '.join(sorted(list(concepts)[:5]))}") 
                
                # Show representative sample
                if docs:
                    print(f"  Sample: {docs[0]['content'][:100]}...")

        # 2. Simulation Capabilities
        print(colored("\n🎯 Enhanced Capabilities:", "yellow"))
        print("\nThis knowledge enables the chatbot to:")
        
        capabilities = [
            ("Scenario Generation", [
                "• Create realistic classroom situations based on {n} patterns",
                "• Generate {n} different student behavior types",
                "• Incorporate {n} subject-specific challenges"
            ]),
            ("Student Responses", [
                "• Generate contextual reactions using {n} behavior patterns",
                "• Simulate {n} different learning styles",
                "• Model {n} emotional states"
            ]),
            ("Response Evaluation", [
                "• Assess strategies across {n} teaching dimensions",
                "• Consider {n} contextual factors",
                "• Provide feedback based on {n} evidence-based practices"
            ])
        ]
        
        for capability, details in capabilities:
            print(f"\n{capability}:")
            for detail in details:
                relevant_count = len(all_docs) // 6  # Simplified count
                print(detail.format(n=relevant_count))

        # 3. Coverage Analysis
        print(colored("\n📈 Knowledge Coverage Analysis:", "yellow"))
        total_docs = len(all_docs)
        coverage = {
            'Time Periods': ['morning', 'afternoon', 'after lunch'],
            'Learning Styles': ['visual', 'auditory', 'kinesthetic'],
            'Subjects': ['math', 'reading', 'writing'],
            'Behaviors': ['attention', 'frustration', 'engagement'],
            'Teaching Methods': ['direct', 'group', 'individual'],
            'Assessment Types': ['formative', 'observation', 'feedback']
        }
        
        for aspect, keywords in coverage.items():
            covered = sum(1 for doc in all_docs 
                         if any(keyword in doc['content'].lower() 
                         for keyword in keywords))
            percentage = (covered / total_docs) * 100
            print(f"\n• {aspect}:")
            print(f"  Coverage: {percentage:.1f}%")
            print(f"  Keywords: {', '.join(keywords)}")
            
            # Show examples if coverage exists
            if covered > 0:
                examples = [doc['content'][:50] + "..." 
                           for doc in all_docs 
                           if any(keyword in doc['content'].lower() 
                           for keyword in keywords)][:2]
                print("  Examples:")
                for ex in examples:
                    print(f"   - {ex}")

        # 4. Integration Analysis
        print(colored("\n🔄 Knowledge Integration:", "yellow"))
        print("\nCross-domain Connections:")
        domains = {
            'Behavior-Teaching': ['classroom management', 'student engagement'],
            'Content-Method': ['learning activities', 'teaching strategies'],
            'Assessment-Support': ['feedback methods', 'intervention strategies']
        }
        
        for connection, keywords in domains.items():
            related_docs = [doc for doc in all_docs 
                           if any(keyword in doc['content'].lower() 
                           for keyword in keywords)]
            if related_docs:
                print(f"\n• {connection}:")
                print(f"  Connected concepts: {len(related_docs)}")
                print(f"  Sample integration: {related_docs[0]['content'][:100]}...")

    def process_knowledge_base(self):
        """Process all files in the knowledge base."""
        print(colored("\n📂 Processing Knowledge Base", "blue", attrs=["bold"]))
        print("="*70)
        
        print(colored("\n🔍 Step 1: Scanning Knowledge Base Directory", "yellow"))
        files = list(self.knowledge_base_path.glob('*.*'))
        print(f"Found {len(files)} files to process")
        
        print(colored("\n🔄 Step 2: Processing Files", "yellow"))
        for file in files:
            if file.name == 'knowledge.db':
                continue
            
            print(f"\nProcessing: {file.name}")
            print("─"*50)
            print(f"• Format: {file.suffix}")
            print(f"• Size: {file.stat().st_size / 1024:.1f} KB")
            
            try:
                # Process file
                print("\nExtraction:")
                docs = self.chatbot.knowledge_manager.extract_text(file)
                print(f"✓ Extracted {len(docs)} chunks")
                
                # Show chunking details
                chunk_types = {}
                for doc in docs:
                    chunk_type = doc['metadata'].get('type', 'unknown')
                    if chunk_type not in chunk_types:
                        chunk_types[chunk_type] = []
                    chunk_types[chunk_type].append(len(doc['content']))
                
                print("\nChunk Analysis:")
                for chunk_type, lengths in chunk_types.items():
                    avg_length = sum(lengths) / len(lengths)
                    print(f"• {chunk_type}:")
                    print(f"  - Count: {len(lengths)}")
                    print(f"  - Avg Length: {avg_length:.0f} chars")
                    print(f"  - Range: {min(lengths)} to {max(lengths)} chars")
                
                print("\nEmbedding Generation:")
                print("• Model: all-MiniLM-L6-v2")
                print("• Dimension: 384")
                print("• Batch Size: 32")
                
                # Add to knowledge store with progress bar
                with tqdm(total=len(docs), desc="Generating embeddings", unit="chunk") as pbar:
                    for i in range(0, len(docs), 32):
                        batch = docs[i:i+32]
                        # Simulate embedding generation time
                        time.sleep(0.1)  # Remove this in production
                        for doc in batch:
                            self.chatbot.knowledge_manager.knowledge_store.add_document(
                                content=doc['content'],
                                source=file.name,
                                chunk_type=doc['metadata']['type'],
                                metadata=doc['metadata'],
                                embedding=self.chatbot.knowledge_manager.embedding_model.encode([doc['content']])[0]
                            )
                        pbar.update(len(batch))
                print(f"✓ Added {len(docs)} embeddings to vector store")
                
            except Exception as e:
                print(colored(f"✗ Error: {str(e)}", "red"))
        
        print(colored("\n💾 Step 3: Storage Information", "yellow"))
        db_path = self.knowledge_base_path / "knowledge.db"
        print(f"• Database: {db_path}")
        print(f"• Size: {db_path.stat().st_size / 1024:.1f} KB")
        
        # Show vector store stats
        print("\nVector Store Stats:")
        all_docs = self.chatbot.knowledge_manager.knowledge_store.get_all_documents()
        total_size = sum(sys.getsizeof(doc['embedding']) for doc in all_docs)
        print(f"• Total Vectors: {len(all_docs)}")
        print(f"• Memory Usage: {total_size / 1024 / 1024:.1f} MB")
        
        print(colored("\n📚 Knowledge Base Impact Summary", "blue", attrs=["bold"]))
        self._analyze_knowledge_impact()

def main():
    """Run the terminal application."""
    app = TerminalInterface()
    try:
        app.run()
    except KeyboardInterrupt:
        print(colored("\n\nSession terminated by user. Goodbye! 👋", "cyan"))
    except Exception as e:
        print(colored(f"\n\nAn error occurred: {str(e)}", "red"))
        print("Please try restarting the application.")

if __name__ == "__main__":
    main() 