"""
Pedagogical Knowledge Management System

This module manages the processing, storage, and retrieval of teaching knowledge
and educational resources. It provides a unified system for accessing and applying
pedagogical strategies, student behavior patterns, and subject-specific content.

Components:
1. Resource Processing - Handles various educational content formats
2. Knowledge Organization - Categorizes teaching strategies and materials
3. Semantic Search - Finds relevant pedagogical approaches
4. Context Matching - Aligns strategies with teaching scenarios
5. Knowledge Integration - Combines multiple teaching resources

The system supports various educational content formats while maintaining
pedagogical context and relationships between different teaching concepts.
"""

import os
from typing import List, Dict, Any, Optional
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path
import shutil
from PyPDF2 import PdfReader
from docx import Document
import markdown
import yaml
import csv
import chardet
from knowledge_store import KnowledgeStore
from dataclasses import dataclass

@dataclass
class TeachingStrategy:
    """
    Data class representing a teaching strategy.
    
    Attributes:
        name: Name or title of the strategy
        description: Detailed description of how to implement
        subjects: List of subjects where this strategy is applicable
        grade_levels: List of appropriate grade levels
        effectiveness_data: Optional historical effectiveness data
    """
    name: str
    description: str
    subjects: List[str]
    grade_levels: List[str]
    effectiveness_data: Optional[Dict[str, float]] = None

class PedagogicalKnowledgeManager:
    """
    Manages educational resources and teaching knowledge.
    
    This class handles:
    - Processing of educational materials
    - Organization of teaching strategies
    - Semantic search of pedagogical approaches
    - Integration of multiple knowledge sources
    - Contextual retrieval of teaching methods
    
    The system maintains relationships between different types of educational
    knowledge while providing efficient access to relevant teaching strategies.
    
    Attributes:
        embedding_model (SentenceTransformer): Neural model for semantic understanding
        knowledge_store (KnowledgeStore): Persistent storage for educational resources
        categories (dict): Organized teaching knowledge by domain
        vector_dim (int): Dimension of semantic embeddings
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.vector_dim = 384  # Dimension for this model
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.documents = []
        self.categories = {
            "student_behavior": [],
            "teaching_strategies": [],
            "academic_content": [],
            "developmental_psychology": [],
            "classroom_management": []
        }
        self.knowledge_base_dir = Path("knowledge_base")
        self.processed_dir = Path("processed_knowledge")
        self.default_dir = Path("default_knowledge")
        self.knowledge_store = KnowledgeStore()
        
        # Create directories if they don't exist
        self.knowledge_base_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)

    def extract_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract text from various file formats."""
        documents = []
        file_type = file_path.suffix.lower()

        try:
            if file_type == '.pdf':
                # Handle PDF files
                reader = PdfReader(file_path)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        documents.append({
                            'content': text,
                            'metadata': {
                                'source': file_path.name,
                                'page': i + 1,
                                'type': 'pdf'
                            }
                        })

            elif file_type == '.docx':
                # Handle Word documents
                doc = Document(file_path)
                for i, para in enumerate(doc.paragraphs):
                    if para.text.strip():
                        documents.append({
                            'content': para.text,
                            'metadata': {
                                'source': file_path.name,
                                'paragraph': i + 1,
                                'type': 'docx'
                            }
                        })

            elif file_type == '.txt':
                # Handle text files with encoding detection
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding']
                
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                    sections = text.split('\n\n')
                    for i, section in enumerate(sections):
                        if section.strip():
                            documents.append({
                                'content': section,
                                'metadata': {
                                    'source': file_path.name,
                                    'section': i + 1,
                                    'type': 'txt'
                                }
                            })

            elif file_type == '.md':
                # Handle Markdown files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    html = markdown.markdown(text)
                    # Split by headers or paragraphs
                    sections = html.split('<h')
                    for i, section in enumerate(sections):
                        if section.strip():
                            documents.append({
                                'content': section,
                                'metadata': {
                                    'source': file_path.name,
                                    'section': i,
                                    'type': 'markdown'
                                }
                            })

            elif file_type in ['.json', '.yaml', '.yml']:
                # Handle structured data files
                if file_type == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                
                # Recursively process structured data
                self._process_structured_data(data, documents, file_path.name)

            elif file_type == '.csv':
                # Handle CSV files
                df = pd.read_csv(file_path)
                for i, row in df.iterrows():
                    documents.append({
                        'content': ' '.join(str(v) for v in row.values),
                        'metadata': {
                            'source': file_path.name,
                            'row': i + 1,
                            'type': 'csv',
                            'columns': list(row.index)
                        }
                    })

            else:
                print(f"Unsupported file type: {file_type}")
                return []

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

        return documents

    def _process_structured_data(self, data: Any, documents: List[Dict], source: str, path: str = ""):
        """Recursively process structured data (JSON/YAML)."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, (str, int, float)):
                    documents.append({
                        'content': f"{key}: {value}",
                        'metadata': {
                            'source': source,
                            'path': current_path,
                            'type': 'structured'
                        }
                    })
                else:
                    self._process_structured_data(value, documents, source, current_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                if isinstance(item, (str, int, float)):
                    documents.append({
                        'content': str(item),
                        'metadata': {
                            'source': source,
                            'path': current_path,
                            'type': 'structured'
                        }
                    })
                else:
                    self._process_structured_data(item, documents, source, current_path)

    def add_to_knowledge_base(self, file_path: str):
        """Add any file to the knowledge base."""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return

        print(f"\nProcessing {file_path.name}...")
        
        # Extract text from file
        documents = self.extract_text(file_path)
        if not documents:
            print("No content extracted from file")
            return

        # Add to knowledge store
        for doc in documents:
            embedding = self.embedding_model.encode([doc['content']])[0]
            self.knowledge_store.add_document(
                content=doc['content'],
                source=doc['metadata']['source'],
                chunk_type=doc['metadata']['type'],
                metadata=doc['metadata'],
                embedding=embedding
            )

        print(f"✓ Added {len(documents)} chunks to knowledge base")

    def add_educational_resource(self, file_path: str, category: str):
        """Process and add educational resource files."""
        _, ext = os.path.splitext(file_path)
        
        if ext == '.txt':
            with open(file_path, 'r') as f:
                content = f.read()
                sections = content.split('\n\n')
                for section in sections:
                    self.add_to_knowledge_base(file_path)
                    self.categories[category].append(section)
        
        elif ext == '.csv':
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                text = ' '.join(str(v) for v in row.values)
                self.add_to_knowledge_base(file_path)
                self.categories[category].append(text)
        
        elif ext == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
                self._process_json_data(data, category, file_path)

    def _process_json_data(self, data: Any, category: str, source: str):
        """Recursively process JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    text = f"{key}: {value}"
                    self.add_to_knowledge_base(source)
                    self.categories[category].append(text)
                else:
                    self._process_json_data(value, category, source)
        elif isinstance(data, list):
            for item in data:
                self._process_json_data(item, category, source)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for relevant knowledge."""
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.knowledge_store.search(query_embedding, k)
        
        return [{
            'text': r['content'],
            'metadata': r['metadata'],
            'relevance': r['similarity']
        } for r in results]

    def get_teaching_context(self, scenario: Dict) -> Dict[str, List[str]]:
        """Get relevant teaching context for a scenario."""
        context = {
            "student_behavior": [],
            "teaching_strategies": [],
            "academic_content": []
        }
        
        # Search for relevant behavior information
        behavior_query = f"{scenario['behavioral_context']['type']} behavior in {scenario['subject']} class"
        behavior_results = self.search(behavior_query)
        context["student_behavior"] = [r['text'] for r in behavior_results]
        
        # Search for teaching strategies
        strategy_query = f"teaching strategies for {scenario['difficulty']} in {scenario['subject']}"
        strategy_results = self.search(strategy_query)
        context["teaching_strategies"] = [r['text'] for r in strategy_results]
        
        # Search for academic content
        content_query = f"{scenario['subject']} {scenario['difficulty']} second grade"
        content_results = self.search(content_query)
        context["academic_content"] = [r['text'] for r in content_results]
        
        return context

    def evaluate_response(self, response: str, scenario: Dict) -> Dict:
        """Evaluate teacher's response using knowledge base."""
        # Search for relevant teaching strategies
        relevant_strategies = self.search(
            f"effective teaching strategies for {scenario['behavioral_context']['type']} "
            f"in {scenario['subject']} {scenario['difficulty']}"
        )
        
        # Search for behavioral interventions
        relevant_interventions = self.search(
            f"interventions for {scenario['behavioral_context']['manifestation']} "
            f"in second grade"
        )
        
        # Calculate scores
        strategy_score = self._calculate_strategy_score(response, relevant_strategies)
        intervention_score = self._calculate_intervention_score(response, relevant_interventions)
        
        return {
            "strategy_score": strategy_score,
            "intervention_score": intervention_score,
            "total_score": (strategy_score + intervention_score) / 2,
            "relevant_strategies": [s['text'] for s in relevant_strategies],
            "relevant_interventions": [i['text'] for i in relevant_interventions]
        }

    def _calculate_strategy_score(self, response: str, strategies: List[Dict]) -> float:
        """Calculate how well the response uses recommended strategies."""
        score = 0.0
        response_embedding = self.embedding_model.encode([response])[0]
        
        for strategy in strategies:
            strategy_embedding = self.embedding_model.encode([strategy['text']])[0]
            similarity = float(faiss.pairwise_distances(
                response_embedding.reshape(1, -1),
                strategy_embedding.reshape(1, -1)
            ))
            score += similarity * strategy['relevance']
        
        return min(1.0, score / len(strategies) if strategies else 0.0)

    def _calculate_intervention_score(self, response: str, interventions: List[Dict]) -> float:
        """Calculate how well the response implements interventions."""
        score = 0.0
        response_embedding = self.embedding_model.encode([response])[0]
        
        for intervention in interventions:
            intervention_embedding = self.embedding_model.encode([intervention['text']])[0]
            similarity = float(faiss.pairwise_distances(
                response_embedding.reshape(1, -1),
                intervention_embedding.reshape(1, -1)
            ))
            score += similarity * intervention['relevance']
        
        return min(1.0, score / len(interventions) if interventions else 0.0)

    def save(self, directory: str = 'data/knowledge_base'):
        """Save the knowledge base."""
        os.makedirs(directory, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, f'{directory}/vectors.index')
        
        # Save documents and categories
        with open(f'{directory}/documents.json', 'w') as f:
            json.dump({
                'documents': self.documents,
                'categories': self.categories
            }, f)

    def load(self, directory: str = 'data/knowledge_base'):
        """Load the knowledge base."""
        # Load index
        self.index = faiss.read_index(f'{directory}/vectors.index')
        
        # Load documents and categories
        with open(f'{directory}/documents.json', 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.categories = data['categories']

    def load_default_knowledge(self):
        """Load default knowledge base files."""
        default_dir = "default_knowledge"
        
        try:
            # Load teaching strategies
            self.add_educational_resource(
                f"{default_dir}/teaching_strategies.txt",
                "teaching_strategies"
            )
            
            # Load student behaviors
            self.add_educational_resource(
                f"{default_dir}/student_behaviors.json",
                "student_behavior"
            )
            
            # Load academic content
            self.add_educational_resource(
                f"{default_dir}/math_strategies.csv",
                "academic_content"
            )
            
            print("Default knowledge base loaded successfully")
        except Exception as e:
            print(f"Error loading default knowledge: {e}")

    def get_stats(self) -> dict:
        """Get statistics about the knowledge base."""
        return {
            category: len(items) 
            for category, items in self.categories.items()
            if items  # Only show categories with items
        }

    def is_loaded(self) -> bool:
        """Check if knowledge base has content."""
        return any(len(items) > 0 for items in self.categories.values())

    def get_category_sample(self, category: str, n: int = 3) -> list:
        """Get sample entries from a category."""
        items = self.categories.get(category, [])
        return items[:n] if items else []

    def validate_knowledge_file(self, file_path: str) -> bool:
        """Validate a knowledge file's format."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if file_path.suffix == '.json':
                with open(file_path) as f:
                    data = json.load(f)
                    # Validate JSON structure
                    if not isinstance(data, dict):
                        raise ValueError("JSON must have a root dictionary")
                    for category, content in data.items():
                        if not isinstance(content, dict):
                            raise ValueError(f"Category '{category}' must contain a dictionary")
                        for subcategory, items in content.items():
                            if not isinstance(items, dict):
                                raise ValueError(f"Subcategory '{subcategory}' must contain a dictionary")
                            if 'strategies' not in items:
                                raise ValueError(f"Missing 'strategies' in {subcategory}")
                            if not isinstance(items['strategies'], list):
                                raise ValueError(f"'strategies' must be a list in {subcategory}")
                return True

            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
                required_columns = {'category', 'strategy'}
                if not required_columns.issubset(df.columns):
                    raise ValueError(f"CSV must contain columns: {required_columns}")
                return True

            elif file_path.suffix == '.txt':
                with open(file_path) as f:
                    content = f.read()
                    sections = content.split('\n\n')
                    if not sections:
                        raise ValueError("TXT file must contain sections separated by double newlines")
                    for section in sections:
                        if not section.strip():
                            continue
                        if 'Strategy:' not in section:
                            raise ValueError("Each section must contain 'Strategy:'")
                return True

            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")

        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False

    def add_knowledge_file(self, file_path: str, category: str):
        """Add a new knowledge file after validation."""
        if not self.validate_knowledge_file(file_path):
            print("File validation failed. Please check the format.")
            return

        source_path = Path(file_path)
        target_name = f"{category}_{source_path.stem}{source_path.suffix}"
        target_path = self.knowledge_base_dir / target_name

        # Copy file to knowledge base
        shutil.copy2(source_path, target_path)
        print(f"Added {target_name} to knowledge base")

        # Process the file based on its type
        self.add_educational_resource(str(target_path), category)
        print(f"Processed {target_name} into knowledge base")
    
    def process_knowledge_files(self):
        """Process all knowledge files into vector embeddings."""
        # This would be implemented in knowledge_processor.py 

    def process_all_files(self):
        """Process all files in the knowledge base directory."""
        print("\nProcessing knowledge base files...")
        
        # Clear existing knowledge
        self.knowledge_store.clear()
        
        # Process each file
        for file_path in self.knowledge_base_dir.glob('*.*'):
            if file_path.name != 'knowledge.db':  # Skip the database file
                print(f"\nProcessing {file_path.name}...")
                self.add_to_knowledge_base(str(file_path))

    def get_teaching_strategies(
        self, 
        context: Dict[str, Any]
    ) -> List[TeachingStrategy]:
        """
        Retrieve relevant teaching strategies based on context.
        
        Args:
            context: Dictionary containing search criteria:
                - subject: Subject area
                - grade_level: Target grade level
                - student_needs: Specific student requirements
                - learning_objectives: Desired learning outcomes
                
        Returns:
            List of TeachingStrategy objects matching the criteria
            
        Example:
            strategies = manager.get_teaching_strategies({
                "subject": "mathematics",
                "grade_level": "3rd",
                "student_needs": ["visual_learner", "needs_scaffolding"]
            })
        """
        query = self._build_search_query(context)
        results = self.search(query)
        return [self._parse_strategy(result) for result in results]
        
    def add_teaching_strategy(
        self, 
        strategy: TeachingStrategy
    ) -> bool:
        """
        Add a new teaching strategy to the knowledge base.
        
        Args:
            strategy: TeachingStrategy object containing the new strategy
            
        Returns:
            Boolean indicating success of the operation
            
        Example:
            success = manager.add_teaching_strategy(TeachingStrategy(
                name="Visual Fraction Learning",
                description="Use manipulatives to teach fractions...",
                subjects=["mathematics"],
                grade_levels=["3rd", "4th"]
            ))
        """
        try:
            # Convert strategy to vector embedding
            vector = self.embedding_model.encode([strategy.description])[0]
            
            # Store in vector database
            self.knowledge_store.add_document(
                content=strategy.description,
                source="Teaching Strategy",
                chunk_type="strategy",
                metadata={
                    "name": strategy.name,
                    "description": strategy.description,
                    "subjects": strategy.subjects,
                    "grade_levels": strategy.grade_levels,
                    "effectiveness_data": strategy.effectiveness_data
                },
                embedding=vector
            )
            
            return True
        except Exception as e:
            print(f"Error adding strategy: {str(e)}")
            return False
            
    def update_effectiveness(
        self, 
        strategy_id: str, 
        effectiveness_data: Dict[str, float]
    ) -> None:
        """
        Update effectiveness metrics for a teaching strategy.
        
        Args:
            strategy_id: Unique identifier for the strategy
            effectiveness_data: Dictionary containing metrics:
                - student_engagement: Float between 0 and 1
                - learning_outcomes: Float between 0 and 1
                - teacher_feedback: Float between 0 and 1
                
        Example:
            manager.update_effectiveness("strategy_123", {
                "student_engagement": 0.85,
                "learning_outcomes": 0.78,
                "teacher_feedback": 0.92
            })
        """
        # This method is not implemented in the original code block or the new class
        # It's assumed to exist as it's called in the update_effectiveness method
        pass
        
    def _build_search_query(self, context: Dict[str, Any]) -> str:
        """
        Build a search query from context parameters.
        
        Args:
            context: Search context dictionary
            
        Returns:
            Formatted search query string
        """
        return json.dumps({
            "type": "teaching_strategy",
            "parameters": context
        })
        
    def _parse_strategy(self, result: Dict[str, Any]) -> TeachingStrategy:
        """
        Parse search result into TeachingStrategy object.
        
        Args:
            result: Raw search result dictionary
            
        Returns:
            TeachingStrategy object
        """
        return TeachingStrategy(
            name=result["name"],
            description=result["description"],
            subjects=result["subjects"],
            grade_levels=result["grade_levels"],
            effectiveness_data=result.get("effectiveness_data")
        ) 