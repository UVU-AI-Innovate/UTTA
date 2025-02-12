"""Process and manage knowledge base for the teacher training simulator."""
import os
from typing import Dict, List, Any
import json
import pandas as pd
import yaml
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llm_handler import LLMHandler

class KnowledgeProcessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name, cache_folder='model_cache')
        self.vector_dim = 384
        self.index = faiss.IndexFlatL2(self.vector_dim)
        self.documents = []
        self.llm = LLMHandler()
        self.knowledge_summary = {
            "total_documents": 0,
            "categories": {},
            "key_concepts": {},
            "examples": {}
        }

    def process_directory(self, directory: str = "knowledge_base"):
        """Process all files in the knowledge base directory."""
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                category = os.path.basename(root)
                self._process_file(file_path, category)

    def _process_file(self, file_path: Path, category: str):
        """Process a single file based on its extension."""
        content = None
        ext = file_path.suffix.lower()

        try:
            if ext == '.txt':
                content = self._process_text(file_path)
            elif ext == '.json':
                content = self._process_json(file_path)
            elif ext == '.csv':
                content = self._process_csv(file_path)
            elif ext == '.yaml' or ext == '.yml':
                content = self._process_yaml(file_path)
            
            if content:
                self._add_to_knowledge_base(content, category, file_path.name)
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _process_text(self, file_path: Path) -> List[str]:
        """Process text files into chunks."""
        with open(file_path, 'r') as f:
            text = f.read()
        return [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]

    def _process_json(self, file_path: Path) -> List[str]:
        """Process JSON files."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self._flatten_dict(data)

    def _process_csv(self, file_path: Path) -> List[str]:
        """Process CSV files."""
        df = pd.read_csv(file_path)
        return [
            f"{', '.join(f'{k}: {v}' for k, v in row.items() if pd.notna(v))}"
            for _, row in df.iterrows()
        ]

    def _process_yaml(self, file_path: Path) -> List[str]:
        """Process YAML files."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return self._flatten_dict(data)

    def _flatten_dict(self, d: Any, prefix: str = '') -> List[str]:
        """Flatten nested dictionaries/lists into strings."""
        results = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_prefix = f"{prefix}.{k}" if prefix else k
                results.extend(self._flatten_dict(v, new_prefix))
        elif isinstance(d, list):
            for item in d:
                results.extend(self._flatten_dict(item, prefix))
        else:
            results.append(f"{prefix}: {d}")
        return results

    def _add_to_knowledge_base(self, content: List[str], category: str, source: str):
        """Add processed content to vector store and update summary."""
        for text in content:
            # Create embedding
            embedding = self.embedding_model.encode([text])[0]
            self.index.add(np.array([embedding]).astype('float32'))
            
            # Store document
            self.documents.append({
                'text': text,
                'category': category,
                'source': source
            })
            
            # Update summary
            self.knowledge_summary["total_documents"] += 1
            self.knowledge_summary["categories"][category] = \
                self.knowledge_summary["categories"].get(category, 0) + 1
            
            # Extract key concepts (simple keyword extraction)
            words = text.lower().split()
            for word in words:
                if len(word) > 4:  # Simple filter for significant words
                    self.knowledge_summary["key_concepts"][word] = \
                        self.knowledge_summary["key_concepts"].get(word, 0) + 1
            
            # Store example if it's a good representative
            if len(text) > 50 and len(text) < 200:  # Good length for an example
                if category not in self.knowledge_summary["examples"]:
                    self.knowledge_summary["examples"][category] = text

    def get_summary(self) -> Dict:
        """Get a summary of the knowledge base."""
        return {
            "total_documents": self.knowledge_summary["total_documents"],
            "categories": dict(sorted(
                self.knowledge_summary["categories"].items(),
                key=lambda x: x[1],
                reverse=True
            )),
            "top_concepts": dict(sorted(
                self.knowledge_summary["key_concepts"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]),
            "examples": self.knowledge_summary["examples"]
        }

    def save(self, directory: str = 'processed_knowledge'):
        """Save processed knowledge base."""
        os.makedirs(directory, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, f'{directory}/vectors.index')
        
        # Save documents and summary
        with open(f'{directory}/knowledge_base.json', 'w') as f:
            json.dump({
                'documents': self.documents,
                'summary': self.knowledge_summary
            }, f, indent=2)

    def load(self, directory: str = 'processed_knowledge'):
        """Load processed knowledge base."""
        self.index = faiss.read_index(f'{directory}/vectors.index')
        
        with open(f'{directory}/knowledge_base.json', 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.knowledge_summary = data['summary'] 

    def get_relevant_responses(self, context: dict) -> list:
        """Get relevant response options for a given context."""
        try:
            # Query the vector store for similar scenarios
            query = f"subject:{context['subject']} behavior:{context['behavior']} learning_style:{context['learning_style']}"
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get similar responses from knowledge base
            D, I = self.index.search(
                np.array([query_embedding]).astype('float32'), 
                k=10  # Get top 10 matches
            )
            
            # Format responses with explanations
            responses = []
            for idx in I[0]:
                if idx < len(self.documents):  # Check if index is valid
                    doc = self.documents[idx]
                    responses.append({
                        "text": doc["text"],
                        "effectiveness": doc["effectiveness"],
                        "explanation": doc["explanation"]
                    })
            
            # If no valid responses found, return default responses
            if not responses:
                return self._get_default_responses(context)
            
            return responses
        
        except Exception as e:
            print(f"Error getting responses: {e}")
            return self._get_default_responses(context)

    def _get_default_responses(self, context: dict) -> list:
        """Generate default responses based on context."""
        subject = context["subject"]
        behavior = context["behavior"]
        learning_style = context["learning_style"]
        
        # Default response templates
        responses = []
        
        # Response 1: Direct instruction with learning style
        if learning_style == "visual":
            text = f"Let me draw this {subject} problem on the board so you can see how it works."
        elif learning_style == "auditory":
            text = f"Let's talk through this {subject} problem step by step."
        else:  # kinesthetic
            text = f"Let's use these manipulatives to solve this {subject} problem together."
        
        responses.append({
            "text": text,
            "effectiveness": 0.8,
            "explanation": f"Matches student's {learning_style} learning style and provides clear instruction"
        })
        
        # Response 2: Behavioral support
        if behavior == "attention":
            text = "Let's take a quick break and then try a different approach. Would that help?"
        else:  # frustration
            text = "I can see this is challenging. Let's break it down into smaller steps."
        
        responses.append({
            "text": text,
            "effectiveness": 0.85,
            "explanation": f"Addresses student's {behavior} behavior with appropriate support"
        })
        
        # Response 3: Engagement with scaffolding
        responses.append({
            "text": f"You're doing great with {subject}. Let me show you a trick that might help.",
            "effectiveness": 0.75,
            "explanation": "Provides encouragement and offers additional support"
        })
        
        # Response 4: Student-led approach
        responses.append({
            "text": f"Can you tell me which part of this {subject} problem is confusing? We can work on that part together.",
            "effectiveness": 0.9,
            "explanation": "Encourages student reflection and offers targeted support"
        })
        
        return responses 

    def evaluate_response(self, response: str, context: dict) -> dict:
        """Evaluate teacher response using semantic understanding."""
        try:
            # Create evaluation prompt with safe JSON example
            prompt = f"""
            You are an expert teacher evaluator. Evaluate this teacher's response to a second-grade student:

            Scenario Context:
            - Subject: {context['subject']}
            - Student's Learning Style: {context['student_context']['learning_style']}
            - Student's Current State: {context['behavioral_context']['type']}
            - Specific Challenge: {context['difficulty']}
            - Student's Behavior: {context['behavioral_context']['manifestation']}

            Teacher's Response: "{response}"

            Expected Teaching Strategies:
            1. Address the specific {context['subject']} challenge
            2. Use {context['student_context']['learning_style']} learning style approaches
            3. Manage {context['behavioral_context']['type']} behavior
            4. Build student confidence and engagement

            Evaluate and provide:
            1. Score (0.0-1.0) based on strategy alignment
            2. Specific strengths of the response
            3. Concrete suggestions for improvement
            4. Brief explanation of the evaluation

            Return your evaluation as a JSON object with these exact keys and example values:
            {{
                "score": 0.75,
                "strengths": ["Clear explanation", "Age-appropriate language"],
                "suggestions": ["Add more visual aids", "Include student participation"],
                "explanation": "Good basic approach that could be enhanced with more engagement"
            }}
            """
            
            # Get LLM evaluation
            evaluation = self.llm.analyze_teaching_response(prompt)
            return evaluation
            
        except Exception as e:
            print(f"Error evaluating response: {e}")
            return {
                "score": 0.5,
                "strengths": ["Basic supportive response"],
                "suggestions": ["Consider more specific strategies"],
                "explanation": "Unable to perform detailed evaluation"
            } 