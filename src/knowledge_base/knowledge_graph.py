"""Knowledge graph for managing pedagogical concepts and relationships."""

import logging
from typing import Dict, List, Any
import networkx as nx
import json
from pathlib import Path

class KnowledgeGraph:
    """Manages a graph of pedagogical knowledge and relationships."""
    
    def __init__(self):
        """Initialize the knowledge graph."""
        self.graph = nx.DiGraph()
        self.concepts = {}
        self.relationships = []
        
        # Create storage directory
        self.storage_dir = Path("data/knowledge_base")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("Initialized KnowledgeGraph")
    
    def add_concept(self, concept_id: str, attributes: Dict[str, Any]) -> bool:
        """Add a concept to the knowledge graph.
        
        Args:
            concept_id: Unique identifier for the concept
            attributes: Dictionary of concept attributes
            
        Returns:
            bool: Success status
        """
        try:
            # Add to graph
            self.graph.add_node(concept_id, **attributes)
            
            # Store in concepts dictionary
            self.concepts[concept_id] = attributes
            
            logging.info(f"Added concept: {concept_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding concept: {e}")
            return False
    
    def add_relationship(self,
                        from_concept: str,
                        to_concept: str,
                        relationship_type: str,
                        attributes: Dict[str, Any] = None) -> bool:
        """Add a relationship between concepts.
        
        Args:
            from_concept: Source concept ID
            to_concept: Target concept ID
            relationship_type: Type of relationship
            attributes: Additional relationship attributes
            
        Returns:
            bool: Success status
        """
        try:
            if attributes is None:
                attributes = {}
            
            # Add relationship to graph
            self.graph.add_edge(
                from_concept,
                to_concept,
                relationship_type=relationship_type,
                **attributes
            )
            
            # Store relationship
            relationship = {
                "from": from_concept,
                "to": to_concept,
                "type": relationship_type,
                "attributes": attributes
            }
            self.relationships.append(relationship)
            
            logging.info(f"Added relationship: {from_concept} -> {to_concept}")
            return True
            
        except Exception as e:
            logging.error(f"Error adding relationship: {e}")
            return False
    
    def get_related_concepts(self,
                           concept_id: str,
                           relationship_type: str = None) -> List[Dict[str, Any]]:
        """Get concepts related to a given concept.
        
        Args:
            concept_id: The concept to find relationships for
            relationship_type: Optional filter for relationship type
            
        Returns:
            List of related concepts with their relationships
        """
        try:
            related = []
            
            # Get all neighbors
            for neighbor in self.graph.neighbors(concept_id):
                edge_data = self.graph.get_edge_data(concept_id, neighbor)
                
                # Check relationship type if specified
                if (relationship_type is None or
                    edge_data.get('relationship_type') == relationship_type):
                    
                    related.append({
                        "concept_id": neighbor,
                        "attributes": self.concepts.get(neighbor, {}),
                        "relationship": edge_data
                    })
            
            return related
            
        except Exception as e:
            logging.error(f"Error getting related concepts: {e}")
            return []
    
    def save(self, filename: str = "knowledge_graph.json") -> bool:
        """Save the knowledge graph to a file.
        
        Args:
            filename: Name of the file to save to
            
        Returns:
            bool: Success status
        """
        try:
            save_path = self.storage_dir / filename
            
            data = {
                "concepts": self.concepts,
                "relationships": self.relationships
            }
            
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logging.info(f"Saved knowledge graph to: {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving knowledge graph: {e}")
            return False
    
    def load(self, filename: str = "knowledge_graph.json") -> bool:
        """Load the knowledge graph from a file.
        
        Args:
            filename: Name of the file to load from
            
        Returns:
            bool: Success status
        """
        try:
            load_path = self.storage_dir / filename
            
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Clear existing data
            self.graph.clear()
            self.concepts.clear()
            self.relationships.clear()
            
            # Load concepts
            for concept_id, attributes in data["concepts"].items():
                self.add_concept(concept_id, attributes)
            
            # Load relationships
            for rel in data["relationships"]:
                self.add_relationship(
                    rel["from"],
                    rel["to"],
                    rel["type"],
                    rel.get("attributes", {})
                )
            
            logging.info(f"Loaded knowledge graph from: {load_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading knowledge graph: {e}")
            return False 