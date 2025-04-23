from typing import List, Tuple, Union, Dict, Any
from langchain_experimental.graph_transformers import LLMGraphTransformer
from src.model.model_catalogue import ModelType
from src.model.wrappers import ChatModelWrapper
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import networkx as nx
import spacy
import re

# It's a simple implementation, but it immediately plugs any custom model into the LLMGraphTransformer
class ModelKGTransformer(LLMGraphTransformer):
    def __init__(self, model: Union[ChatModelWrapper, ModelType], allowed_nodes: List[str] = [], allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = []):
        if isinstance(model, ModelType):    
            super().__init__(llm=ChatModelWrapper(model).model, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships)
        else:
            super().__init__(llm=model.model, allowed_nodes=allowed_nodes, allowed_relationships=allowed_relationships)

class BERTKGTransformer:
    def __init__(self, 
                 model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
                 allowed_nodes: List[str] = None,
                 allowed_relationships: Union[List[str], List[Tuple[str, str, str]]] = None):
        """
        Initialize BERT-based knowledge graph transformer.
        
        Args:
            model_name: Name of the BERT model to use for NER (Named Entity Recognition)
            allowed_nodes: List of allowed entity types (e.g., ['PERSON', 'ORG', 'LOC'])
            allowed_relationships: List of allowed relationship types or tuples of (subject_type, relationship, object_type)
        """
        # Loads the BERT model for NER
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        
        # Load spaCy for dependency parsing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Set allowed nodes and relationships
        self.allowed_nodes = set(allowed_nodes) if allowed_nodes else None
        self.allowed_relationships = set(allowed_relationships) if allowed_relationships else None
        
    def _is_allowed_node(self, entity_type: str) -> bool:
        """Check if an entity type is allowed"""
        return self.allowed_nodes is None or entity_type in self.allowed_nodes
    
    def _is_allowed_relationship(self, subj_type: str, rel: str, obj_type: str) -> bool:
        """Check if a relationship is allowed"""
        if self.allowed_relationships is None:
            return True
            
        # Check if relationship is allowed as a simple string
        if rel in self.allowed_relationships:
            return True
            
        # Check if relationship is allowed as a tuple
        for allowed_rel in self.allowed_relationships:
            if isinstance(allowed_rel, tuple) and len(allowed_rel) == 3:
                subj_allowed, rel_allowed, obj_allowed = allowed_rel
                if (subj_allowed == subj_type and 
                    rel_allowed == rel and 
                    obj_allowed == obj_type):
                    return True
        return False
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using BERT.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of dictionaries containing entity information
        """
        # Get NER predictions
        ner_results = self.ner_pipeline(text)
        
        # Group consecutive tokens with same entity type
        entities = []
        current_entity = {}
        
        for token in ner_results:
            if token['entity'].startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token['word'],
                    'type': token['entity'][2:],
                    'start': token['start'],
                    'end': token['end']
                }
            elif token['entity'].startswith('I-') and current_entity:
                current_entity['text'] += ' ' + token['word']
                current_entity['end'] = token['end']
        
        if current_entity:
            entities.append(current_entity)
            
        # Filter entities based on allowed nodes
        if self.allowed_nodes:
            entities = [e for e in entities if self._is_allowed_node(e['type'])]
            
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities using dependency parsing.
        
        Args:
            text: Input text
            entities: List of extracted entities
            
        Returns:
            List of (subject, relationship, object) tuples
        """
        # Use regex to find potential relationships
        relationships = []
        
        # Simple pattern to find subject-verb-object relationships
        pattern = r'(\w+)\s+(\w+)\s+(\w+)'
        matches = re.finditer(pattern, text)
        
        for match in matches:
            subj, rel, obj = match.groups()
            
            # Find corresponding entities
            subj_entity = next((e for e in entities if subj in e['text']), None)
            obj_entity = next((e for e in entities if obj in e['text']), None)
            
            if subj_entity and obj_entity:
                # Check if relationship is allowed
                if self._is_allowed_relationship(subj_entity['type'], rel, obj_entity['type']):
                    relationships.append((
                        subj_entity['text'],
                        rel,
                        obj_entity['text']
                    ))
        
        return relationships
    
    def process_text(self, text: str) -> nx.DiGraph:
        """
        Process text to create a knowledge graph.
        
        Args:
            text: Input text to process
            
        Returns:
            NetworkX directed graph representing the knowledge graph
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Add entities as nodes
        for entity in entities:
            self.graph.add_node(
                entity['text'],
                type=entity['type'],
                start=entity['start'],
                end=entity['end']
            )
        
        # Extract and add relationships as edges
        relationships = self.extract_relationships(text, entities)
        for subj, rel, obj in relationships:
            self.graph.add_edge(subj, obj, relationship=rel)
        
        return self.graph
    
    def get_graph(self) -> nx.DiGraph:
        """Return the current knowledge graph"""
        return self.graph
    
    def clear_graph(self):
        """Clear the current knowledge graph"""
        self.graph = nx.DiGraph()
