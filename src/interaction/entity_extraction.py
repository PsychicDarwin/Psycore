"""
Entity extraction module for RAG systems.
"""
from typing import List
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.model.wrappers import ChatModelWrapper
from src.model.model_catalogue import ModelType

class Entities(BaseModel):
    """
    Pydantic model for structured entity extraction.
    """
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that appear in the text",
    )

class EntityExtractor:
    """
    A class for extracting entities from text using LLMs.
    """
    
    def __init__(self, model_wrapper: ChatModelWrapper):
        """
        Initialize the EntityExtractor.
        
        Args:
            model_wrapper: ChatModelWrapper instance for the LLM
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        
        # Create the entity extraction prompt
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ])
        
        # Create the entity extraction chain
        self.entity_chain = self.prompt | self.model.with_structured_output(Entities)
    
    @classmethod
    def from_model_type(cls, model_type: ModelType) -> 'EntityExtractor':
        """
        Create an EntityExtractor from a ModelType.
        
        Args:
            model_type: ModelType instance
            
        Returns:
            EntityExtractor instance
        """
        from src.model.wrappers import ChatModelWrapper
        model_wrapper = ChatModelWrapper(model_type)
        return cls(model_wrapper)
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.
        
        Args:
            text: The text to extract entities from
            
        Returns:
            List of extracted entity names
        """
        result = self.entity_chain.invoke({"question": text})
        return result.names
