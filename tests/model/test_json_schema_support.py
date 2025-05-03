"""
Tests for structured output support across different models.
"""
import pytest
from pydantic.v1 import BaseModel, Field
from typing import List
from src.model.model_catalogue import ModelCatalogue, ModelType, LocalModelType
from src.model.wrappers import ChatModelWrapper
from langchain.prompts import ChatPromptTemplate

class TestEntities(BaseModel):
    """
    Simple test model for structured output.
    """
    names: List[str] = Field(
        ...,
        description="List of names extracted from the text",
    )

@pytest.fixture
def model_catalogue():
    """Fixture to provide the model catalogue."""
    return ModelCatalogue()

@pytest.fixture
def model_type(model_catalogue):
    """Fixture to provide a model type for testing."""
    # Get the first model that supports JSON schema
    for model in ModelCatalogue.get_models_with_json_schema().values():
        return model
    pytest.skip("No models with JSON schema support available for testing")

def test_model_supports_structured_output(model_type: ModelType):
    """Test if a model supports structured output."""
    # Skip models over 100GB
    if isinstance(model_type, LocalModelType) and model_type.download_size and model_type.download_size > 100:
        pytest.skip(f"Model {model_type.argName} exceeds 100GB limit")
    
    # Initialize the model
    model_wrapper = ChatModelWrapper(model_type)
    model = model_wrapper.model
    
    # Create a prompt that should generate structured output
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract names from the following text. Return them in a structured format."),
        ("human", "John Smith and Jane Doe went to the park with their friend Alice Brown.")
    ])
    
    # Try to get structured output
    try:
        chain = prompt | model.with_structured_output(TestEntities)
        result = chain.invoke({})
        
        # Verify the output is structured correctly
        assert isinstance(result, TestEntities)
        assert len(result.names) > 0
        assert all(isinstance(name, str) for name in result.names)
        
    except Exception as e:
        pytest.fail(f"Model {model_type.argName} failed to produce structured output: {str(e)}") 