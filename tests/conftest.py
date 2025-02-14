import pytest
from unittest.mock import Mock
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def test_data_dir():
    """Return path to test data directory"""
    return os.path.join(os.path.dirname(__file__), "test_data")

@pytest.fixture(scope="session")
def mock_environment():
    """Set up test environment variables"""
    os.environ["TEST_MODEL_KEY"] = "test-key-123"
    yield
    del os.environ["TEST_MODEL_KEY"]

@pytest.fixture
def mock_response():
    """Create a mock response object"""
    class MockResponse:
        def __init__(self, text="", status_code=200):
            self.text = text
            self.status_code = status_code
            
        def json(self):
            return {"response": self.text}
            
    return MockResponse

@pytest.fixture
def sample_queries():
    """Return a list of sample test queries"""
    return [
        "What is machine learning?",
        "Explain neural networks",
        "Define deep learning"
    ]

@pytest.fixture
def sample_responses():
    """Return a list of sample responses"""
    return [
        "Machine learning is...",
        "Neural networks are...",
        "Deep learning refers to..."
    ]