import pytest
from src.model.controller import ModelController
from unittest.mock import Mock, patch, MagicMock
import logging

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@pytest.fixture
def mock_llm():
    mock = Mock()
    mock.predict.return_value = "Mocked response"
    mock.apredict = MagicMock(return_value="Async mocked response")
    return mock

@pytest.fixture
def mock_config():
    return {
        "model_name": "test-model",
        "temperature": 0.7,
        "max_tokens": 100,
        "streaming": False
    }

@pytest.fixture
def controller(mock_llm, mock_config):
    with patch('src.model.controller.load_config') as mock_load_config:
        mock_load_config.return_value = mock_config
        return ModelController(llm=mock_llm)

class TestControllerMock:
    """Test suite for ModelController using mocks"""

    def test_initialization(self, controller, mock_config):
        """Test controller initialization"""
        assert controller.config == mock_config
        assert controller.llm is not None
        logger.info("Controller initialized successfully")

    def test_basic_query(self, controller):
        """Test basic query processing"""
        test_query = "What is AI?"
        expected_response = "Mocked response"
        
        result = controller.process_query(test_query)
        
        assert result == expected_response
        controller.llm.predict.assert_called_once()
        logger.debug(f"Query processed: {test_query}")

    @pytest.mark.asyncio
    async def test_async_query(self, controller):
        """Test async query processing"""
        test_query = "Async test query"
        expected_response = "Async mocked response"
        
        result = await controller.async_process_query(test_query)
        
        assert result == expected_response
        controller.llm.apredict.assert_called_once()

    def test_batch_queries(self, controller):
        """Test batch query processing"""
        queries = ["Query 1", "Query 2", "Query 3"]
        
        results = controller.batch_process_queries(queries)
        
        assert len(results) == len(queries)
        assert controller.llm.predict.call_count == len(queries)

    def test_error_handling(self, controller):
        """Test error handling"""
        controller.llm.predict.side_effect = Exception("API Error")
        
        with pytest.raises(Exception) as exc_info:
            controller.process_query("Test query")
        
        assert "API Error" in str(exc_info.value)

    def test_config_validation(self, controller):
        """Test configuration validation"""
        invalid_config = {"invalid_key": "value"}
        
        with pytest.raises(Exception):
            controller.update_config(invalid_config)

    @pytest.mark.parametrize("query,metadata", [
        ("Test query", {"source": "test"}),
        ("Another query", {"priority": "high"}),
        ("Final query", {"tags": ["test", "mock"]})
    ])
    def test_query_with_metadata(self, controller, query, metadata):
        """Test query processing with different metadata"""
        result = controller.process_query(query, metadata=metadata)
        assert result is not None
        # Verify metadata was passed correctly
        call_kwargs = controller.llm.predict.call_args[1]
        assert call_kwargs.get('metadata') == metadata

    def test_streaming_response(self, controller):
        """Test streaming response handling"""
        controller.llm.predict.return_value = iter(["Part 1", "Part 2", "Part 3"])
        
        result = controller.process_query("Test", streaming=True)
        assert result is not None

    def test_context_handling(self, controller):
        """Test context handling in queries"""
        context = "Previous conversation context"
        query = "Follow-up question"
        
        result = controller.process_query(query, context=context)
        
        assert result is not None
        # Verify context was included in the prompt
        call_args = controller.llm.predict.call_args[0][0]
        assert context in call_args

    def test_cleanup(self, controller):
        """Test cleanup operations"""
        if hasattr(controller, 'cleanup'):
            controller.cleanup()
            assert controller.llm.predict.call_count == 0