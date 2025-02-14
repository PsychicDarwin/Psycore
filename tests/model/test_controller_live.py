import pytest
import os
from src.model.controller import ModelController
from dotenv import load_dotenv
import logging
import asyncio

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Skip these tests if API key is not available
requires_api_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="API key not found in environment variables"
)

@pytest.fixture(scope="module")
def controller():
    """Create a real ModelController instance for testing"""
    try:
        controller = ModelController()
        logger.info("Controller initialized successfully")
        return controller
    except Exception as e:
        logger.error(f"Failed to initialize controller: {e}")
        raise

@requires_api_key
class TestControllerLive:
    """Live API test suite for ModelController"""

    def test_basic_query(self, controller):
        """Test basic query with live API"""
        query = "What is artificial intelligence?"
        
        try:
            result = controller.process_query(query)
            
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            logger.info("Basic query test successful")
            
        except Exception as e:
            logger.error(f"Basic query test failed: {e}")
            raise

    @pytest.mark.asyncio
    async def test_async_query(self, controller):
        """Test async query processing with live API"""
        query = "Explain machine learning briefly."
        
        try:
            result = await controller.async_process_query(query)
            
            assert result is not None
            assert isinstance(result, str)
            assert len(result) > 0
            logger.info("Async query test successful")
            
        except Exception as e:
            logger.error(f"Async query test failed: {e}")
            raise

    def test_scientific_query(self, controller):
        """Test processing of scientific queries"""
        query = """
        Explain the relationship between quantum entanglement and quantum computing
        in terms of qubits and quantum gates.
        """
        
        try:
            result = controller.process_query(query)
            
            assert result is not None
            assert len(result) > 100  # Expect detailed response
            logger.info("Scientific query test successful")
            
        except Exception as e:
            logger.error(f"Scientific query test failed: {e}")
            raise

    def test_streaming_response(self, controller):
        """Test streaming response functionality"""
        query = "Write a brief story about AI."
        
        try:
            result = controller.process_query(query, streaming=True)
            
            assert result is not None
            logger.info("Streaming response test successful")
            
        except Exception as e:
            logger.error(f"Streaming response test failed: {e}")
            raise

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
    def test_temperature_variations(self, controller, temperature):
        """Test response variations with different temperatures"""
        query = "What might the future of AI look like?"
        
        try:
            controller.update_config({"temperature": temperature})
            result = controller.process_query(query)
            
            assert result is not None
            logger.info(f"Temperature {temperature} test successful")
            
        except Exception as e:
            logger.error(f"Temperature {temperature} test failed: {e}")
            raise

    def test_error_case(self, controller):
        """Test API error handling with invalid input"""
        # Create a very long query that exceeds token limits
        very_long_query = "test " * 5000
        
        with pytest.raises(Exception) as exc_info:
            controller.process_query(very_long_query)
        logger.info("Error case test successful")

    def test_batch_processing(self, controller):
        """Test batch processing with actual API calls"""
        queries = [
            "What is deep learning?",
            "Explain neural networks.",
            "What is machine learning?"
        ]
        
        try:
            results = controller.batch_process_queries(queries)
            
            assert len(results) == len(queries)
            assert all(isinstance(r, str) for r in results)
            assert all(len(r) > 0 for r in results)
            logger.info("Batch processing test successful")
            
        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            raise

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, controller):
        """Test handling of concurrent API requests"""
        queries = [
            "What is AI?",
            "Explain ML",
            "Define DL"
        ]
        
        try:
            tasks = [
                controller.async_process_query(query)
                for query in queries
            ]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(queries)
            assert all(isinstance(r, str) for r in results)
            logger.info("Concurrent queries test successful")
            
        except Exception as e:
            logger.error(f"Concurrent queries test failed: {e}")
            raise

    def test_context_window(self, controller):
        """Test handling of context window limits"""
        # Test with a conversation that builds context
        conversation = []
        try:
            for i in range(3):
                query = f"Question {i+1}: Tell me more about what you just explained."
                context = "\n".join(conversation)
                result = controller.process_query(query, context=context)
                conversation.append(result)
                
                assert result is not None
                assert len(result) > 0
            
            logger.info("Context window test successful")
            
        except Exception as e:
            logger.error(f"Context window test failed: {e}")
            raise