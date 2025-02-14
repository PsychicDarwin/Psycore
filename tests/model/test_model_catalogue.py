import pytest
from src.model.model_catalogue import (
    ModelCatalogue, 
    ModelType, 
    LocalModelType, 
    EmbeddingType, 
    Providers
)

def test_providers_enum():
    """Test that all providers are correctly defined in the enum."""
    assert Providers.OLLAMA.value == 1
    assert Providers.OPENAI.value == 2
    assert Providers.BEDROCK.value == 3
    assert Providers.GEMINI.value == 4
    assert Providers.HUGGINGFACE.value == 5

def test_model_type_creation():
    """Test creation of basic ModelType instances."""
    model = ModelType(
        argName="test-model",
        multiModal=True,
        provider=Providers.OPENAI,
        model_tokens=1000,
        embedding_tokens=100
    )
    
    assert model.argName == "test-model"
    assert model.multiModal is True
    assert model.provider == Providers.OPENAI
    assert model.model_tokens == 1000
    assert model.embedding_tokens == 100

def test_local_model_type_creation():
    """Test creation of LocalModelType instances."""
    local_model = LocalModelType(
        argName="test-local-model",
        multiModal=False,
        provider=Providers.OLLAMA,
        model_tokens=1000,
        embedding_tokens=100,
        download_size=2.5
    )
    
    assert local_model.argName == "test-local-model"
    assert local_model.multiModal is False
    assert local_model.provider == Providers.OLLAMA
    assert local_model.model_tokens == 1000
    assert local_model.embedding_tokens == 100
    assert local_model.download_size == 2.5

def test_embedding_type_creation():
    """Test creation of EmbeddingType instances."""
    embedding = EmbeddingType(
        model="test-embedding",
        provider=Providers.OPENAI,
        embedding_tokens=100,
        multiModal=True
    )
    
    assert embedding.model == "test-embedding"
    assert embedding.provider == Providers.OPENAI
    assert embedding.embedding_tokens == 100
    assert embedding.multiModal is True

def test_model_catalogue_mllms():
    """Test filtering of multimodal models."""
    mllms = ModelCatalogue.get_MLLMs()
    
    # Check some known multimodal models
    assert "oai_4o_latest" in mllms
    assert "llava_7b" in mllms
    
    # Verify all returned models are actually multimodal
    assert all(model.multiModal for model in mllms.values())

def test_model_catalogue_text_llms():
    """Test filtering of text-only models."""
    text_llms = ModelCatalogue.get_textLLMs()
    
    # Check some known text-only models
    assert "deepseek_7b_r1" in text_llms
    assert "qwen_7b_2.5" in text_llms
    
    # Verify all returned models are not multimodal
    assert all(not model.multiModal for model in text_llms.values())

def test_model_catalogue_multimodal_embeddings():
    """Test filtering of multimodal embeddings."""
    m_embeddings = ModelCatalogue.get_MEmbeddings()
    
    # Check the known multimodal embedding
    assert "bedrock_multimodal_g1_titan" in m_embeddings
    
    # Verify all returned embeddings are multimodal
    assert all(emb.multiModal for emb in m_embeddings.values())

def test_model_catalogue_text_embeddings():
    """Test filtering of text embeddings."""
    text_embeddings = ModelCatalogue.get_textEmbeddings()
    
    # Check some known text embeddings
    assert "oai_text_3_large" in text_embeddings
    assert "gemini_4_text" in text_embeddings
    
    # Verify all returned embeddings are not multimodal
    assert all(not emb.multiModal for emb in text_embeddings.values())

def test_model_catalogue_provider_consistency():
    """Test that all models have valid providers."""
    all_models = ModelCatalogue._models
    all_embeddings = ModelCatalogue._embeddings
    
    # Check all models have valid providers
    for model in all_models.values():
        assert isinstance(model.provider, Providers)
        assert model.provider in Providers
    
    # Check all embeddings have valid providers
    for embedding in all_embeddings.values():
        assert isinstance(embedding.provider, Providers)
        assert embedding.provider in Providers

def test_model_tokens_validity():
    """Test that token counts are valid where specified."""
    for model in ModelCatalogue._models.values():
        if model.model_tokens is not None:
            assert isinstance(model.model_tokens, int)
            assert model.model_tokens > 0
        
        if model.embedding_tokens is not None:
            assert isinstance(model.embedding_tokens, int)
            assert model.embedding_tokens > 0

def test_embedding_tokens_validity():
    """Test that embedding token counts are valid."""
    for embedding in ModelCatalogue._embeddings.values():
        assert isinstance(embedding.embedding_tokens, int)
        assert embedding.embedding_tokens > 0