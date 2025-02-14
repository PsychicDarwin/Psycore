
from enum import Enum

class Providers(Enum):
    OLLAMA = 1
    OPENAI = 2
    BEDROCK = 3
    GEMINI = 4
    HUGGINGFACE = 5

# We create a model type class that allows for easy switching between models and providers like ollama or APIs
class ModelType:
    def __init__(self, argName: str, multiModal: bool, provider: Providers, model_tokens: int | None = None, embedding_tokens: int | None = None):
        self.argName = argName
        self.multiModal = multiModal
        self.provider = provider
        self.model_tokens = model_tokens # This is number of tokens in context window, if can't source then None
        self.embedding_tokens = embedding_tokens # This is the maximum number of tokens in the output, if can't source then None


class LocalModelType(ModelType):
    def __init__(self, argName: str, multiModal: bool, provider: Providers, model_tokens: int | None = None, embedding_tokens: int | None = None, download_size: float | None = None):
        super().__init__(argName, multiModal, provider, model_tokens, embedding_tokens)
        self.download_size = download_size

class EmbeddingType:
    def __init__(self, model: str, provider: Providers, embedding_tokens: int, multiModal: bool):
        self.model = model
        self.provider = provider
        self.embedding_tokens = embedding_tokens
        self.multiModal = multiModal

# We create a static class to store our model types and any info that could be useful
class ModelCatalogue:

    _models = {
        "oai_4o_latest": ModelType('gpt-4o-2024-08-06',True,Providers.OPENAI,128000,16384), # Latest 4o Model
        "oai_chatgpt_latest" :  ModelType('chatgpt-4o-latest',True,Providers.OPENAI,128000,16384), # Latest ChatGPT Model
        "oai_3.5_final" : ModelType('gpt-3.5-turbo-0125',False,Providers.OPENAI,16385,4096), # Latest 3.5 Model before 
        "claude_3_sonnet" : ModelType('anthropic.claude-3-sonnet-20240229-v1:0',True,Providers.BEDROCK,200000,28000), # Claude 3 Sonnet
        "claude_3_haiku" : ModelType('anthropic.claude-3-haiku-20240307-v1:0',True,Providers.BEDROCK,200000,48000), # Claude 3 Haiku
        "meta_llama_3_70b_instruct" : ModelType('meta.llama3-70b-instruct-v1:0',False,Providers.BEDROCK,8000,8000),
        "meta_llama_3_8b_instruct" : ModelType('meta.llama3-8b-instruct-v1:0',False,Providers.BEDROCK,8000,8000),
        "mistral_24.02_large": ModelType('mistral.mistral-large-2402-v1:0',False,Providers.BEDROCK,131000,32000),
        "mistral_7b_instruct": ModelType('mistral.mistral-7b-instruct-v0:2',False,Providers.BEDROCK,131000,32000),
        "mistral_8x7b_instruct" : ModelType('mistral.mixtral-8x7b-instruct-v0:1', False, Providers.BEDROCK, 131000, 32000),
        "gemini_2.0_flash_lite" : ModelType('gemini-2.0-flash-lite-preview-02-05',True,Providers.GEMINI,1048576,8192), # No MLLM output but can take multimodal input
        "gemini_1.5_flash" : ModelType('gemini-1.5-flash',True,Providers.GEMINI,1048576,8192), # No MLLM output but can take multimodal input
        "gemini_1.5_8b_flash": ModelType('gemini-1.5-flash-8b',True,Providers.GEMINI,1048576,8192), # No MLLM output but can take multimodal input
        "gemini_1.5_pro" : ModelType('gemini-1.5-pro',True,Providers.GEMINI,2097152,8192), # No MLLM output but can take multimodal input
        "deepseek_1.5b_r1" : LocalModelType('deepseek-r1:1.5b', False, Providers.OLLAMA,128000,32768,1.1),
        "deepseek_7b_r1" : LocalModelType('deepseek-r1:7b', False, Providers.OLLAMA,128000,32768,4.7),
        "deepseek_8b_r1" : LocalModelType('deepseek-r1:8b', False, Providers.OLLAMA,128000,32768,4.9),
        "deepseek_14b_r1" : LocalModelType('deepseek-r1:14b', False, Providers.OLLAMA,128000,32768,9.0),
        "deepseek_32b_r1" : LocalModelType('deepseek-r1:32b', False, Providers.OLLAMA,128000,32768,20),
        "deepseek_70b_r1" : LocalModelType('deepseek-r1:70b', False, Providers.OLLAMA,128000,32768,43),
        "deepseek_671b_r1" : LocalModelType('deepseek-r1:671b', False, Providers.OLLAMA,128000,32768,404), # This should not be ran on a small local machine, it downloads at 400GB
        "llava_7b" : LocalModelType('llava', True, Providers.OLLAMA, 224000, 4096,4.7),
        "llava_13b" : LocalModelType('llava', True, Providers.OLLAMA, 224000, 4096,8.0),
        "llava_34b" : LocalModelType('llava', True, Providers.OLLAMA, 224000, 4096,20),
        "bakllava_7b" : LocalModelType('bakllava', True, Providers.OLLAMA, None, 2048,4.7),
        "qwen_0.5b_2.5": LocalModelType('qwen2.5:0.5b',False,Providers.OLLAMA,128000,8000,0.398),
        "qwen_1.5b_2.5": LocalModelType('qwen2.5:1.5b',False,Providers.OLLAMA,128000,8000,0.986),
        "qwen_3b_2.5": LocalModelType('qwen2.5:3b',False,Providers.OLLAMA,128000,8000,1.9),
        "qwen_7b_2.5": LocalModelType('qwen2.5:7b',False,Providers.OLLAMA,128000,8000,4.7),
        "qwen_14b_2.5": LocalModelType('qwen2.5:14b',False,Providers.OLLAMA,128000,8000,9.0),
        "qwen_32b_2.5": LocalModelType('qwen2.5:32b',False,Providers.OLLAMA,128000,8000,20),
        "qwen_72b_2.5": LocalModelType('qwen2.5:72b',False,Providers.OLLAMA,128000,8000,47),
        "microsoft_3.8b_phi3" : LocalModelType("phi3",False,Providers.OLLAMA,4000,None, 2.2),
        "microsoft_14b_phi3" : LocalModelType("phi3:14b",False,Providers.OLLAMA,4000,None, 7.9)
    }
    
    _embeddings = {
        "oai_text_3_large" : EmbeddingType('text-embedding-3-large',Providers.OPENAI,3072,False), # Text Embedding 3 Large OpenAI
        "bedrock_text_2_titan" : EmbeddingType('amazon.titan-embed-text-v2:0', Providers.BEDROCK, 8000,False), # Text Embedding 2 Titan Bedrock
        "bedrock_multimodal_g1_titan" : EmbeddingType('amazon.titan-embed-image-v1', Providers.BEDROCK, 128,True), # Multimodal Embedding G1 Titan Bedrock
        "gemini_4_text" : EmbeddingType('text-embedding-004', Providers.GEMINI,2048,False),
        "bge_m3" : EmbeddingType('bge-m3', Providers.OLLAMA, 8192, False),
    }


    def get_MLLMs():
        # Filter through models and return only multimodal models
        return {k:v for k,v in ModelCatalogue._models.items() if v.multiModal}
    
    def get_textLLMs():
        # Filter through models and return only text models
        return {k:v for k,v in ModelCatalogue._models.items() if not v.multiModal}
    
    def get_MEmbeddings():
        # Filter through embeddings and return only multimodal embeddings
        return {k:v for k,v in ModelCatalogue._embeddings.items() if v.multiModal}
    
    def get_textEmbeddings():
        # Filter through embeddings and return only text embeddings
        return {k:v for k,v in ModelCatalogue._embeddings.items() if not v.multiModal}