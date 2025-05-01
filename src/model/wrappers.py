from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_community.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from src.credential_manager.LocalCredentials import LocalCredentials
from src.model.model_catalogue import ModelType, EmbeddingType, Providers, ModelCatalogue
from langchain_xai import ChatXAI


class ChatModelWrapper:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        provider = model_type.provider
        if provider == Providers.OPENAI:
            credential = LocalCredentials.get_credential('OPENAI_API_KEY')
            self.model = ChatOpenAI(
                model=model_type.argName,
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                api_key=credential.secret_key,
            )
        elif provider == Providers.BEDROCK:
            credential = LocalCredentials.get_credential('AWS_IAM_KEY')
            self.model = ChatBedrock(model_id=model_type.argName, aws_access_key_id=credential.user_key, aws_secret_access_key=credential.secret_key)
        elif provider == Providers.GEMINI:
            credential = LocalCredentials.get_credential('GEMINI_API_KEY')
            self.model = ChatGoogleGenerativeAI(model=model_type.argName, google_api_key=credential.secret_key)
        elif provider == Providers.OLLAMA:
            self.model = ChatOllama(model=model_type.argName)
        elif provider == Providers.XAI:
            credential = LocalCredentials.get_credential('XAI_API_KEY')
            self.model = ChatXAI(model=model_type.argName,api_key=credential.secret_key)
        elif provider == Providers.HUGGINGFACE:
            raise NotImplementedError("Huggingface is not yet supported")
        else:
            raise ValueError("Invalid provider")

# This is more for integration with node4j knowledge graphs
class BaseModelWrapper:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        provider = model_type.provider
        # Some models may break as they may require chat specific APIs like ChatGPT latest
        if provider == Providers.OPENAI:
            credential = LocalCredentials.get_credential('OPENAI_API_KEY')
            self.model = OpenAI(
                model_name=model_type.argName, 
                api_key=credential.secret_key,
                temperature=0,
            )
        elif provider == Providers.BEDROCK:
            credential = LocalCredentials.get_credential('AWS_IAM_KEY')
            self.model = Bedrock(model_id=model_type.argName, aws_access_key_id=credential.user_key, aws_secret_access_key=credential.secret_key)
        elif provider == Providers.GEMINI:
            credential = LocalCredentials.get_credential('GEMINI_API_KEY')
            self.model = GoogleGenerativeAI(model=model_type.argName, google_api_key=credential.secret_key)
        elif provider == Providers.OLLAMA:
            self.model = OllamaLLM(model=model_type.argName)
        elif provider == Providers.XAI:
            credential = LocalCredentials.get_credential('XAI_API_KEY')
            # XAI only has chat models
            self.model = ChatXAI(model=model_type.argName,api_key=credential.secret_key)
        elif provider == Providers.HUGGINGFACE:
            raise NotImplementedError("Huggingface is not yet supported")
        else:
            raise ValueError("Invalid provider")

class EmbeddingWrapper:
    def __init__(self, embedding_type: EmbeddingType):
        self.embedding_type = embedding_type
        if embedding_type.provider == Providers.OPENAI:
            credential = LocalCredentials.get_credential('OPENAI_API_KEY')
            self.embedding = OpenAIEmbeddings(
                model=embedding_type.model, 
                api_key=credential.secret_key,
            )
        elif embedding_type.provider == Providers.BEDROCK:
            credential = LocalCredentials.get_credential('AWS_IAM_KEY')
            raise NotImplementedError("Bedrock is not yet supported")
            # This needs an AWS agent loaded with credentials profile as opposed to just keys like chatbedrock
        elif embedding_type.provider == Providers.GEMINI:
            credential = LocalCredentials.get_credential('GEMINI_API_KEY')
            self.embedding = GoogleGenerativeAIEmbeddings(model=embedding_type.model, google_api_key=credential.secret_key)
        elif embedding_type.provider == Providers.OLLAMA:
            self.embedding = OllamaEmbeddings(model=embedding_type.model)
        elif embedding_type.provider == Providers.HUGGINGFACE:
            raise NotImplementedError("Huggingface is not yet supported")
        else:
            raise ValueError("Invalid provider")
