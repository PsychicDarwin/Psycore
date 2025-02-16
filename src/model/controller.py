from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from credential_manager.LocalCredentials import LocalCredentials
from model_catalogue import ModelType, EmbeddingType, Providers

class ModelWrapper:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        match model_type.provider:
            # Important to remember each model will need it's own input template and prior converter for additional input
            case Providers.OPENAI:
                credential = LocalCredentials.get_credential('OPENAI_API_KEY')
                self.model = OpenAI(model_name=model_type.argName, api_key=credential.secret_key)
            case Providers.BEDROCK:
                credential = LocalCredentials.get_credential('AWS_IAM_KEY')
                self.model = ChatBedrock(model_id=model_type.argName, aws_access_key_id=credential.user_key, aws_secret_access_key=credential.secret_key)
            case Providers.GEMINI:
                credential = LocalCredentials.get_credential('GEMINI_API_KEY')
                self.model = ChatGoogleGenerativeAI(model=model_type.argName, google_api_key=credential.secret_key)
            case Providers.OLLAMA:
                self.model = OllamaLLM(model=model_type.argName, base_url=None)
            case Providers.HUGGINGFACE:
                raise NotImplementedError("Huggingface is not yet supported")
            case _:
                raise ValueError("Invalid provider")
            

class EmbeddingWrapper:
    def __init__(self, embedding_type: EmbeddingType):
        self.embedding_type = embedding_type
        if embedding_type.provider == Providers.OPENAI:
            credential = LocalCredentials.get_credential('OPENAI_API_KEY')
            self.embedding = OpenAIEmbeddings(model=embedding_type.model, api_key=credential.secret_key)
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

