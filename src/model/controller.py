from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.llms.bedrock import Bedrock
from langchain_aws import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM
from credential_manager.LocalCredentials import LocalCredentials

# We create a model type class that allows for easy switching between models and providers like ollama or APIs
class ModelType:
    argName: str
    provider: str
    multiModal: bool
    embeddingModel: str


class LLMWrapper:
    def __init__(self, llm: ModelType):
        self.configure(llm)

    def __init__(self, argName: str, provider: str, multiModal: bool, embeddingModel: str):
        self.configure(ModelType(argName, provider, multiModal, embeddingModel))

    def configure(self, llm: ModelType):
        self.llm = llm
        if llm.provider == 'ollama':
            self.model = OllamaLLM(model=llm.argName, base_url=None)
        elif llm.provider == 'openai':
            creds = LocalCredentials.get_credential('OPENAI_API_KEY')
            self.model = OpenAI(model_name=llm.argName,api_key=creds.secret_key)
            self.embeddings = OpenAIEmbeddings(model=llm.embeddingModel,api_key=creds.secret_key)
        elif llm.provider == 'gemini':
            creds = LocalCredentials.get_credential('GEMINI_API_KEY')
            self.model = ChatGoogleGenerativeAI(model=llm.argName,google_api_key=creds.secret_key)
            self.embeddings = GoogleGenerativeAIEmbeddings(model=llm.embeddingModel,google_api_key=creds.secret_key)
        elif llm.provider == 'bedrock':
            creds = LocalCredentials.get_credential('AWS_IAM_KEY')
            self.model = ChatBedrock(model_id=llm.argName,aws_access_key_id=creds.user_key,aws_secret_access_key=creds.secret_key)
            # This needs updating as bedrock needs AWS agent loaded with credentials profile for embeddings, though chatbedrock can use pure keys
            self.embeddings = None
        else:
            raise ValueError("Invalid provider")
        
    
    

