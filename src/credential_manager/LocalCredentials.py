import os
from dotenv import load_dotenv

# We load args before declaring the static class
load_dotenv(override=False)

class APICredential:
    def __init__(self, secret_key: str, user_key: str | None = None):
        self.secret_key = secret_key
        self.user_key = user_key

class LocalCredentials:
    # Static class to store credentials array
    _credentials: dict[str, APICredential] = {}

    @staticmethod
    def add_credential(name: str, secret_key: str, user_key: str | None = None):
        LocalCredentials._credentials[name] = APICredential(secret_key, user_key)
    
    @staticmethod
    def get_credential(name: str) -> APICredential:
        return LocalCredentials._credentials[name]
    
    @staticmethod
    def remove_credential(name: str):
        del LocalCredentials._credentials[name]


# We load env variables and add them to the static class
        
LocalCredentials.add_credential('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
LocalCredentials.add_credential('AWS_IAM_KEY', os.getenv('AWS_SECRET_ACCESS_KEY'), os.getenv('AWS_ACCESS_KEY_ID'))
LocalCredentials.add_credential('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY'))
LocalCredentials.add_credential('XAI_API_KEY', os.getenv('XAI_API_KEY'))

# Add S3 bucket names as individual credentials
account_id = os.getenv('AWS_ACCOUNT_ID', 'default')
LocalCredentials.add_credential('S3_DOCUMENTS_BUCKET', os.getenv('DOCUMENTS_BUCKET_NAME', 'test-documents-bucket'))
LocalCredentials.add_credential('S3_TEXT_BUCKET', os.getenv('DOCUMENT_TEXT_BUCKET_NAME', 'test-text-bucket'))
LocalCredentials.add_credential('S3_IMAGES_BUCKET', os.getenv('DOCUMENT_IMAGES_BUCKET_NAME', 'test-images-bucket'))
LocalCredentials.add_credential('S3_GRAPHS_BUCKET', os.getenv('DOCUMENT_GRAPHS_BUCKET_NAME', 'test-graphs-bucket'))

# Add AWS region
LocalCredentials.add_credential('AWS_DEFAULT_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))

# Add DynamoDB table name
LocalCredentials.add_credential('DYNAMODB_DOCUMENT_RELATIONSHIPS_TABLE', os.getenv('DYNAMODB_DOCUMENT_RELATIONSHIPS_TABLE'))

# ChromaDB Configuration
LocalCredentials.add_credential('CHROMADB_HOST', os.getenv('CHROMADB_HOST', '13.42.151.24'))
LocalCredentials.add_credential('CHROMADB_PORT', os.getenv('CHROMADB_PORT', '8000'))