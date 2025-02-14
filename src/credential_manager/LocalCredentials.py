import os
from dotenv import load_dotenv

# We load args before declaring the static class
load_dotenv(override=False)

class APICredential:
    secret_key: str
    user_key: str | None

class LocalCredentials:
    # Static class to store credentials array
    credentials: dict[str, APICredential] = {}

    @staticmethod
    def add_credential(name: str, secret_key: str, user_key: str | None = None):
        LocalCredentials.credentials[name] = APICredential(secret_key, user_key)
    
    @staticmethod
    def get_credential(name: str) -> APICredential:
        return LocalCredentials.credentials[name]
    
    @staticmethod
    def remove_credential(name: str):
        del LocalCredentials.credentials[name]


# We load env variables and add them to the static class
        
LocalCredentials.add_credential('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY'))
LocalCredentials.add_credential('AWS_IAM_KEY', os.getenv('AWS_SECRET_ACCESS_KEY'), os.getenv('AWS_ACCESS_KEY_ID'))
LocalCredentials.add_credential('GEMINI_API_KEY', os.getenv('GEMINI_API_KEY'))