import pytest
import os
import boto3
from dotenv import load_dotenv

# Load environment variables for tests
load_dotenv()

def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--real-aws", 
        action="store_true", 
        default=False, 
        help="Run tests that interact with real AWS services"
    )

def pytest_configure(config):
    """Configure pytest based on command-line options."""
    # If --real-aws flag is provided, don't skip real_aws tests
    if config.getoption("--real-aws"):
        # Remove the "not real_aws" from the default options
        config.option.markexpr = ""
        
        # Check required environment variables
        required_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'DOCUMENTS_BUCKET_NAME', 
            'DOCUMENT_TEXT_BUCKET_NAME',
            'DOCUMENT_IMAGES_BUCKET_NAME',
            'DOCUMENT_GRAPHS_BUCKET_NAME'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            pytest.skip(
                f"Skipping real AWS tests. Missing environment variables: {', '.join(missing_vars)}",
                allow_module_level=True
            )

@pytest.fixture(scope="session")
def aws_session():
    """Create a boto3 session for testing."""
    return boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    ) 