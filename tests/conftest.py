import pytest
import os
import boto3
from dotenv import load_dotenv
from unittest.mock import MagicMock
from moto import mock_aws

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

# Fixture for mocked AWS session (unit tests)
@pytest.fixture
def mock_aws_session():
    """Mock AWS session for testing."""
    with mock_aws():
        yield

# Fixture for mocked S3 client (unit tests)
@pytest.fixture
def mock_s3_client(mock_aws_session):
    """Create a mock S3 client and test buckets."""
    s3_client = boto3.client('s3', region_name='us-east-1')
    
    # Create test buckets
    test_buckets = ['test-bucket-1', 'test-bucket-2']
    for bucket in test_buckets:
        s3_client.create_bucket(Bucket=bucket)
    
    return s3_client

# Fixture for real AWS session (integration tests)
@pytest.fixture
def aws_session():
    """Real AWS session for integration testing."""
    return boto3.Session()

# Fixture for test bucket names (integration tests)
@pytest.fixture
def bucket_names():
    """Get bucket names from environment variables for integration testing."""
    bucket_vars = [
        os.getenv('PSYCORE_DOCUMENT_BUCKET'),
        os.getenv('PSYCORE_SUMMARY_BUCKET'),
        os.getenv('PSYCORE_IMAGE_BUCKET')
    ]
    return [bucket for bucket in bucket_vars if bucket]

# Fixture for a single test bucket name (for simpler tests)
@pytest.fixture
def bucket_name(bucket_names):
    """Get a single bucket name for simpler tests."""
    return bucket_names[0] if bucket_names else None 