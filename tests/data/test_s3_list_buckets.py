import pytest
import boto3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.mark.real_aws
def test_list_buckets():
    """Test that we can list buckets with the provided credentials."""
    # Create an S3 client
    s3_client = boto3.client('s3')
    
    # Try to list buckets
    response = s3_client.list_buckets()
    
    # Extract bucket names
    bucket_names = [bucket['Name'] for bucket in response['Buckets']]
    
    # Print the bucket names for debugging
    print("\nAvailable S3 buckets:")
    for name in bucket_names:
        print(f"  - {name}")
    
    # Assert that we got a response with buckets
    assert 'Buckets' in response
    
    # Check if any of our expected buckets exist
    expected_buckets = [
        os.getenv('DOCUMENTS_BUCKET_NAME'),
        os.getenv('DOCUMENT_TEXT_BUCKET_NAME'),
        os.getenv('DOCUMENT_IMAGES_BUCKET_NAME'),
        os.getenv('DOCUMENT_GRAPHS_BUCKET_NAME')
    ]
    
    # Filter out None values
    expected_buckets = [b for b in expected_buckets if b]
    
    if expected_buckets:
        # Check which of our expected buckets exist
        existing_expected = [b for b in expected_buckets if b in bucket_names]
        missing_expected = [b for b in expected_buckets if b not in bucket_names]
        
        print("\nExpected buckets that exist:")
        for name in existing_expected:
            print(f"  - {name}")
        
        if missing_expected:
            print("\nExpected buckets that don't exist:")
            for name in missing_expected:
                print(f"  - {name}")
    
    # No assertion here, just informational 