import pytest
import boto3
import os
import uuid
from io import BytesIO
from botocore.exceptions import ClientError

@pytest.mark.integration
def test_direct_bucket_access(aws_session, bucket_name):
    """Test direct access to an S3 bucket using real AWS credentials."""
    if not bucket_name:
        pytest.skip("No bucket name provided for integration test")
    
    s3_client = aws_session.client('s3')
    test_key = 'test/direct_access_test.txt'
    test_content = b'Test content for direct S3 access'
    
    try:
        # Test put object
        s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=test_content)
        print(f"Successfully put object in bucket {bucket_name}")
        
        # Test get object
        response = s3_client.get_object(Bucket=bucket_name, Key=test_key)
        retrieved_content = response['Body'].read()
        assert retrieved_content == test_content
        print(f"Successfully retrieved object from bucket {bucket_name}")
        
        # Test delete object
        s3_client.delete_object(Bucket=bucket_name, Key=test_key)
        print(f"Successfully deleted object from bucket {bucket_name}")
        
    except ClientError as e:
        pytest.fail(f"AWS operation failed: {str(e)}")

def test_mock_bucket_access(mock_s3_client):
    """Test S3 operations using mocked client."""
    bucket_name = 'test-bucket-1'
    test_key = 'test/mock_test.txt'
    test_content = b'Test content for mocked S3'
    
    # Test put object
    mock_s3_client.put_object(Bucket=bucket_name, Key=test_key, Body=test_content)
    
    # Test get object
    response = mock_s3_client.get_object(Bucket=bucket_name, Key=test_key)
    retrieved_content = response['Body'].read()
    assert retrieved_content == test_content
    
    # Test delete object
    mock_s3_client.delete_object(Bucket=bucket_name, Key=test_key)
    
    # Verify deletion by checking the object no longer exists
    with pytest.raises(ClientError) as exc_info:
        mock_s3_client.get_object(Bucket=bucket_name, Key=test_key)
    assert exc_info.value.response['Error']['Code'] == 'NoSuchKey'

if __name__ == "__main__":
    # Test all buckets from .env
    BUCKETS = [
        os.getenv('DOCUMENTS_BUCKET_NAME', ''),
        os.getenv('DOCUMENT_TEXT_BUCKET_NAME', ''),
        os.getenv('DOCUMENT_IMAGES_BUCKET_NAME', ''),
        os.getenv('DOCUMENT_GRAPHS_BUCKET_NAME', '')
    ]
    
    # Filter out empty values
    BUCKETS = [b for b in BUCKETS if b]
    
    print(f"Will test {len(BUCKETS)} buckets:")
    for bucket in BUCKETS:
        print(f"  - {bucket}")
    
    # Test each bucket
    results = {}
    for bucket in BUCKETS:
        results[bucket] = test_direct_bucket_access(bucket)
    
    # Summary
    print("\n=== Summary ===")
    for bucket, success in results.items():
        print(f"{bucket}: {'✓ Accessible' if success else '❌ Not accessible'}") 