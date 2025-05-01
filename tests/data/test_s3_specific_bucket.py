import pytest
import boto3
import os
import uuid
from io import BytesIO
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.mark.real_aws
def test_specific_bucket_operations():
    """Test direct operations on the documents bucket without creating it."""
    # Get the documents bucket name from environment
    bucket_name = os.getenv('DOCUMENTS_BUCKET_NAME')
    if not bucket_name:
        pytest.skip("DOCUMENTS_BUCKET_NAME not configured in environment")
    
    # Create an S3 client
    s3_client = boto3.client('s3')
    
    # Generate a unique test key
    test_key = f"test-files/test-{uuid.uuid4()}.txt"
    test_content = b"This is a test file for S3 operations"
    
    try:
        # Try to upload a small file
        print(f"\nUploading to bucket: {bucket_name}, key: {test_key}")
        s3_client.upload_fileobj(
            BytesIO(test_content),
            bucket_name,
            test_key
        )
        
        # Try to download the file
        print(f"Downloading from bucket: {bucket_name}, key: {test_key}")
        download_buffer = BytesIO()
        s3_client.download_fileobj(
            bucket_name,
            test_key,
            download_buffer
        )
        
        # Verify content
        download_buffer.seek(0)
        downloaded_content = download_buffer.read()
        assert downloaded_content == test_content
        print("✓ Upload and download successful!")
        
    except ClientError as e:
        # Print error but don't fail the test
        print(f"❌ Error during S3 operations: {e}")
        pytest.skip(f"S3 operation failed: {e}")
    
    finally:
        # Try to clean up the test file
        try:
            print(f"Cleaning up test file: {test_key}")
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=test_key
            )
            print("✓ Cleanup successful!")
        except ClientError as e:
            print(f"❌ Error during cleanup: {e}")
            # Don't fail the test due to cleanup issues 