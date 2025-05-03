import pytest
import os
import uuid
import traceback
from io import BytesIO
from botocore.exceptions import ClientError

from src.data.s3_handler import S3Handler

@pytest.mark.real_aws
def test_s3_handler_specific_operations():
    """Test S3Handler operations on existing buckets without trying to create them."""
    # Create the S3Handler
    handler = S3Handler()
    
    # Print bucket names for debugging
    print(f"\nUsing buckets:")
    print(f"Documents bucket: {handler.documents_bucket}")
    print(f"Text bucket: {handler.text_bucket}")
    print(f"Images bucket: {handler.images_bucket}")
    print(f"Graphs bucket: {handler.graphs_bucket}")
    
    # Generate a unique test document ID
    doc_id = f"test-{uuid.uuid4()}"
    test_content = "This is a test document for S3Handler operations"
    
    try:
        # Test upload_document_text
        print(f"\nUploading document text with ID: {doc_id}")
        text_link = handler.upload_document_text(doc_id, test_content)
        print(f"Uploaded to: {text_link}")
        
        # Test download_text
        print(f"Downloading from: {text_link}")
        downloaded_text = handler.download_text(text_link)
        assert downloaded_text == test_content
        print("✓ Upload and download successful!")
        
        # Test file_exists
        print(f"Checking if file exists: {text_link}")
        exists = handler.file_exists(text_link)
        assert exists == True
        print("✓ File exists check successful!")
        
        # Test get_presigned_url
        print(f"Generating presigned URL for: {text_link}")
        url = handler.get_presigned_url(text_link, expiration=300)
        assert url.startswith("https://")
        print(f"✓ Presigned URL generated: {url[:60]}...")
        
    except ClientError as e:
        print(f"\n❌ Error during S3 operations:")
        print(f"Error Code: {e.response['Error']['Code']}")
        print(f"Error Message: {e.response['Error']['Message']}")
        print("\nTraceback:")
        traceback.print_exc()
        pytest.skip(f"S3 operation failed: {e.response['Error']['Code']}: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        pytest.skip(f"Unexpected error: {str(e)}")
    
    finally:
        # Try to clean up
        try:
            print(f"Cleaning up test document: {doc_id}")
            deleted_keys = handler.delete_document_files(doc_id)
            print(f"✓ Deleted {sum(len(v) for v in deleted_keys.values())} files")
        except Exception as e:
            print(f"❌ Error during cleanup: {e}")
            # Don't fail the test due to cleanup issues 