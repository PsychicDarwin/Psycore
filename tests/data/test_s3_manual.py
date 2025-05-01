import os
import uuid
import sys
import boto3
import traceback
from io import BytesIO
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load environment variables
load_dotenv()

def main():
    """Manual test for S3 operations with direct console output."""
    print("\n=== S3 Direct Test ===\n")
    
    # Print environment variables for debugging
    print("Environment variables:")
    print(f"AWS_ACCESS_KEY_ID: {'✓ Set' if os.getenv('AWS_ACCESS_KEY_ID') else '❌ Missing'}")
    print(f"AWS_SECRET_ACCESS_KEY: {'✓ Set' if os.getenv('AWS_SECRET_ACCESS_KEY') else '❌ Missing'}")
    print(f"AWS_DEFAULT_REGION: {os.getenv('AWS_DEFAULT_REGION', 'not set')}")
    print(f"DOCUMENTS_BUCKET_NAME: {os.getenv('DOCUMENTS_BUCKET_NAME', 'not set')}")
    print(f"DOCUMENT_TEXT_BUCKET_NAME: {os.getenv('DOCUMENT_TEXT_BUCKET_NAME', 'not set')}")
    
    # Get the bucket name
    bucket_name = os.getenv('DOCUMENT_TEXT_BUCKET_NAME')
    if not bucket_name:
        print("\n❌ Error: DOCUMENT_TEXT_BUCKET_NAME environment variable not set")
        return
    
    # Create an S3 client
    print("\nCreating S3 client...")
    s3_client = boto3.client('s3')
    
    # Generate test parameters
    test_id = str(uuid.uuid4())[:8]
    test_key = f"test-files/manual-test-{test_id}.txt"
    test_content = f"This is a test file generated at test-{test_id}".encode('utf-8')
    
    print(f"Test key: {test_key}")
    print(f"Bucket: {bucket_name}")
    
    # Test upload
    try:
        print("\nUploading test file...")
        s3_client.upload_fileobj(
            BytesIO(test_content),
            bucket_name,
            test_key
        )
        print("✓ Upload successful")
    except ClientError as e:
        print(f"\n❌ Upload error:")
        print(f"Error Code: {e.response['Error']['Code']}")
        print(f"Error Message: {e.response['Error']['Message']}")
        print("\nTraceback:")
        traceback.print_exc()
        return
    except Exception as e:
        print(f"\n❌ Unexpected upload error: {e}")
        traceback.print_exc()
        return
    
    # Test download
    try:
        print("\nDownloading test file...")
        download_buffer = BytesIO()
        s3_client.download_fileobj(
            bucket_name,
            test_key,
            download_buffer
        )
        
        # Check content
        download_buffer.seek(0)
        downloaded_content = download_buffer.read()
        
        if downloaded_content == test_content:
            print("✓ Download successful - content matches")
        else:
            print("❌ Content mismatch:")
            print(f"Original: {test_content}")
            print(f"Downloaded: {downloaded_content}")
    except ClientError as e:
        print(f"\n❌ Download error:")
        print(f"Error Code: {e.response['Error']['Code']}")
        print(f"Error Message: {e.response['Error']['Message']}")
        print("\nTraceback:")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected download error: {e}")
        traceback.print_exc()
    
    # Test delete
    try:
        print("\nDeleting test file...")
        s3_client.delete_object(
            Bucket=bucket_name,
            Key=test_key
        )
        print("✓ Delete successful")
    except ClientError as e:
        print(f"\n❌ Delete error:")
        print(f"Error Code: {e.response['Error']['Code']}")
        print(f"Error Message: {e.response['Error']['Message']}")
        print("\nTraceback:")
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ Unexpected delete error: {e}")
        traceback.print_exc()
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main() 