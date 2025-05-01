import boto3
import os
import uuid
from io import BytesIO
from botocore.exceptions import ClientError

# Function for direct S3 bucket access test
def test_direct_bucket_access(bucket_name):
    print(f"\nTesting direct access to bucket: {bucket_name}")
    
    # Create the S3 client
    s3 = boto3.client('s3')
    
    # Create a test key
    test_key = f"test-files/test-{uuid.uuid4()}.txt"
    test_content = b"This is a test file"
    
    try:
        # Put an object directly
        print(f"Putting object to {bucket_name}/{test_key}")
        s3.put_object(
            Bucket=bucket_name,
            Key=test_key,
            Body=test_content
        )
        print(f"✓ Successfully put object to bucket")
        
        # Try to get object
        print(f"Getting object from {bucket_name}/{test_key}")
        response = s3.get_object(
            Bucket=bucket_name,
            Key=test_key
        )
        content = response['Body'].read()
        if content == test_content:
            print(f"✓ Successfully retrieved object with matching content")
        else:
            print(f"❌ Content mismatch!")
            
        # Clean up
        print(f"Deleting test object")
        s3.delete_object(
            Bucket=bucket_name,
            Key=test_key
        )
        print(f"✓ Successfully deleted test object")
        
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f"❌ Error: {error_code} - {error_message}")
        return False

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