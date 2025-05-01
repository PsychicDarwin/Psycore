import pytest
import os
import uuid
from io import BytesIO
from botocore.exceptions import ClientError
from src.data.s3_handler import S3Handler

@pytest.mark.real_aws
def test_s3_handler_direct():
    """Test S3Handler with direct bucket operations."""
    handler = S3Handler()
    
    # Print bucket names
    print(f"\nUsing buckets:")
    print(f"Documents bucket: {handler.documents_bucket}")
    print(f"Text bucket: {handler.text_bucket}")
    print(f"Images bucket: {handler.images_bucket}")
    print(f"Graphs bucket: {handler.graphs_bucket}")
    
    # Generate a unique document ID
    doc_id = f"test-{uuid.uuid4()}"
    print(f"Test document ID: {doc_id}")
    
    try:
        # Test document text operations
        print("\nTesting document text operations...")
        text_content = "This is a test document text"
        text_link = handler.upload_document_text(doc_id, text_content)
        print(f"Uploaded document text to {text_link}")
        
        # Test document summary
        print("\nTesting document summary operations...")
        summary_content = "This is a test document summary"
        summary_link = handler.upload_document_summary(doc_id, summary_content)
        print(f"Uploaded document summary to {summary_link}")
        
        # Test image operations
        print("\nTesting image operations...")
        image_data = BytesIO(b"fake image data")
        image_number = 1
        image_link = handler.upload_image(doc_id, image_data, image_number)
        print(f"Uploaded image to {image_link}")
        
        # Test image text
        print("\nTesting image text operations...")
        image_text = "Text extracted from test image"
        image_text_link = handler.upload_image_text(doc_id, image_text, image_number)
        print(f"Uploaded image text to {image_text_link}")
        
        # Test graph operations
        print("\nTesting graph operations...")
        graph_json = '{"nodes": [], "edges": []}'
        graph_link = handler.upload_graph(doc_id, graph_json)
        print(f"Uploaded graph to {graph_link}")
        
        # Test download operations
        print("\nTesting download operations...")
        downloaded_text = handler.download_text(text_link)
        assert downloaded_text == text_content
        print("✓ Successfully downloaded document text")
        
        downloaded_summary = handler.download_text(summary_link)
        assert downloaded_summary == summary_content
        print("✓ Successfully downloaded document summary")
        
        downloaded_image_text = handler.download_text(image_text_link)
        assert downloaded_image_text == image_text
        print("✓ Successfully downloaded image text")
        
        downloaded_graph = handler.download_text(graph_link)
        assert downloaded_graph == graph_json
        print("✓ Successfully downloaded graph")
        
        # Test file existence
        print("\nTesting file existence...")
        assert handler.file_exists(text_link) == True
        assert handler.file_exists(summary_link) == True
        assert handler.file_exists(image_link) == True
        assert handler.file_exists(image_text_link) == True
        assert handler.file_exists(graph_link) == True
        print("✓ Successfully verified file existence")
        
        # Test listing files
        print("\nTesting file listing...")
        try:
            files = handler.list_document_files(doc_id)
            print(f"Documents: {len(files['documents'])}")
            print(f"Text: {len(files['text'])}")
            print(f"Images: {len(files['images'])}")
            print(f"Graphs: {len(files['graphs'])}")
        except ClientError as e:
            print(f"❌ List files error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
            print("This is expected if the user doesn't have list bucket permissions")
        
        # Test getting presigned URL
        print("\nTesting presigned URL generation...")
        url = handler.get_presigned_url(text_link)
        assert url.startswith("https://")
        print(f"✓ Successfully generated presigned URL: {url[:60]}...")
        
    except ClientError as e:
        print(f"❌ Error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        pytest.fail(f"S3 operation failed: {e}")
    
    finally:
        # Clean up
        print("\nCleaning up test files...")
        try:
            deleted = handler.delete_document_files(doc_id)
            deleted_count = sum(len(files) for files in deleted.values())
            print(f"✓ Successfully deleted {deleted_count} test files")
        except ClientError as e:
            print(f"❌ Cleanup error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
            print("This may happen if the user doesn't have list bucket permissions") 