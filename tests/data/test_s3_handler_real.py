import pytest
import os
import uuid
import json
import boto3
from io import BytesIO
from botocore.exceptions import ClientError

from src.data.s3_handler import S3Handler

# Skip all tests if credentials unavailable
pytestmark = pytest.mark.real_aws

@pytest.fixture(scope="module")
def s3_handler(aws_session):
    """Create a real S3Handler instance."""
    handler = S3Handler()
    # Make sure buckets exist for testing
    _ensure_buckets_exist(handler, aws_session)
    return handler

@pytest.fixture
def test_document_id():
    """Generate a unique document ID for testing."""
    return f"test-doc-{uuid.uuid4()}"

def _ensure_buckets_exist(handler, aws_session):
    """Check if required buckets exist."""
    s3_client = aws_session.client('s3')
    region = aws_session.region_name
    
    # Check if buckets exist
    missing_buckets = []
    for bucket_name in [
        handler.documents_bucket,
        handler.text_bucket,
        handler.images_bucket,
        handler.graphs_bucket
    ]:
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            print(f"✓ Bucket exists: {bucket_name}")
        except ClientError as e:
            missing_buckets.append(bucket_name)
            print(f"✗ Bucket missing: {bucket_name}")
    
    if missing_buckets:
        pytest.skip(f"Skipping tests because the following buckets don't exist: {', '.join(missing_buckets)}")
    
    return True

def _cleanup_test_files(handler, doc_id):
    """Clean up all test files after tests."""
    handler.delete_document_files(doc_id)

@pytest.mark.real_aws
def test_real_upload_document(s3_handler):
    """Test uploading a document to real S3."""
    # Create test file
    test_file = BytesIO(b"test file content")
    original_filename = "test_doc.pdf"
    
    # Upload document
    doc_id, s3_link = s3_handler.upload_document(test_file, original_filename)
    
    try:
        # Verify document was uploaded
        assert s3_handler.file_exists(s3_link) == True
        
        # Download and verify content
        content = s3_handler.download_file(s3_link)
        assert content == b"test file content"
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id)

@pytest.mark.real_aws
def test_real_document_text_operations(s3_handler, test_document_id):
    """Test document text operations with real S3."""
    doc_id = test_document_id
    text_content = "This is a test document content"
    
    try:
        # Upload text
        s3_link = s3_handler.upload_document_text(doc_id, text_content)
        
        # Verify text was uploaded
        assert s3_handler.file_exists(s3_link) == True
        
        # Download and verify content
        downloaded_text = s3_handler.download_text(s3_link)
        assert downloaded_text == text_content
        
        # Test summary upload/download
        summary_content = "This is a document summary"
        summary_link = s3_handler.upload_document_summary(doc_id, summary_content)
        
        downloaded_summary = s3_handler.download_text(summary_link)
        assert downloaded_summary == summary_content
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id)

@pytest.mark.real_aws
def test_real_image_operations(s3_handler, test_document_id):
    """Test image operations with real S3."""
    doc_id = test_document_id
    image_data = BytesIO(b"test image data")
    image_number = 1
    
    try:
        # Upload image
        s3_link = s3_handler.upload_image(doc_id, image_data, image_number)
        
        # Verify image was uploaded
        assert s3_handler.file_exists(s3_link) == True
        
        # Download and verify content
        content = s3_handler.download_file(s3_link)
        assert content == b"test image data"
        
        # Test image text
        image_text = "Text extracted from test image"
        text_link = s3_handler.upload_image_text(doc_id, image_text, image_number)
        
        downloaded_text = s3_handler.download_text(text_link)
        assert downloaded_text == image_text
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id)

@pytest.mark.real_aws
def test_real_graph_operations(s3_handler, test_document_id):
    """Test graph operations with real S3."""
    doc_id = test_document_id
    graph_data = {
        "nodes": [{"id": "1", "label": "Node 1"}, {"id": "2", "label": "Node 2"}],
        "edges": [{"from": "1", "to": "2", "label": "connects to"}]
    }
    graph_json = json.dumps(graph_data)
    
    try:
        # Upload graph
        s3_link = s3_handler.upload_graph(doc_id, graph_json)
        
        # Verify graph was uploaded
        assert s3_handler.file_exists(s3_link) == True
        
        # Download and verify content
        downloaded_graph = s3_handler.download_text(s3_link)
        assert json.loads(downloaded_graph) == graph_data
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id)

@pytest.mark.real_aws
def test_real_list_document_files(s3_handler, test_document_id):
    """Test listing document files with real S3."""
    doc_id = test_document_id
    
    try:
        # Upload various files for the document
        s3_handler.upload_document_text(doc_id, "Main text")
        s3_handler.upload_document_summary(doc_id, "Summary text")
        s3_handler.upload_image(doc_id, BytesIO(b"image1"), 1)
        s3_handler.upload_image(doc_id, BytesIO(b"image2"), 2)
        s3_handler.upload_image_text(doc_id, "Image 1 text", 1)
        s3_handler.upload_image_text(doc_id, "Image 2 text", 2)
        s3_handler.upload_graph(doc_id, json.dumps({"nodes": [], "edges": []}))
        
        # List files
        files = s3_handler.list_document_files(doc_id)
        
        # Verify counts
        assert len(files['text']) == 4  # main, summary, image1 text, image2 text
        assert len(files['images']) == 2  # image1, image2
        assert len(files['graphs']) == 1  # graph
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id)

@pytest.mark.real_aws
def test_real_delete_document_files(s3_handler, test_document_id):
    """Test deleting document files with real S3."""
    doc_id = test_document_id
    
    # Upload various files for the document
    s3_handler.upload_document_text(doc_id, "Main text")
    s3_handler.upload_document_summary(doc_id, "Summary text")
    s3_handler.upload_image(doc_id, BytesIO(b"image1"), 1)
    s3_handler.upload_graph(doc_id, json.dumps({"nodes": [], "edges": []}))
    
    # Verify files exist
    files_before = s3_handler.list_document_files(doc_id)
    assert len(files_before['text']) > 0
    assert len(files_before['images']) > 0
    assert len(files_before['graphs']) > 0
    
    # Delete files
    deleted_keys = s3_handler.delete_document_files(doc_id)
    
    # Verify deletion
    files_after = s3_handler.list_document_files(doc_id)
    assert len(files_after['text']) == 0
    assert len(files_after['images']) == 0
    assert len(files_after['graphs']) == 0

@pytest.mark.real_aws
def test_real_presigned_url(s3_handler, test_document_id):
    """Test generating presigned URLs with real S3."""
    doc_id = test_document_id
    
    try:
        # Upload a file
        text_link = s3_handler.upload_document_text(doc_id, "Test content")
        
        # Generate presigned URL
        presigned_url = s3_handler.get_presigned_url(text_link)
        
        # Verify URL format
        assert presigned_url.startswith("https://")
        assert "AWSAccessKeyId=" in presigned_url or "X-Amz-Credential=" in presigned_url
        assert "Signature=" in presigned_url or "X-Amz-Signature=" in presigned_url
        
        # Note: We don't test actual URL access as it would require HTTP client
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id)

@pytest.mark.real_aws
def test_real_process_s3_file(s3_handler, test_document_id):
    """Test processing a file from S3 using the process_s3_file method."""
    doc_id = test_document_id
    test_content = b"This is test content for processing with callback"
    
    try:
        # First upload a test file
        from io import BytesIO
        test_file = BytesIO(test_content)
        original_filename = f"{doc_id}.txt"
        
        _, s3_link = s3_handler.upload_document(test_file, original_filename)
        
        # Parse bucket and key from S3 link
        parts = s3_link.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        # Create file_info dict
        file_info = {
            'Key': key,
            'Bucket': bucket
        }
        
        # Process file using process_s3_file
        result = s3_handler.process_s3_file(
            file_info,
            lambda path: {
                'success': True,
                'file_size': os.path.getsize(path),
                'content': open(path, 'rb').read()
            }
        )
        
        # Verify results
        assert result['success'] == True
        assert result['file_size'] == len(test_content)
        assert result['content'] == test_content
        
        # Test download_to_temp_and_process directly
        processed = s3_handler.download_to_temp_and_process(
            bucket, 
            key,
            lambda path: {'path_exists': os.path.exists(path), 'path': path}
        )
        
        assert processed['path_exists'] == True
        assert os.path.basename(processed['path']).startswith('test-')
            
    finally:
        # Clean up
        _cleanup_test_files(s3_handler, doc_id) 