import pytest
import os
import uuid
import json
from unittest.mock import patch, MagicMock, mock_open
from io import BytesIO
from botocore.exceptions import ClientError

from src.data.s3_handler import S3Handler

# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv('DOCUMENTS_BUCKET_NAME', 'test-documents-bucket')
    monkeypatch.setenv('DOCUMENT_TEXT_BUCKET_NAME', 'test-text-bucket')
    monkeypatch.setenv('DOCUMENT_IMAGES_BUCKET_NAME', 'test-images-bucket')
    monkeypatch.setenv('DOCUMENT_GRAPHS_BUCKET_NAME', 'test-graphs-bucket')

# Create a mock S3 client
@pytest.fixture
def mock_s3_client():
    """Create a mock boto3 S3 client."""
    with patch('boto3.Session') as mock_session:
        s3_mock = MagicMock()
        session_instance = MagicMock()
        session_instance.client.return_value = s3_mock
        mock_session.return_value = session_instance
        yield s3_mock

@pytest.fixture
def s3_handler(mock_s3_client):
    """Create an S3Handler instance with mocked S3 client."""
    with patch('src.credential_manager.LocalCredentials.LocalCredentials.get_credential') as mock_get_cred:
        # Mock AWS credentials
        mock_get_cred.side_effect = lambda x: MagicMock(
            user_key='test-key',
            secret_key='test-secret' if x == 'AWS_IAM_KEY' else {
                'AWS_DEFAULT_REGION': 'us-east-1',
                'S3_DOCUMENTS_BUCKET': 'test-documents-bucket',
                'S3_TEXT_BUCKET': 'test-text-bucket',
                'S3_IMAGES_BUCKET': 'test-images-bucket',
                'S3_GRAPHS_BUCKET': 'test-graphs-bucket'
            }.get(x, 'test-value')
        )
        handler = S3Handler()
        return handler

def test_init(s3_handler):
    """Test S3Handler initialization."""
    assert s3_handler.documents_bucket == 'test-documents-bucket'
    assert s3_handler.text_bucket == 'test-text-bucket'
    assert s3_handler.images_bucket == 'test-images-bucket'
    assert s3_handler.graphs_bucket == 'test-graphs-bucket'

def test_upload_document(s3_handler, mock_s3_client, monkeypatch):
    """Test document upload."""
    # Mock uuid generation to get a deterministic result
    fixed_uuid = "550e8400-e29b-41d4-a716-446655440000"
    monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(fixed_uuid))
    
    # Create test file
    test_file = BytesIO(b"test file content")
    original_filename = "test_doc.pdf"
    
    # Configure mock
    mock_s3_client.upload_fileobj.return_value = None
    
    # Call function
    doc_id, s3_link = s3_handler.upload_document(test_file, original_filename)
    
    # Verify results
    assert doc_id == f"doc-{fixed_uuid}"
    assert s3_link == f"s3://test-documents-bucket/documents/doc-{fixed_uuid}.pdf"
    
    # Verify the mock was called correctly
    mock_s3_client.upload_fileobj.assert_called_once_with(
        test_file, 
        'test-documents-bucket', 
        f"documents/doc-{fixed_uuid}.pdf"
    )

def test_upload_document_error(s3_handler, mock_s3_client):
    """Test document upload with error."""
    # Create test file
    test_file = BytesIO(b"test file content")
    original_filename = "test_doc.pdf"
    
    # Configure mock to raise an exception
    error = ClientError({"Error": {"Code": "TestException", "Message": "Test error"}}, "upload_fileobj")
    mock_s3_client.upload_fileobj.side_effect = error
    
    # Test that the exception is raised
    with pytest.raises(ClientError):
        s3_handler.upload_document(test_file, original_filename)

def test_upload_document_text(s3_handler, mock_s3_client):
    """Test uploading document text."""
    doc_id = "doc-test-id"
    text_content = "This is a test document content"
    
    # Configure mock
    mock_s3_client.put_object.return_value = None
    
    # Call function
    s3_link = s3_handler.upload_document_text(doc_id, text_content)
    
    # Verify results
    assert s3_link == f"s3://test-text-bucket/document-text/{doc_id}/main.txt"
    
    # Verify the mock was called correctly
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='test-text-bucket',
        Key=f"document-text/{doc_id}/main.txt",
        Body=text_content.encode('utf-8'),
        ContentType='text/plain'
    )

def test_upload_document_summary(s3_handler, mock_s3_client):
    """Test uploading document summary."""
    doc_id = "doc-test-id"
    summary_content = "This is a summary of the document"
    
    # Configure mock
    mock_s3_client.put_object.return_value = None
    
    # Call function
    s3_link = s3_handler.upload_document_summary(doc_id, summary_content)
    
    # Verify results
    assert s3_link == f"s3://test-text-bucket/document-text/{doc_id}/summary.txt"
    
    # Verify the mock was called correctly
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='test-text-bucket',
        Key=f"document-text/{doc_id}/summary.txt",
        Body=summary_content.encode('utf-8'),
        ContentType='text/plain'
    )

def test_upload_image(s3_handler, mock_s3_client):
    """Test uploading an image."""
    doc_id = "doc-test-id"
    image_data = BytesIO(b"test image data")
    image_number = 1
    extension = ".jpg"
    
    # Configure mock
    mock_s3_client.upload_fileobj.return_value = None
    
    # Call function
    s3_link = s3_handler.upload_image(doc_id, image_data, image_number, extension)
    
    # Verify results
    assert s3_link == f"s3://test-images-bucket/document-images/{doc_id}/image1.jpg"
    
    # Verify the mock was called correctly
    mock_s3_client.upload_fileobj.assert_called_once_with(
        image_data,
        'test-images-bucket',
        f"document-images/{doc_id}/image1.jpg"
    )

def test_upload_image_text(s3_handler, mock_s3_client):
    """Test uploading text extracted from an image."""
    doc_id = "doc-test-id"
    text_content = "Text extracted from image"
    image_number = 1
    
    # Configure mock
    mock_s3_client.put_object.return_value = None
    
    # Call function
    s3_link = s3_handler.upload_image_text(doc_id, text_content, image_number)
    
    # Verify results
    assert s3_link == f"s3://test-text-bucket/document-text/{doc_id}/image1.txt"
    
    # Verify the mock was called correctly
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='test-text-bucket',
        Key=f"document-text/{doc_id}/image1.txt",
        Body=text_content.encode('utf-8'),
        ContentType='text/plain'
    )

def test_upload_graph(s3_handler, mock_s3_client):
    """Test uploading a document graph."""
    doc_id = "doc-test-id"
    graph_json = json.dumps({"nodes": [], "edges": []})
    
    # Configure mock
    mock_s3_client.put_object.return_value = None
    
    # Call function
    s3_link = s3_handler.upload_graph(doc_id, graph_json)
    
    # Verify results
    assert s3_link == f"s3://test-graphs-bucket/document-graphs/{doc_id}.json"
    
    # Verify the mock was called correctly
    mock_s3_client.put_object.assert_called_once_with(
        Bucket='test-graphs-bucket',
        Key=f"document-graphs/{doc_id}.json",
        Body=graph_json.encode('utf-8'),
        ContentType='application/json'
    )

def test_download_file(s3_handler, mock_s3_client):
    """Test downloading a file."""
    # Setup
    s3_link = "s3://test-bucket/test-key"
    mock_response = {"Body": BytesIO(b"test file content")}
    mock_s3_client.get_object.return_value = mock_response
    
    # Call function
    content = s3_handler.download_file(s3_link)
    
    # Verify results
    assert content == b"test file content"
    
    # Verify the mock was called correctly
    mock_s3_client.get_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test-key"
    )

def test_download_text(s3_handler, mock_s3_client):
    """Test downloading and decoding a text file."""
    # Setup
    s3_link = "s3://test-bucket/test-key"
    mock_response = {"Body": BytesIO(b"test text content")}
    mock_s3_client.get_object.return_value = mock_response
    
    # Call function
    content = s3_handler.download_text(s3_link)
    
    # Verify results
    assert content == "test text content"

def test_delete_document_files(s3_handler, mock_s3_client):
    """Test deleting all files associated with a document."""
    # Setup
    doc_id = "doc-test-id"
    
    # Mock responses for list_objects_v2
    docs_response = {
        "Contents": [
            {"Key": f"documents/{doc_id}.pdf"},
            {"Key": f"documents/{doc_id}_v2.pdf"}
        ]
    }
    text_response = {
        "Contents": [
            {"Key": f"document-text/{doc_id}/main.txt"},
            {"Key": f"document-text/{doc_id}/summary.txt"}
        ]
    }
    images_response = {
        "Contents": [
            {"Key": f"document-images/{doc_id}/image1.png"},
            {"Key": f"document-images/{doc_id}/image2.png"}
        ]
    }
    
    # Configure mock
    mock_s3_client.list_objects_v2.side_effect = [
        docs_response, text_response, images_response, {}
    ]
    
    # Call function
    deleted_keys = s3_handler.delete_document_files(doc_id)
    
    # Verify results
    assert len(deleted_keys['test-documents-bucket']) == 2
    assert len(deleted_keys['test-text-bucket']) == 2
    assert len(deleted_keys['test-images-bucket']) == 2
    
    # Verify delete_object was called for each file
    assert mock_s3_client.delete_object.call_count == 7  # 2 docs + 2 texts + 2 images + 1 graph

def test_file_exists(s3_handler, mock_s3_client):
    """Test checking if a file exists."""
    # Setup
    s3_link = "s3://test-bucket/test-key"
    
    # Test existing file
    mock_s3_client.head_object.return_value = {}
    assert s3_handler.file_exists(s3_link) == True
    
    # Test non-existing file
    mock_s3_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, 
        "head_object"
    )
    assert s3_handler.file_exists(s3_link) == False

def test_get_presigned_url(s3_handler, mock_s3_client):
    """Test generating a presigned URL."""
    # Setup
    s3_link = "s3://test-bucket/test-key"
    presigned_url = "https://test-bucket.s3.amazonaws.com/test-key?signature=xxx"
    
    # Configure mock
    mock_s3_client.generate_presigned_url.return_value = presigned_url
    
    # Call function
    url = s3_handler.get_presigned_url(s3_link)
    
    # Verify results
    assert url == presigned_url
    
    # Verify the mock was called correctly
    mock_s3_client.generate_presigned_url.assert_called_once_with(
        'get_object',
        Params={'Bucket': 'test-bucket', 'Key': 'test-key'},
        ExpiresIn=3600
    )

def test_list_document_files(s3_handler, mock_s3_client):
    """Test listing all files associated with a document."""
    # Setup
    doc_id = "doc-test-id"
    
    # Mock responses for list_objects_v2
    docs_response = {
        "Contents": [
            {"Key": f"documents/{doc_id}.pdf"},
            {"Key": f"documents/{doc_id}_v2.pdf"}
        ]
    }
    text_response = {
        "Contents": [
            {"Key": f"document-text/{doc_id}/main.txt"},
            {"Key": f"document-text/{doc_id}/summary.txt"}
        ]
    }
    images_response = {
        "Contents": [
            {"Key": f"document-images/{doc_id}/image1.png"},
            {"Key": f"document-images/{doc_id}/image2.png"}
        ]
    }
    graphs_response = {
        "Contents": [
            {"Key": f"document-graphs/{doc_id}.json"}
        ]
    }
    
    # Configure mock
    mock_s3_client.list_objects_v2.side_effect = [
        docs_response, text_response, images_response, graphs_response
    ]
    
    # Call function
    files = s3_handler.list_document_files(doc_id)
    
    # Verify results
    assert len(files['documents']) == 2
    assert len(files['text']) == 2
    assert len(files['images']) == 2
    assert len(files['graphs']) == 1
    
    # Verify list_objects_v2 was called for each bucket
    assert mock_s3_client.list_objects_v2.call_count == 4 

def test_delete_file(s3_handler, mock_s3_client):
    """Test deleting a specific file."""
    # Setup
    s3_link = "s3://test-bucket/test-key"
    
    # Test successful deletion
    mock_s3_client.delete_object.return_value = {}
    result = s3_handler.delete_file(s3_link)
    
    # Verify results
    assert result == True
    mock_s3_client.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="test-key"
    )
    
    # Test deletion with error
    error = ClientError({"Error": {"Code": "TestException", "Message": "Test error"}}, "delete_object")
    mock_s3_client.delete_object.side_effect = error
    result = s3_handler.delete_file(s3_link)
    
    # Verify results
    assert result == False 

def test_download_to_temp_and_process(s3_handler, mock_s3_client, monkeypatch):
    """Test downloading a file to a temporary location and processing it with a callback."""
    # Mock os.path.exists and os.unlink to avoid file system operations
    monkeypatch.setattr('os.path.exists', lambda path: True)
    monkeypatch.setattr('os.unlink', lambda path: None)
    
    # Setup
    bucket = "test-bucket"
    key = "test-key.pdf"
    
    # Mock the callback function
    def process_callback(local_path):
        assert local_path.endswith('.pdf')  # Verify file extension is preserved
        return {"processed": True, "path": local_path}
    
    # Configure mock
    mock_s3_client.download_file.return_value = None
    
    # Call function
    result = s3_handler.download_to_temp_and_process(bucket, key, process_callback)
    
    # Verify results
    assert result["processed"] == True
    assert result["path"].endswith('.pdf')
    
    # Verify the mock was called correctly
    mock_s3_client.download_file.assert_called_once()
    assert mock_s3_client.download_file.call_args[0][0] == bucket
    assert mock_s3_client.download_file.call_args[0][1] == key
    assert mock_s3_client.download_file.call_args[0][2].endswith('.pdf')

def test_download_to_temp_and_process_error(s3_handler, mock_s3_client, monkeypatch):
    """Test error handling in download_to_temp_and_process."""
    # Mock os.path.exists and os.unlink to avoid file system operations
    monkeypatch.setattr('os.path.exists', lambda path: True)
    monkeypatch.setattr('os.unlink', lambda path: None)
    
    # Setup
    bucket = "test-bucket"
    key = "test-key.pdf"
    
    # Configure mock to raise an exception during download
    error = ClientError({"Error": {"Code": "TestException", "Message": "Test error"}}, "download_file")
    mock_s3_client.download_file.side_effect = error
    
    # Mock callback function (should not be called)
    def process_callback(local_path):
        assert False, "Callback should not be called when download fails"
        return {}
    
    # Call function and expect exception
    with pytest.raises(ClientError):
        s3_handler.download_to_temp_and_process(bucket, key, process_callback)
    
def test_process_s3_file(s3_handler, mock_s3_client, monkeypatch):
    """Test processing a file from S3 using process_s3_file."""
    # Mock download_to_temp_and_process to avoid duplicating test logic
    def mock_download_to_temp_and_process(bucket, key, callback, extension=None):
        assert bucket == "test-bucket"
        assert key == "test-key.pdf"
        assert extension == ".pdf"
        return callback("mock_local_path.pdf")
    
    monkeypatch.setattr(s3_handler, 'download_to_temp_and_process', mock_download_to_temp_and_process)
    
    # Setup file_info
    file_info = {
        'Key': 'test-key.pdf',
        'Bucket': 'test-bucket'
    }
    
    # Mock callback function
    def process_callback(local_path):
        assert local_path == "mock_local_path.pdf"
        return {"processed": True, "file": file_info['Key']}
    
    # Call function
    result = s3_handler.process_s3_file(file_info, process_callback)
    
    # Verify results
    assert result["processed"] == True
    assert result["file"] == file_info['Key'] 