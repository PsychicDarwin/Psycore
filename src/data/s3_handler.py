"""
S3 Handler for Psycore

This module provides a handler class for interacting with S3 buckets
for document storage, text extraction, images, and graph storage.
"""

import os
import uuid
import boto3
from typing import Dict, List, Any, BinaryIO, Optional, Tuple
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import tempfile
from src.credential_manager.LocalCredentials import LocalCredentials

# Load environment variables
load_dotenv()

class S3Handler:
    """Handler for S3 operations related to document storage and retrieval."""
    
    def __init__(self):
        """Initialize the S3 handler."""
        # Get AWS credentials
        aws_cred = LocalCredentials.get_credential('AWS_IAM_KEY')
        
        # Set up session with credentials
        session = boto3.Session(
            aws_access_key_id=aws_cred.user_key,
            aws_secret_access_key=aws_cred.secret_key,
            region_name=LocalCredentials.get_credential('AWS_DEFAULT_REGION').secret_key
        )
        self.s3 = session.client('s3')
        
        # Get bucket names from LocalCredentials
        self.documents_bucket = LocalCredentials.get_credential('S3_DOCUMENTS_BUCKET').secret_key
        self.text_bucket = LocalCredentials.get_credential('S3_TEXT_BUCKET').secret_key
        self.images_bucket = LocalCredentials.get_credential('S3_IMAGES_BUCKET').secret_key
        self.graphs_bucket = LocalCredentials.get_credential('S3_GRAPHS_BUCKET').secret_key
    
    def upload_document(self, file_obj: BinaryIO, 
                      original_filename: str) -> Tuple[str, str]:
        """
        Upload an original document to S3.
        
        Args:
            file_obj: File object to upload
            original_filename: Original filename
            
        Returns:
            Tuple of (document_id, s3_link)
        """
        # Generate a unique document ID
        document_id = f"doc-{uuid.uuid4()}"
        
        # Extract file extension
        _, extension = os.path.splitext(original_filename)
        
        # Create S3 key
        key = f"documents/{document_id}{extension}"
        
        try:
            # Upload the file
            self.s3.upload_fileobj(file_obj, self.documents_bucket, key)
            
            # Generate S3 link
            s3_link = f"s3://{self.documents_bucket}/{key}"
            
            return document_id, s3_link
        except ClientError as e:
            print(f"Error uploading document: {e}")
            raise
    
    def upload_document_text(self, document_id: str, text_content: str, 
                           file_type: str = "main") -> str:
        """
        Upload extracted text from a document.
        
        Args:
            document_id: Document identifier
            text_content: Extracted text content
            file_type: Type of text file (main, summary, or image{N})
            
        Returns:    
            S3 link to the uploaded text file
        """
        # Create S3 key based on file type
        key = f"{document_id}/{file_type}.txt"
        
        try:
            # Upload the text content
            self.s3.put_object(
                Bucket=self.text_bucket,
                Key=key,
                Body=text_content.encode('utf-8'),
                ContentType='text/plain'
            )
            
            # Generate S3 link
            s3_link = f"s3://{self.text_bucket}/{key}"
            
            return s3_link
        except ClientError as e:
            print(f"Error uploading document text: {e}")
            raise
    
    def upload_document_summary(self, document_id: str, 
                              summary_content: str) -> str:
        """
        Upload document summary text.
        
        Args:
            document_id: Document identifier
            summary_content: Document summary text
            
        Returns:
            S3 link to the summary file
        """
        return self.upload_document_text(
            document_id, summary_content, file_type="summary")
    
    def upload_image(self, document_id: str, image_data: BinaryIO, 
                   image_number: int, extension: str = ".png") -> str:
        """
        Upload an extracted image from a document.
        
        Args:
            document_id: Document identifier
            image_data: Binary image data
            image_number: Sequential image number
            extension: Image file extension
            
        Returns:
            S3 link to the uploaded image
        """
        # Create S3 key
        key = f"{document_id}/image{image_number}{extension}"
        
        try:
            # Upload the image
            self.s3.upload_fileobj(image_data, self.images_bucket, key)
            
            # Generate S3 link
            s3_link = f"s3://{self.images_bucket}/{key}"
            
            return s3_link
        except ClientError as e:
            print(f"Error uploading image: {e}")
            raise
    
    def upload_image_text(self, document_id: str, text_content: str,
                        image_number: int) -> str:
        """
        Upload text extracted from an image.
        
        Args:
            document_id: Document identifier
            text_content: Extracted text from the image
            image_number: Sequential image number
            
        Returns:
            S3 link to the image text file
        """
        return self.upload_document_text(
            document_id, text_content, file_type=f"image{image_number}")
    
    def upload_graph(self, document_id: str, graph_json: str) -> str:
        """
        Upload document graph JSON.
        
        Args:
            document_id: Document identifier
            graph_json: Graph data in JSON format
            
        Returns:
            S3 link to the graph file
        """
        # Create S3 key
        key = f"{document_id}/graph.json"
        
        try:
            # Upload the JSON content
            self.s3.put_object(
                Bucket=self.graphs_bucket,
                Key=key,
                Body=graph_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            # Generate S3 link
            s3_link = f"s3://{self.graphs_bucket}/{key}"
            
            return s3_link
        except ClientError as e:
            print(f"Error uploading graph: {e}")
            raise
    
    def download_file(self, s3_link: str) -> bytes:
        """
        Download a file from S3.
        
        Args:
            s3_link: S3 link to the file
            
        Returns:
            Binary file content
        """
        # Parse bucket and key from S3 link
        parts = s3_link.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        try:
            # Download the file
            response = self.s3.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            print(f"Error downloading file: {e}")
            raise
    
    def download_to_temp_and_process(self, bucket: str, key: str, process_callback, file_extension: str = None) -> Any:
        """
        Download a file to a temporary location, process it with a callback, and clean up.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            process_callback: Callback function that receives the local file path and returns any result
            file_extension: Optional file extension for the temporary file
            
        Returns:
            The result from the process_callback
        """
        import tempfile
        import os
        
        # Use the file extension from the key if not provided
        if file_extension is None and '.' in key:
            _, file_extension = os.path.splitext(key)
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        local_path = temp_file.name
        temp_file.close()
        
        try:
            # Download the file
            self.s3.download_file(bucket, key, local_path)
            
            # Process the file with the callback
            result = process_callback(local_path)
            
            return result
        except Exception as e:
            # Print error but still ensure cleanup
            print(f"Error processing file {bucket}/{key}: {e}")
            raise
        finally:
            # Always clean up the temporary file
            if os.path.exists(local_path):
                os.unlink(local_path)
                
    def process_s3_file(self, file_info: Dict, process_callback) -> Any:
        """
        Process a file from S3 using the provided callback.
        
        Args:
            file_info: Dictionary with 'Key' and 'Bucket' fields
            process_callback: Callback function that receives the local file path and returns any result
            
        Returns:
            The result from the process_callback
        """
        key = file_info['Key']
        bucket = file_info.get('Bucket')
        
        # Extract file extension
        _, file_extension = os.path.splitext(key)
        
        return self.download_to_temp_and_process(bucket, key, process_callback, file_extension)
    
    def download_text(self, s3_link: str) -> str:
        """
        Download and decode a text file from S3.
        
        Args:
            s3_link: S3 link to the text file
            
        Returns:
            Decoded text content
        """
        binary_content = self.download_file(s3_link)
        return binary_content.decode('utf-8')
    
    def delete_document_files(self, document_id: str) -> Dict[str, List[str]]:
        """
        Delete all files associated with a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary of deleted keys by bucket
        """
        deleted_keys = {
            self.documents_bucket: [],
            self.text_bucket: [],
            self.images_bucket: [], 
            self.graphs_bucket: []
        }
        
        # Delete document file
        try:
            # List and delete document files (could be multiple if versions exist)
            response = self.s3.list_objects_v2(
                Bucket=self.documents_bucket,
                Prefix=f"documents/{document_id}"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    self.s3.delete_object(
                        Bucket=self.documents_bucket,
                        Key=obj['Key']
                    )
                    deleted_keys[self.documents_bucket].append(obj['Key'])
        except ClientError as e:
            print(f"Error deleting document files: {e}")
        
        # Delete text files
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.text_bucket,
                Prefix=f"document-text/{document_id}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    self.s3.delete_object(
                        Bucket=self.text_bucket,
                        Key=obj['Key']
                    )
                    deleted_keys[self.text_bucket].append(obj['Key'])
        except ClientError as e:
            print(f"Error deleting text files: {e}")
        
        # Delete image files
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.images_bucket,
                Prefix=f"document-images/{document_id}/"
            )
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    self.s3.delete_object(
                        Bucket=self.images_bucket,
                        Key=obj['Key']
                    )
                    deleted_keys[self.images_bucket].append(obj['Key'])
        except ClientError as e:
            print(f"Error deleting image files: {e}")
        
        # Delete graph file
        try:
            key = f"document-graphs/{document_id}.json"
            self.s3.delete_object(
                Bucket=self.graphs_bucket,
                Key=key
            )
            deleted_keys[self.graphs_bucket].append(key)
        except ClientError as e:
            print(f"Error deleting graph file: {e}")
        
        return deleted_keys
    
    def delete_file(self, s3_link: str) -> bool:
        """
        Delete a specific file from S3.
        
        Args:
            s3_link: S3 link to the file
            
        Returns:
            True if deleted successfully, False otherwise
        """
        # Parse bucket and key from S3 link
        parts = s3_link.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        try:
            # Delete the file
            self.s3.delete_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            print(f"Error deleting file {s3_link}: {e}")
            return False
    
    def file_exists(self, s3_link: str) -> bool:
        """
        Check if a file exists in S3.
        
        Args:
            s3_link: S3 link to check
            
        Returns:
            True if the file exists, False otherwise
        """
        # Parse bucket and key from S3 link
        parts = s3_link.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    def list_document_files(self, document_id: str) -> Dict[str, List[str]]:
        """
        List all files associated with a document.
        
        Args:
            document_id: Document identifier
            
        Returns:
            Dictionary of files by bucket
        """
        files = {
            'documents': [],
            'text': [],
            'images': [],
            'graphs': []
        }
        
        # List document files
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.documents_bucket,
                Prefix=f"documents/{document_id}"
            )
            
            if 'Contents' in response:
                files['documents'] = [
                    f"s3://{self.documents_bucket}/{obj['Key']}"
                    for obj in response['Contents']
                ]
        except ClientError as e:
            print(f"Error listing document files: {e}")
        
        # List text files
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.text_bucket,
                Prefix=f"document-text/{document_id}/"
            )
            
            if 'Contents' in response:
                files['text'] = [
                    f"s3://{self.text_bucket}/{obj['Key']}"
                    for obj in response['Contents']
                ]
        except ClientError as e:
            print(f"Error listing text files: {e}")
        
        # List image files
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.images_bucket,
                Prefix=f"document-images/{document_id}/"
            )
            
            if 'Contents' in response:
                files['images'] = [
                    f"s3://{self.images_bucket}/{obj['Key']}"
                    for obj in response['Contents']
                ]
        except ClientError as e:
            print(f"Error listing image files: {e}")
        
        # List graph files
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.graphs_bucket,
                Prefix=f"document-graphs/{document_id}.json"
            )
            
            if 'Contents' in response:
                files['graphs'] = [
                    f"s3://{self.graphs_bucket}/{obj['Key']}"
                    for obj in response['Contents']
                ]
        except ClientError as e:
            print(f"Error listing graph files: {e}")
        
        return files
    
    def get_presigned_url(self, s3_link: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for temporary access to a file.
        
        Args:
            s3_link: S3 link to the file
            expiration: Expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL
        """
        # Parse bucket and key from S3 link
        parts = s3_link.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1]
        
        try:
            # Generate presigned URL
            url = self.s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            print(f"Error generating presigned URL: {e}")
            raise