"""
DynamoDB Handler for Psycore

This module provides a handler class for interacting with the DynamoDB table
for document relationships in the Psycore project.
"""

import time
import boto3
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
from botocore.exceptions import ClientError
from src.credential_manager.LocalCredentials import LocalCredentials

# Load environment variables
load_dotenv()

class DynamoHandler:
    """Handler for DynamoDB operations related to document relationships."""
    
    def __init__(self):
        """Initialize the DynamoDB handler."""
        # Get AWS credentials
        aws_cred = LocalCredentials.get_credential('AWS_IAM_KEY')
        
        # Get DynamoDB table name
        self.table_name = LocalCredentials.get_credential('DYNAMODB_DOCUMENT_RELATIONSHIPS_TABLE').secret_key
        
        # Set up session with credentials
        session = boto3.Session(
            aws_access_key_id=aws_cred.user_key,
            aws_secret_access_key=aws_cred.secret_key,
            region_name=LocalCredentials.get_credential('AWS_DEFAULT_REGION').secret_key
        )
        self.dynamodb = session.resource('dynamodb')
        print(f"Using credentials from LocalCredentials for DynamoDB table: {self.table_name}")
        
        self.table = self.dynamodb.Table(self.table_name)
        self.gsi_name = 'CreatedAtIndex'
    
    def create_document_entry(self, document_id: str, document_s3_link: str, 
                             metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Create a new document entry in DynamoDB.
        
        Args:
            document_id: Unique identifier for the document
            document_s3_link: S3 link to the original document
            metadata: Optional document metadata including title, author, created_date
            
        Returns:
            The created item
        """
        timestamp = int(time.time())
        
        # Create item with the schema structure specified
        item = {
            'document_id': document_id,
            'created_at': timestamp,
            'document_s3_link': document_s3_link,
            'images': []
        }
        
        # Add metadata if provided
        if metadata:
            item['metadata'] = metadata
            
        try:
            self.table.put_item(Item=item)
            return item
        except ClientError as e:
            print(f"Error creating document entry: {e}")
            raise
    
    def update_document_summary(self, document_id: str, 
                               text_summary_s3_link: str) -> Dict[str, Any]:
        """
        Update a document entry with the text summary S3 link.
        
        Args:
            document_id: Unique identifier for the document
            text_summary_s3_link: S3 link to the text summary
            
        Returns:
            The updated item
        """
        try:
            response = self.table.update_item(
                Key={'document_id': document_id},
                UpdateExpression="set text_summary_s3_link = :s",
                ExpressionAttributeValues={':s': text_summary_s3_link},
                ReturnValues="ALL_NEW"
            )
            return response.get('Attributes', {})
        except ClientError as e:
            print(f"Error updating document summary: {e}")
            raise
    
    def update_document_graph(self, document_id: str, 
                             graph_s3_link: str) -> Dict[str, Any]:
        """
        Update a document entry with the graph S3 link.
        
        Args:
            document_id: Unique identifier for the document
            graph_s3_link: S3 link to the document graph
            
        Returns:
            The updated item
        """
        try:
            response = self.table.update_item(
                Key={'document_id': document_id},
                UpdateExpression="set graph_s3_link = :g",
                ExpressionAttributeValues={':g': graph_s3_link},
                ReturnValues="ALL_NEW"
            )
            return response.get('Attributes', {})
        except ClientError as e:
            print(f"Error updating document graph: {e}")
            raise
    
    def add_image_to_document(self, document_id: str, page_number: int,
                             image_s3_link: str, text_summary: str = "") -> Dict[str, Any]:
        """
        Add an extracted image to a document entry.
        
        Args:
            document_id: Unique identifier for the document
            page_number: Page number where the image was found
            image_s3_link: S3 link to the extracted image
            text_summary: Optional text summary of the image content
            
        Returns:
            The updated item
        """
        image_item = {
            'page_number': page_number,
            'image_s3_link': image_s3_link,
            'text_summary': text_summary
        }
        
        try:
            response = self.table.update_item(
                Key={'document_id': document_id},
                UpdateExpression="set images = list_append(if_not_exists(images, :empty_list), :i)",
                ExpressionAttributeValues={
                    ':i': [image_item],
                    ':empty_list': []
                },
                ReturnValues="ALL_NEW"
            )
            return response.get('Attributes', {})
        except ClientError as e:
            print(f"Error adding image to document: {e}")
            raise
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a document entry by its ID.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            The document item
        """
        try:
            response = self.table.get_item(Key={'document_id': document_id})
            return response.get('Item', {})
        except ClientError as e:
            print(f"Error getting document: {e}")
            raise
    
    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all document entries, sorted by creation time (newest first).
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of document items
        """
        try:
            response = self.table.scan(Limit=limit)
            items = response.get('Items', [])
            # Sort by created_at in descending order
            items.sort(key=lambda x: x.get('created_at', 0), reverse=True)
            return items
        except ClientError as e:
            print(f"Error listing documents: {e}")
            raise
    
    def query_documents_by_creation_time(self, start_time: int, 
                                        end_time: Optional[int] = None,
                                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query documents by creation time range using the GSI.
        
        Args:
            start_time: Start timestamp (inclusive)
            end_time: Optional end timestamp (inclusive)
            limit: Maximum number of items to return
            
        Returns:
            List of document items
        """
        try:
            if end_time is None:
                end_time = int(time.time())  # Current time
                
            # Using KeyConditionExpression directly
            response = self.table.query(
                IndexName=self.gsi_name,
                KeyConditionExpression=boto3.dynamodb.conditions.Key('created_at').between(start_time, end_time),
                Limit=limit
            )
            return response.get('Items', [])
        except ClientError as e:
            print(f"Error querying documents by creation time: {e}")
            raise
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document entry.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Response from the delete operation
        """
        try:
            response = self.table.delete_item(
                Key={'document_id': document_id},
                ReturnValues="ALL_OLD"
            )
            return response.get('Attributes', {})
        except ClientError as e:
            print(f"Error deleting document: {e}")
            raise
    
    def batch_delete_documents(self, document_ids: List[str]) -> Dict[str, Any]:
        """
        Delete multiple document entries in a batch.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            Summary of the delete operation
        """
        deleted_count = 0
        failed_ids = []
        
        for doc_id in document_ids:
            try:
                self.delete_document(doc_id)
                deleted_count += 1
            except Exception as e:
                failed_ids.append({
                    'document_id': doc_id,
                    'error': str(e)
                })
        
        return {
            'success_count': deleted_count,
            'failed_count': len(failed_ids),
            'failed_ids': failed_ids
        }
    
    def get_document_s3_paths(self, document_id: str) -> Dict[str, Any]:
        """
        Get all S3 paths associated with a document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Dictionary of S3 paths
        """
        document = self.get_document(document_id)
        if not document:
            return {}
        
        s3_paths = {
            'document': document.get('document_s3_link'),
            'text_summary': document.get('text_summary_s3_link'),
            'graph': document.get('graph_s3_link'),
            'images': [img.get('image_s3_link') for img in document.get('images', [])]
        }
        
        return s3_paths