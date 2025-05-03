"""
Preprocessing Pipeline Module

This module provides the main preprocessing pipeline functionality,
coordinating S3 operations and document processing.
"""

import os
import json
import time
import logging
import sys
from typing import List, Dict, Optional, Any, Tuple
from io import BytesIO

# Add parent directory to path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.processor import DocumentProcessor
from src.data.s3_handler import S3Handler
from src.data.chroma_handler import ChromaHandler
from transformers import CLIPProcessor, CLIPModel
import torch

try:
    from src.data.db_handler import DynamoHandler
    dynamodb_available = True
except ImportError:
    dynamodb_available = False
    logging.warning("DynamoHandler not available. Document relationships will not be stored in DynamoDB.")

# Configure logging
logger = logging.getLogger('preprocessing')

class PreprocessingPipeline:
    """Main preprocessing pipeline for documents."""
    
    def __init__(self):
        """Initialize the preprocessing pipeline."""
        self.s3_handler = S3Handler()
        self.processor = DocumentProcessor()
        
        # Initialize ChromaDB handler
        self.chroma_handler = ChromaHandler()
        
        # Initialize DynamoDB handler if available
        self.db_handler = None
        if dynamodb_available:
            try:
                self.db_handler = DynamoHandler()
                logger.info("DynamoDB handler initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DynamoDB handler: {e}")
    
    def process_documents(self, bucket_name: Optional[str] = None, 
                         prefix: str = "", 
                         limit: Optional[int] = None,
                         clean_output: bool = False) -> List[Dict]:
        """
        Process documents from an S3 bucket.
        
        Args:
            bucket_name: Name of the bucket to process. If None, uses the documents bucket.
            prefix: Only process objects with this prefix.
            limit: Max number of documents to process. If None, process all.
            clean_output: If True, clean output buckets and DynamoDB before processing.
            
        Returns:
            List of processing results.
        """
        # Clean output buckets and DynamoDB if requested
        if clean_output:
            logger.info("Cleaning output buckets and DynamoDB before processing")
            
            # Clean S3 buckets
            try:
                # Clean text bucket
                text_count = self._clean_bucket(self.s3_handler.text_bucket)
                logger.info(f"Cleaned {text_count} objects from {self.s3_handler.text_bucket}")
                
                # Clean images bucket
                images_count = self._clean_bucket(self.s3_handler.images_bucket)
                logger.info(f"Cleaned {images_count} objects from {self.s3_handler.images_bucket}")
                
                # Clean graphs bucket
                graphs_count = self._clean_bucket(self.s3_handler.graphs_bucket)
                logger.info(f"Cleaned {graphs_count} objects from {self.s3_handler.graphs_bucket}")
                
                logger.info(f"Successfully cleaned {text_count + images_count + graphs_count} objects from S3 buckets")
            except Exception as e:
                logger.error(f"Error cleaning S3 buckets: {e}")
            
            # Clean DynamoDB if available
            if self.db_handler:
                try:
                    # List all documents
                    documents = self.db_handler.list_documents()
                    doc_ids = [doc.get('document_id') for doc in documents if 'document_id' in doc]
                    
                    if doc_ids:
                        logger.info(f"Cleaning {len(doc_ids)} documents from DynamoDB")
                        
                        # Delete documents in batches
                        batch_size = 25  # DynamoDB has limits on batch operations
                        deleted_count = 0
                        
                        for i in range(0, len(doc_ids), batch_size):
                            batch = doc_ids[i:i+batch_size]
                            try:
                                result = self.db_handler.batch_delete_documents(batch)
                                deleted_count += result.get('success_count', 0)
                            except Exception as batch_error:
                                logger.error(f"Error deleting batch from DynamoDB: {batch_error}")
                                # Fall back to individual deletes if batch fails
                                for doc_id in batch:
                                    try:
                                        self.db_handler.delete_document(doc_id)
                                        deleted_count += 1
                                    except Exception as e:
                                        logger.error(f"Error deleting document {doc_id}: {e}")
                        
                        logger.info(f"Successfully deleted {deleted_count} documents from DynamoDB")
                    else:
                        logger.info("No documents found in DynamoDB to clean")
                        
                except Exception as e:
                    logger.error(f"Error cleaning DynamoDB: {e}")
                    logger.warning("Continuing processing despite DynamoDB cleaning error")
        
        # List all files
        files = self._list_files(bucket_name or self.s3_handler.documents_bucket, prefix)
        
        if limit:
            files = files[:limit]
            
        logger.info(f"Starting processing of {len(files)} files")
        
        results = []
        for i, file_info in enumerate(files):
            try:
                logger.info(f"Processing file {i+1}/{len(files)}: {file_info['Key']}")
                result = self.process_file(file_info)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing file {file_info['Key']}: {e}")
                results.append({
                    'original_key': file_info['Key'],
                    'error': str(e)
                })
        
        logger.info(f"Completed processing {len(results)} files")
        return results
    
    def _list_files(self, bucket_name: str, prefix: str = "") -> List[Dict]:
        """List files in an S3 bucket with pagination support."""
        files = []
        
        try:
            paginator = self.s3_handler.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        files.append({
                            'Key': obj['Key'],
                            'Size': obj['Size'],
                            'LastModified': obj['LastModified'],
                            'S3Uri': f"s3://{bucket_name}/{obj['Key']}",
                            'Bucket': bucket_name
                        })
            
            logger.info(f"Found {len(files)} files in bucket: {bucket_name} with prefix: {prefix}")
            return files
        except Exception as e:
            logger.error(f"Error listing files in bucket {bucket_name}: {e}")
            raise
    
    def _clean_bucket(self, bucket_name: str) -> int:
        """Clean a bucket by deleting all objects."""
        try:
            deleted_count = 0
            paginator = self.s3_handler.s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        self.s3_handler.s3.delete_object(
                            Bucket=bucket_name,
                            Key=obj['Key']
                        )
                        deleted_count += 1
            
            # Also clean ChromaDB collections
            try:
                # Get all collections
                collections = self.chroma_handler.client.list_collections()
                
                # Delete all items from each collection
                for collection in collections:
                    # Delete all items by using a where clause that matches all documents
                    collection.delete(where={"document_id": {"$ne": ""}})
                    logger.info(f"Cleaned ChromaDB collection: {collection.name}")
            except Exception as e:
                logger.error(f"Error cleaning ChromaDB collections: {e}")
            
            logger.info(f"Deleted {deleted_count} objects from {bucket_name}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning bucket {bucket_name}: {e}")
            raise
    
    def process_file(self, file_info: Dict) -> Dict:
        """
        Process a single file.
        
        Args:
            file_info: Dictionary containing file information.
            
        Returns:
            Processing result dictionary.
        """
        result = {
            'original_key': file_info['Key'],
            'original_bucket': file_info.get('Bucket'),
            'text_key': None,
            'text_uri': None,
            'metadata_key': None,
            'metadata_uri': None,
            'status': 'success',
            'error': None
        }
        
        try:
            # Process the file using the S3Handler's download_to_temp_and_process method
            # This will handle downloading, processing, and cleanup of temporary files
            processing_result = self.s3_handler.download_to_temp_and_process(
                bucket=file_info.get('Bucket', self.s3_handler.documents_bucket),
                key=file_info['Key'],
                process_callback=lambda local_path: self.processor.process_document(local_path, file_info)
            )
            
            # Extract document_id for subsequent operations
            document_id = processing_result['document_id']
            
            # Upload text content if available
            if processing_result.get('text_content'):
                text_uri = self.s3_handler.upload_document_text(
                    document_id=document_id,
                    text_content=processing_result['text_content']
                )
                text_key = f"document-text/{document_id}/main.txt"
                result['text_key'] = text_key
                result['text_uri'] = text_uri
                
                # Add text embedding to ChromaDB
                try:
                    # Get CLIP text embedding
                    inputs = self.chroma_handler.clip_processor(
                        text=[processing_result['text_content']], 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=77
                    )
                    with torch.no_grad():
                        embedding = self.chroma_handler.clip_model.get_text_features(**inputs)
                    embedding = (embedding[0] / embedding.norm()).tolist()
                    
                    # Add to ChromaDB
                    self.chroma_handler.add_multimodal_item(
                        item_id=document_id,
                        text_content=processing_result['text_content'],
                        document_id=document_id,
                        metadata=processing_result.get('metadata', {})
                    )
                except Exception as e:
                    logger.error(f"Failed to add text embedding to ChromaDB: {e}")
            
            # Handle images if available
            if processing_result.get('images'):
                for img_info in processing_result['images']:
                    try:
                        # Get image data
                        image_data = img_info.get('image_data')
                        if image_data:
                            # Get CLIP image embedding
                            inputs = self.chroma_handler.clip_processor(images=image_data, return_tensors="pt")
                            with torch.no_grad():
                                embedding = self.chroma_handler.clip_model.get_image_features(**inputs)
                            embedding = (embedding[0] / embedding.norm()).tolist()
                            
                            # Add to ChromaDB
                            self.chroma_handler.add_image_embedding(
                                image_id=f"{document_id}_img_{img_info.get('page_number', 0)}",
                                image_text=img_info.get('text_content', ''),
                                document_id=document_id,
                                page_number=img_info.get('page_number', 0),
                                metadata=img_info.get('metadata', {}),
                                embedding=embedding,
                                image_data=image_data
                            )
                            
                            # Upload image to S3
                            image_uri = self.s3_handler.upload_image(
                                document_id=document_id,
                                image_data=image_data,
                                image_number=img_info.get('page_number', 0)
                            )
                            
                            # Upload image text if available
                            if img_info.get('text_content'):
                                self.s3_handler.upload_image_text(
                                    document_id=document_id,
                                    text_content=img_info['text_content'],
                                    image_number=img_info.get('page_number', 0)
                                )
                    except Exception as e:
                        logger.error(f"Failed to process image: {e}")
            
            # Handle graph if available
            if processing_result.get('graph_data'):
                try:
                    # Upload graph to dedicated graph bucket
                    graph_uri = self.s3_handler.upload_graph(
                        document_id=document_id,
                        graph_json=json.dumps(processing_result['graph_data'])
                    )
                    result['graph_uri'] = graph_uri
                except Exception as e:
                    logger.error(f"Failed to upload graph: {e}")
            
            # Upload metadata
            metadata = {
                'document_id': document_id,
                'original_filename': file_info['Key'].split('/')[-1],
                'processing_status': 'success',
                'processing_timestamp': str(time.time()),
                'text_uri': result.get('text_uri'),
                'graph_uri': result.get('graph_uri')
            }
            
            metadata_uri = self.s3_handler.upload_document_text(
                document_id=document_id,
                text_content=json.dumps(metadata),
                file_type="metadata"
            )
            metadata_key = f"document-text/{document_id}/metadata.json"
            result['metadata_key'] = metadata_key
            result['metadata_uri'] = metadata_uri
            
            # Update DynamoDB if available
            if self.db_handler:
                try:
                    # Create or update document entry
                    self.db_handler.create_document_entry(
                        document_id=document_id,
                        document_s3_link=file_info.get('S3Uri', ''),
                        metadata=metadata
                    )
                    
                    # Update with text summary if available
                    if result.get('text_uri'):
                        self.db_handler.update_document_summary(
                            document_id=document_id,
                            text_summary_s3_link=result['text_uri']
                        )
                    
                    # Update with graph if available
                    if result.get('graph_uri'):
                        self.db_handler.update_document_graph(
                            document_id=document_id,
                            graph_s3_link=result['graph_uri']
                        )
                except Exception as e:
                    logger.error(f"Failed to update DynamoDB: {e}")
            
            if processing_result.get('error'):
                result['status'] = 'partial'
                result['error'] = processing_result['error']
                
            return result
            
        except Exception as e:
            logger.error(f"Error in pipeline processing file {file_info['Key']}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            return result 