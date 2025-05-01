"""
Document Storage Handler for Psycore

This module provides a comprehensive handler for document storage operations,
integrating DynamoDB, S3, and ChromaDB handling.
"""

import os
import uuid
import json
from typing import Dict, List, Any, BinaryIO, Optional, Union, Tuple
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DocumentStorageHandler:
    """Integrated handler for document storage operations."""
    
    def __init__(self, 
                dynamo_handler=None, 
                s3_handler=None, 
                chroma_handler=None,
                text_embedding_model: Optional[Embeddings] = None,
                image_embedding_model: Optional[Embeddings] = None):
        """
        Initialize the document storage handler.
        
        Args:
            dynamo_handler: Optional DynamoDB handler instance
            s3_handler: Optional S3 handler instance
            chroma_handler: Optional ChromaDB handler instance
            text_embedding_model: Optional LangChain embeddings model for text
            image_embedding_model: Optional LangChain embeddings model for images
        """
        # Import here to avoid circular imports
        from src.data.db_handler import DynamoHandler
        from src.data.s3_handler import S3Handler
        from src.data.chroma_handler import ChromaHandler
        
        # Initialize handlers if not provided
        self.dynamo_handler = dynamo_handler or DynamoHandler()
        self.s3_handler = s3_handler or S3Handler()
        self.chroma_handler = chroma_handler or ChromaHandler(
            text_embedding_model=text_embedding_model,
            image_embedding_model=image_embedding_model
        )
        
        # Store the embedding models
        self.text_embedding_model = text_embedding_model
        self.image_embedding_model = image_embedding_model
    
    def store_document(self, file_obj: BinaryIO, filename: str, 
                     metadata: Dict[str, str] = None) -> str:
        """
        Store a new document in the system.
        
        Args:
            file_obj: File object to upload
            filename: Original filename
            metadata: Optional document metadata (title, author, created_date)
            
        Returns:
            Document ID
        """
        # Upload file to S3
        document_id, s3_link = self.s3_handler.upload_document(file_obj, filename)
        
        # Create entry in DynamoDB
        self.dynamo_handler.create_document_entry(
            document_id=document_id,
            document_s3_link=s3_link,
            metadata=metadata
        )
        
        return document_id
    
    def store_document_text(self, document_id: str, 
                          text_content: str) -> Dict[str, Any]:
        """
        Store extracted text from a document.
        
        Args:
            document_id: Document ID
            text_content: Extracted text content
            
        Returns:
            Updated document entry
        """
        # Upload text to S3
        s3_link = self.s3_handler.upload_document_text(
            document_id=document_id,
            text_content=text_content
        )
        
        # Update DynamoDB entry
        updated_entry = self.dynamo_handler.update_document_summary(
            document_id=document_id,
            text_summary_s3_link=s3_link
        )
        
        # Add to ChromaDB if embedding model available
        if self.text_embedding_model:
            # Get metadata from updated entry
            metadata = updated_entry.get('metadata', {})
            
            # Add document embedding
            self.chroma_handler.add_document_embedding(
                document_id=document_id,
                document_text=text_content,
                metadata=metadata
            )
            
            # Add text chunk embeddings (split into manageable chunks)
            chunks = self._split_text_into_chunks(text_content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                self.chroma_handler.add_text_chunk_embedding(
                    chunk_id=chunk_id,
                    text_chunk=chunk,
                    document_id=document_id,
                    chunk_index=i,
                    metadata=metadata
                )
        
        return updated_entry
    
    def store_document_summary(self, document_id: str, 
                             summary_content: str) -> Dict[str, Any]:
        """
        Store document summary text.
        
        Args:
            document_id: Document ID
            summary_content: Document summary text
            
        Returns:
            Updated document entry
        """
        # Upload summary to S3
        s3_link = self.s3_handler.upload_document_summary(
            document_id=document_id,
            summary_content=summary_content
        )
        
        # Update DynamoDB entry
        updated_entry = self.dynamo_handler.update_document_summary(
            document_id=document_id,
            text_summary_s3_link=s3_link
        )
        
        return updated_entry
    
    def store_document_image(self, document_id: str, image_data: BinaryIO,
                           page_number: int, text_content: str = "",
                           extension: str = ".png") -> Dict[str, Any]:
        """
        Store an image extracted from a document.
        
        Args:
            document_id: Document ID
            image_data: Image binary data
            page_number: Page number where the image was found
            text_content: Optional text extracted from the image
            extension: Image file extension
            
        Returns:
            Updated document entry
        """
        # Generate unique image ID
        image_number = page_number  # Use page number as image number for simplicity
        image_id = f"{document_id}_image_{image_number}"
        
        # Upload image to S3
        image_s3_link = self.s3_handler.upload_image(
            document_id=document_id,
            image_data=image_data,
            image_number=image_number,
            extension=extension
        )
        
        # Upload image text to S3 if provided
        if text_content:
            self.s3_handler.upload_image_text(
                document_id=document_id,
                text_content=text_content,
                image_number=image_number
            )
        
        # Update DynamoDB entry
        updated_entry = self.dynamo_handler.add_image_to_document(
            document_id=document_id,
            page_number=page_number,
            image_s3_link=image_s3_link,
            text_summary=text_content
        )
        
        # Store a copy of the image data for ChromaDB
        image_data_copy = image_data
        if hasattr(image_data, 'seek') and callable(image_data.seek):
            image_data.seek(0)  # Reset file pointer to beginning
            image_data_copy = image_data
        
        # Add to ChromaDB if embedding models available
        if self.text_embedding_model and text_content:
            # Get metadata from the document
            document = self.dynamo_handler.get_document(document_id)
            metadata = document.get('metadata', {})
            
            # Add image text embedding
            self.chroma_handler.add_image_embedding(
                image_id=image_id,
                image_text=text_content,
                document_id=document_id,
                page_number=page_number,
                metadata=metadata
            )
        
        # If we have image embedding model, store the binary data too
        if self.image_embedding_model:
            # Get metadata from the document
            document = self.dynamo_handler.get_document(document_id)
            metadata = document.get('metadata', {})
            
            # Add image with binary data
            self.chroma_handler.add_image_with_binary(
                image_id=image_id,
                image_binary=image_data_copy,
                document_id=document_id,
                page_number=page_number,
                text_description=text_content,
                metadata=metadata
            )
            
        # Add as multimodal item if we have both embeddings
        if self.text_embedding_model and self.image_embedding_model and text_content:
            document = self.dynamo_handler.get_document(document_id)
            metadata = document.get('metadata', {})
            metadata.update({
                'page_number': page_number,
                'document_id': document_id
            })
            
            # Add as multimodal item
            self.chroma_handler.add_multimodal_item(
                item_id=f"multimodal_{image_id}",
                text_content=text_content,
                image_binary=image_data_copy,
                document_id=document_id,
                metadata=metadata
            )
        
        return updated_entry
    
    def store_document_graph(self, document_id: str, 
                           graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store document graph data.
        
        Args:
            document_id: Document ID
            graph_data: Graph data as a dictionary
            
        Returns:
            Updated document entry
        """
        # Convert graph data to JSON
        graph_json = json.dumps(graph_data)
        
        # Upload graph to S3
        graph_s3_link = self.s3_handler.upload_graph(
            document_id=document_id,
            graph_json=graph_json
        )
        
        # Update DynamoDB entry
        updated_entry = self.dynamo_handler.update_document_graph(
            document_id=document_id,
            graph_s3_link=graph_s3_link
        )
        
        return updated_entry
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document entry.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document entry
        """
        return self.dynamo_handler.get_document(document_id)
    
    def get_document_content(self, document_id: str) -> bytes:
        """
        Get document binary content.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document binary content
        """
        # Get document entry
        document = self.dynamo_handler.get_document(document_id)
        
        # Get S3 link
        s3_link = document.get('document_s3_link')
        
        # Download file
        return self.s3_handler.download_file(s3_link)
    
    def get_document_text(self, document_id: str) -> str:
        """
        Get document text content.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document text content
        """
        # Get document entry
        document = self.dynamo_handler.get_document(document_id)
        
        # Get S3 link
        s3_link = document.get('text_summary_s3_link')
        
        # Download and decode text
        return self.s3_handler.download_text(s3_link)
    
    def get_document_graph(self, document_id: str) -> Dict[str, Any]:
        """
        Get document graph data.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document graph data
        """
        # Get document entry
        document = self.dynamo_handler.get_document(document_id)
        
        # Get S3 link
        s3_link = document.get('graph_s3_link')
        
        # Download and parse JSON
        graph_text = self.s3_handler.download_text(s3_link)
        return json.loads(graph_text)
    
    def get_document_image(self, document_id: str, 
                         image_number: int) -> bytes:
        """
        Get document image binary content.
        
        Args:
            document_id: Document ID
            image_number: Image number
            
        Returns:
            Image binary content
        """
        # Get document entry
        document = self.dynamo_handler.get_document(document_id)
        
        # Find the image
        for image in document.get('images', []):
            if image.get('page_number') == image_number:
                s3_link = image.get('image_s3_link')
                return self.s3_handler.download_file(s3_link)
        
        # Image not found
        raise ValueError(f"Image {image_number} not found in document {document_id}")
    
    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all documents.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of document entries
        """
        return self.dynamo_handler.list_documents(limit)
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """
        Delete a document and all associated files.
        
        Args:
            document_id: Document ID
            
        Returns:
            Deleted document entry
        """
        # Get document entry
        document = self.dynamo_handler.get_document(document_id)
        
        # Delete from ChromaDB
        self.chroma_handler.delete_document(document_id)
        
        # Delete S3 files
        self.s3_handler.delete_document_files(document_id)
        
        # Delete DynamoDB entry
        return self.dynamo_handler.delete_document(document_id)
    
    def search_documents(self, query_text: str, 
                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for documents by content similarity.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            
        Returns:
            List of document entries with similarity scores
        """
        # Get document vectorstore
        vectorstore = self.chroma_handler.get_document_vectorstore()
        
        # Perform similarity search
        results = self.chroma_handler.similarity_search(
            vectorstore=vectorstore,
            query=query_text,
            k=limit
        )
        
        # Map results to document entries
        document_ids = [doc.metadata.get('document_id') for doc, _ in results]
        documents = []
        
        for i, doc_id in enumerate(document_ids):
            document = self.dynamo_handler.get_document(doc_id)
            if document:
                # Add similarity score
                document['similarity_score'] = results[i][1]
                documents.append(document)
        
        return documents
    
    def search_document_chunks(self, query_text: str, 
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for document chunks by content similarity.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            
        Returns:
            List of document chunks with similarity scores
        """
        # Get text vectorstore
        vectorstore = self.chroma_handler.get_text_vectorstore()
        
        # Perform similarity search
        results = self.chroma_handler.similarity_search(
            vectorstore=vectorstore,
            query=query_text,
            k=limit
        )
        
        # Map results to chunks with metadata
        chunks = []
        for doc, score in results:
            chunks.append({
                'document_id': doc.metadata.get('document_id'),
                'chunk_index': doc.metadata.get('chunk_index'),
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': score
            })
        
        return chunks
    
    def search_images(self, query_text: str, 
                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for images by content similarity.
        
        Args:
            query_text: Query text
            limit: Maximum number of results
            
        Returns:
            List of image entries with similarity scores
        """
        # Get image vectorstore
        vectorstore = self.chroma_handler.get_image_vectorstore()
        
        # Perform similarity search
        results = self.chroma_handler.similarity_search(
            vectorstore=vectorstore,
            query=query_text,
            k=limit
        )
        
        # Map results to image entries
        images = []
        for doc, score in results:
            document_id = doc.metadata.get('document_id')
            page_number = doc.metadata.get('page_number')
            
            # Get document
            document = self.dynamo_handler.get_document(document_id)
            
            # Find the image
            image_entry = None
            for img in document.get('images', []):
                if img.get('page_number') == page_number:
                    image_entry = img
                    break
            
            if image_entry:
                # Add content and score
                image_entry = image_entry.copy()
                image_entry['document_id'] = document_id
                image_entry['text_content'] = doc.page_content
                image_entry['similarity_score'] = score
                images.append(image_entry)
        
        return images
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, 
                              chunk_overlap: int = 200) -> List[str]:
        """
        Split text into chunks for vector storage.
        
        Args:
            text: Text to split
            chunk_size: Maximum chunk size
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If adding this paragraph would exceed chunk size, store current chunk
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap from previous chunk
                current_chunk = current_chunk[-chunk_overlap:] if chunk_overlap > 0 else ""
            
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        
        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_multimodal_document(self, document_id: str, 
                                   document_content: bytes,
                                   text_content: str, 
                                   image_data_list: List[Tuple[BinaryIO, int, str]],
                                   graph_data: Dict[str, Any] = None,
                                   metadata: Dict[str, str] = None):
        """
        Process a complete multimodal document with text, images, and graph.
        
        Args:
            document_id: Document ID (must already exist in the system)
            document_content: Original document binary content
            text_content: Extracted text content
            image_data_list: List of tuples (image_data, page_number, text_content)
            graph_data: Optional graph data
            metadata: Optional metadata updates
        
        Returns:
            Updated document entry
        """
        # Update document entry with text
        self.store_document_text(document_id, text_content)
        
        # Process all images
        for image_data, page_number, image_text in image_data_list:
            self.store_document_image(
                document_id=document_id,
                image_data=image_data,
                page_number=page_number,
                text_content=image_text
            )
        
        # Store graph if provided
        if graph_data:
            self.store_document_graph(document_id, graph_data)
        
        # Update metadata if provided
        if metadata:
            # Get current document
            document = self.dynamo_handler.get_document(document_id)
            
            # Update metadata
            current_metadata = document.get('metadata', {})
            current_metadata.update(metadata)
            
            # Update document
            self.dynamo_handler.table.update_item(
                Key={'document_id': document_id},
                UpdateExpression="set metadata = :m",
                ExpressionAttributeValues={':m': current_metadata}
            )
        
        # Return updated document
        return self.dynamo_handler.get_document(document_id)
    
    def multimodal_search(self, query_text: str, query_image: Optional[BinaryIO] = None,
                        limit: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a multimodal search using both text and image.
        
        Args:
            query_text: Text query
            query_image: Optional image query
            limit: Maximum number of results
            
        Returns:
            List of search results with similarity scores
        """
        # Check if we have both embedding models
        if not (self.text_embedding_model and self.image_embedding_model):
            raise ValueError("Both text and image embedding models are required for multimodal search")
        
        # Get multimodal vectorstore
        multimodal_results = self.chroma_handler.multimodal_search(
            query_text=query_text,
            query_image=query_image,
            n_results=limit
        )
        
        # Map results to document entries
        results = []
        
        for result in multimodal_results['ids'][0]:
            idx = multimodal_results['ids'][0].index(result)
            distance = multimodal_results['distances'][0][idx]
            metadata = multimodal_results['metadatas'][0][idx]
            document_id = metadata.get('document_id')
            
            # Get the associated document
            document = self.dynamo_handler.get_document(document_id)
            if document:
                # Add similarity score
                result_entry = {
                    'document': document,
                    'similarity_score': 1.0 - distance,  # Convert distance to similarity score
                    'metadata': metadata
                }
                results.append(result_entry)
        
        return results
