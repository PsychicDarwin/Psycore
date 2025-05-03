"""
ChromaDB Handler for Psycore

This module provides a handler class for interacting with ChromaDB
for multimodal vector storage and retrieval with Langchain integration.
"""

import os
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, BinaryIO
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load environment variables
load_dotenv()

class ChromaHandler:
    """Handler for ChromaDB operations related to multimodal document embeddings."""
    
    def __init__(self, 
                text_embedding_model: Optional[Embeddings] = None,
                image_embedding_model: Optional[Embeddings] = None):
        """
        Initialize the ChromaDB handler.
        
        Args:
            text_embedding_model: Optional LangChain embeddings model for text
            image_embedding_model: Optional LangChain embeddings model for images
        """
        # Get ChromaDB settings from environment variables
        self.chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        self.chroma_port = int(os.getenv('CHROMA_PORT', '8000'))
        
        # Initialize ChromaDB client
        self.client = chromadb.HttpClient(
            host=self.chroma_host,
            port=self.chroma_port,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Store the embedding models
        self.text_embedding_model = text_embedding_model
        self.image_embedding_model = image_embedding_model
        
        # Initialize CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Collection names
        self.document_collection_name = "document_embeddings"
        self.image_collection_name = "image_embeddings"
        self.text_collection_name = "text_embeddings"
        self.multimodal_collection_name = "multimodal_embeddings"
        
        # Initialize collections if they don't exist
        self._init_collections()
    
    def _init_collections(self):
        """Initialize ChromaDB collections if they don't exist."""
        try:
            # Get existing collections
            existing_collections = [c.name for c in self.client.list_collections()]
            
            # Create document embeddings collection if it doesn't exist
            if self.document_collection_name not in existing_collections:
                self.client.create_collection(self.document_collection_name)
            
            # Create image embeddings collection if it doesn't exist
            if self.image_collection_name not in existing_collections:
                self.client.create_collection(self.image_collection_name)
            
            # Create text embeddings collection if it doesn't exist
            if self.text_collection_name not in existing_collections:
                self.client.create_collection(self.text_collection_name)
                
            # Create multimodal embeddings collection if it doesn't exist
            if self.multimodal_collection_name not in existing_collections:
                self.client.create_collection(
                    name=self.multimodal_collection_name,
                    metadata={"hnsw:space": "cosine"}  # Optimized for multimodal embeddings
                )
        
        except Exception as e:
            print(f"Error initializing ChromaDB collections: {e}")
            raise
    
    def get_document_collection(self):
        """
        Get the document embeddings collection.
        
        Returns:
            ChromaDB collection for document embeddings
        """
        return self.client.get_collection(self.document_collection_name)
    
    def get_image_collection(self):
        """
        Get the image embeddings collection.
        
        Returns:
            ChromaDB collection for image embeddings
        """
        return self.client.get_collection(self.image_collection_name)
    
    def get_text_collection(self):
        """
        Get the text embeddings collection.
        
        Returns:
            ChromaDB collection for text embeddings
        """
        return self.client.get_collection(self.text_collection_name)
    
    def add_document_embedding(self, document_id: str, document_text: str, 
                             metadata: Dict[str, Any] = None,
                             embedding: Optional[List[float]] = None):
        """
        Add document embedding to ChromaDB.
        
        Args:
            document_id: Document identifier
            document_text: Document text content
            metadata: Optional metadata about the document
            embedding: Optional pre-computed embedding vector
        """
        collection = self.get_document_collection()
        
        # If no metadata provided, initialize empty dict
        if metadata is None:
            metadata = {}
        
        # If no embedding provided, compute using CLIP
        if embedding is None:
            embedding = self.get_clip_text_embedding(document_text)
        
        # Add document to collection
        collection.add(
            ids=[document_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[document_text]
        )
    
    def add_image_embedding(self, image_id: str, image_text: str,
                          document_id: str, page_number: int,
                          metadata: Dict[str, Any] = None,
                          embedding: Optional[List[float]] = None,
                          image_data: Optional[Union[Image.Image, BinaryIO]] = None):
        """
        Add image embedding to ChromaDB.
        
        Args:
            image_id: Unique image identifier
            image_text: Text extracted from the image
            document_id: Parent document identifier
            page_number: Page number where the image appears
            metadata: Optional metadata about the image
            embedding: Optional pre-computed embedding vector
            image_data: Optional image data for CLIP embedding
        """
        collection = self.get_image_collection()
        
        # If no metadata provided, initialize with basic info
        if metadata is None:
            metadata = {}
        
        # Always include document_id and page_number in metadata
        metadata.update({
            'document_id': document_id,
            'page_number': page_number
        })
        
        # If no embedding provided and image_data is available, compute using CLIP
        if embedding is None and image_data is not None:
            embedding = self.get_clip_image_embedding(image_data)
        
        # Add image to collection
        collection.add(
            ids=[image_id],
            embeddings=[embedding] if embedding else None,
            metadatas=[metadata],
            documents=[image_text]
        )
    
    def add_text_chunk_embedding(self, chunk_id: str, text_chunk: str,
                               document_id: str, chunk_index: int,
                               metadata: Dict[str, Any] = None,
                               embedding: Optional[List[float]] = None):
        """
        Add text chunk embedding to ChromaDB.
        
        Args:
            chunk_id: Unique chunk identifier
            text_chunk: Text content of the chunk
            document_id: Parent document identifier
            chunk_index: Index of the chunk within the document
            metadata: Optional metadata about the text chunk
            embedding: Optional pre-computed embedding vector
        """
        collection = self.get_text_collection()
        
        # If no metadata provided, initialize with basic info
        if metadata is None:
            metadata = {}
        
        # Always include document_id and chunk_index in metadata
        metadata.update({
            'document_id': document_id,
            'chunk_index': chunk_index
        })
        
        # Add text chunk to collection
        if embedding:
            # Use provided embedding
            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[text_chunk]
            )
        else:
            # Let ChromaDB compute embedding
            collection.add(
                ids=[chunk_id],
                metadatas=[metadata],
                documents=[text_chunk]
            )
    
    def search_documents(self, query_text: str, n_results: int = 5,
                       filter_metadata: Optional[Dict[str, Any]] = None):
        """
        Search for similar documents based on a query.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        collection = self.get_document_collection()
        
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )
    
    def search_images(self, query_text: str, n_results: int = 5,
                    filter_metadata: Optional[Dict[str, Any]] = None):
        """
        Search for similar images based on a query.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        collection = self.get_image_collection()
        
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )
    
    def search_text_chunks(self, query_text: str, n_results: int = 5,
                         filter_metadata: Optional[Dict[str, Any]] = None):
        """
        Search for similar text chunks based on a query.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        collection = self.get_text_collection()
        
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )
    
    def delete_document(self, document_id: str):
        """
        Delete all embeddings related to a document.
        
        Args:
            document_id: Document identifier
        """
        # Delete from document collection
        document_collection = self.get_document_collection()
        document_collection.delete(ids=[document_id])
        
        # Delete all image embeddings for this document
        image_collection = self.get_image_collection()
        image_collection.delete(where={"document_id": document_id})
        
        # Delete all text chunk embeddings for this document
        text_collection = self.get_text_collection()
        text_collection.delete(where={"document_id": document_id})
        
        # Delete from multimodal collection
        multimodal_collection = self.get_multimodal_collection()
        multimodal_collection.delete(where={"document_id": document_id})
    
    def get_langchain_vectorstore(self, collection_name: str) -> Chroma:
        """
        Get a LangChain Chroma vectorstore for a specific collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            LangChain Chroma vectorstore
        """
        if self.text_embedding_model is None:
            raise ValueError("No embedding model provided for LangChain integration")
        
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.text_embedding_model
        )
    
    def get_document_vectorstore(self) -> Chroma:
        """
        Get LangChain vectorstore for document embeddings.
        
        Returns:
            LangChain Chroma vectorstore for documents
        """
        return self.get_langchain_vectorstore(self.document_collection_name)
    
    def get_image_vectorstore(self) -> Chroma:
        """
        Get LangChain vectorstore for image embeddings.
        
        Returns:
            LangChain Chroma vectorstore for images
        """
        return self.get_langchain_vectorstore(self.image_collection_name)
    
    def get_text_vectorstore(self) -> Chroma:
        """
        Get LangChain vectorstore for text chunk embeddings.
        
        Returns:
            LangChain Chroma vectorstore for text chunks
        """
        return self.get_langchain_vectorstore(self.text_collection_name)
    
    def get_multimodal_collection(self):
        """
        Get the multimodal embeddings collection.
        
        Returns:
            ChromaDB collection for multimodal embeddings
        """
        return self.client.get_collection(self.multimodal_collection_name)
    
    def get_multimodal_vectorstore(self) -> Chroma:
        """
        Get LangChain vectorstore for multimodal embeddings.
        
        Returns:
            LangChain Chroma vectorstore for multimodal data
        """
        if self.text_embedding_model is None:
            raise ValueError("No text embedding model provided for LangChain integration")
        
        return self.get_langchain_vectorstore(self.multimodal_collection_name)
    
    def add_image_with_binary(self, image_id: str, image_binary: BinaryIO,
                            document_id: str, page_number: int,
                            text_description: str = "",
                            metadata: Dict[str, Any] = None):
        """
        Add image embedding to ChromaDB using the binary image data.
        
        Args:
            image_id: Unique image identifier
            image_binary: Binary image data
            document_id: Parent document identifier
            page_number: Page number where the image appears
            text_description: Optional text description of the image
            metadata: Optional metadata about the image
        """
        if self.image_embedding_model is None:
            raise ValueError("No image embedding model provided for image embedding")
            
        collection = self.get_image_collection()
        
        # If no metadata provided, initialize with basic info
        if metadata is None:
            metadata = {}
        
        # Always include document_id and page_number in metadata
        metadata.update({
            'document_id': document_id,
            'page_number': page_number,
            'content_type': 'image'
        })
        
        # Convert image to base64 for ChromaDB storage
        if hasattr(image_binary, 'read'):
            # If image_binary is a file-like object
            image_data = image_binary.read()
        else:
            # If image_binary is already bytes
            image_data = image_binary
            
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Store in the collection
        collection.add(
            ids=[image_id],
            metadatas=[metadata],
            documents=[text_description if text_description else f"Image from document {document_id}, page {page_number}"]
        )
        
        # Also add to multimodal collection if we have both embedding models
        if self.text_embedding_model and self.image_embedding_model:
            multimodal_collection = self.get_multimodal_collection()
            
            # Combine text description with image data
            multimodal_collection.add(
                ids=[f"multimodal_{image_id}"],
                metadatas=[metadata],
                documents=[text_description if text_description else f"Image from document {document_id}, page {page_number}"]
            )
    
    def add_multimodal_item(self, item_id: str, text_content: str,
                          image_binary: Optional[BinaryIO] = None,
                          document_id: str = "", 
                          metadata: Dict[str, Any] = None):
        """
        Add a multimodal item (text + optional image) to ChromaDB.
        
        Args:
            item_id: Unique item identifier
            text_content: Text content or description
            image_binary: Optional binary image data
            document_id: Optional parent document identifier
            metadata: Optional metadata
        """
        if self.text_embedding_model is None:
            raise ValueError("No text embedding model provided for multimodal embedding")
            
        collection = self.get_multimodal_collection()
        
        # If no metadata provided, initialize empty dict
        if metadata is None:
            metadata = {}
        
        # Add document_id to metadata if provided
        if document_id:
            metadata['document_id'] = document_id
        
        # Set content type in metadata
        if image_binary:
            metadata['content_type'] = 'multimodal'
        else:
            metadata['content_type'] = 'text'
        
        # Store in the collection
        collection.add(
            ids=[item_id],
            metadatas=[metadata],
            documents=[text_content]
        )
    
    def similarity_search(self, vectorstore: Chroma, query: str, 
                        k: int = 4, filter: Optional[Dict[str, Any]] = None):
        """
        Perform similarity search using LangChain.
        
        Args:
            vectorstore: LangChain vectorstore to search
            query: Query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of documents and their scores
        """
        return vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
    
    def multimodal_search(self, query_text: str, query_image: Optional[BinaryIO] = None,
                        n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None):
        """
        Perform a multimodal search (text + optional image).
        
        Args:
            query_text: Text query
            query_image: Optional image query as binary data
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        collection = self.get_multimodal_collection()
        
        # Perform search
        return collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )

    def get_clip_text_embedding(self, text: str) -> List[float]:
        """
        Get CLIP text embedding for a given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        inputs = self.clip_processor(
            text=[text], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=77
        )
        with torch.no_grad():
            embedding = self.clip_model.get_text_features(**inputs)
        return (embedding[0] / embedding.norm()).tolist()

    def get_clip_image_embedding(self, image: Union[Image.Image, BinaryIO]) -> List[float]:
        """
        Get CLIP image embedding for a given image.
        
        Args:
            image: PIL Image or binary image data
            
        Returns:
            Normalized embedding vector
        """
        if isinstance(image, BinaryIO):
            image = Image.open(image).convert("RGB")
        
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        return (embedding[0] / embedding.norm()).tolist()
