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
from chromadb.api.types import EmbeddingFunction

# Load environment variables
load_dotenv()

class CLIPTextEmbeddingFunction(EmbeddingFunction):
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in input:
            inputs = self.clip_processor(
                text=[text], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=77
            )
            with torch.no_grad():
                embedding = self.clip_model.get_text_features(**inputs)
                embedding = embedding[0] / embedding.norm()
                embeddings.append(embedding.tolist())
        return embeddings

class CLIPImageEmbeddingFunction(EmbeddingFunction):
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor

    def __call__(self, input: List[Union[Image.Image, BinaryIO]]) -> List[List[float]]:
        embeddings = []
        for image in input:
            if isinstance(image, BinaryIO):
                image = Image.open(image).convert("RGB")
            
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embedding = self.clip_model.get_image_features(**inputs)
                embedding = embedding[0] / embedding.norm()
                embeddings.append(embedding.tolist())
        return embeddings

class MultimodalEmbeddingFunction(EmbeddingFunction):
    def __init__(self, clip_model, clip_processor):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.text_embedder = CLIPTextEmbeddingFunction(clip_model, clip_processor)
        self.image_embedder = CLIPImageEmbeddingFunction(clip_model, clip_processor)

    def __call__(self, input: List[Union[str, tuple[str, Union[Image.Image, BinaryIO]]]]) -> List[List[float]]:
        embeddings = []
        for item in input:
            if isinstance(item, tuple):
                # Handle multimodal input (text + image)
                text, image = item
                text_embedding = self.text_embedder([text])[0]
                image_embedding = self.image_embedder([image])[0]
                combined_embedding = text_embedding + image_embedding
                combined_embedding = torch.tensor(combined_embedding)
                combined_embedding = combined_embedding / combined_embedding.norm()
                embeddings.append(combined_embedding.tolist())
            else:
                # Handle text-only input
                text_embedding = self.text_embedder([item])[0]
                embeddings.append(text_embedding)
        return embeddings

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
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.HttpClient(
            host="13.42.151.24",
            port=8000,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Initialize CLIP model and processor for multimodal embeddings
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Set up the embedding functions
        self.text_embedding_function = CLIPTextEmbeddingFunction(self.clip_model, self.clip_processor)
        self.image_embedding_function = CLIPImageEmbeddingFunction(self.clip_model, self.clip_processor)
        self.multimodal_embedding_function = MultimodalEmbeddingFunction(self.clip_model, self.clip_processor)
        
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
            # Get embedding dimension from CLIP model
            # CLIP base model has 512 dimensions for text and 512 for images
            # For multimodal, we'll concatenate them to get 1024 dimensions
            text_dimension = 512
            image_dimension = 512
            multimodal_dimension = text_dimension + image_dimension
            
            # Delete all vectors and then destroy existing collections
            existing_collections = [c.name for c in self.client.list_collections()]
            for collection_name in [self.document_collection_name, 
                                 self.image_collection_name, 
                                 self.text_collection_name, 
                                 self.multimodal_collection_name]:
                if collection_name in existing_collections:
                    collection = self.client.get_collection(collection_name)
                    # Delete all vectors in the collection
                    collection.delete(where={"document_id": {"$ne": ""}})
                    # Delete the collection
                    self.client.delete_collection(collection_name)
            
            # Create collections with proper metadata and embedding function
            self.client.create_collection(
                name=self.document_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": text_dimension
                },
                embedding_function=self.text_embedding_function
            )
            
            self.client.create_collection(
                name=self.image_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": image_dimension
                },
                embedding_function=self.image_embedding_function
            )
            
            self.client.create_collection(
                name=self.text_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": text_dimension
                },
                embedding_function=self.text_embedding_function
            )
            
            self.client.create_collection(
                name=self.multimodal_collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": multimodal_dimension
                },
                embedding_function=self.multimodal_embedding_function
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
        
        # Ensure embedding is flattened to the correct dimension
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.flatten().tolist()
        elif isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], torch.Tensor):
            embedding = [e.flatten().tolist() for e in embedding]
        
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
        if self.text_embedding_function is None:
            raise ValueError("No embedding function provided for LangChain integration")
        
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.text_embedding_function
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
        if self.multimodal_embedding_function is None:
            raise ValueError("No multimodal embedding function provided for LangChain integration")
        
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
        if self.image_embedding_function is None:
            raise ValueError("No image embedding function provided for image embedding")
            
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
        
        # Also add to multimodal collection if we have both embedding functions
        if self.text_embedding_function and self.image_embedding_function:
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
            # Get multimodal embedding
            embedding = self.get_multimodal_embedding(text_content, image_binary)
        else:
            metadata['content_type'] = 'text'
            # If no image, just use text embedding
            embedding = self.get_clip_text_embedding(text_content)
        
        # Store in the collection
        collection.add(
            ids=[item_id],
            embeddings=[embedding],
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
        
        # Get query embedding
        if query_image:
            query_embedding = self.get_multimodal_embedding(query_text, query_image)
        else:
            query_embedding = self.get_clip_text_embedding(query_text)
        
        # Perform search using the embedding
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )

    def get_clip_text_embedding(self, text: str) -> List[float]:
        """Get CLIP text embedding for a given text."""
        return self.text_embedding_function([text])[0]

    def get_clip_image_embedding(self, image: Union[Image.Image, BinaryIO]) -> List[float]:
        """Get CLIP image embedding for a given image."""
        return self.image_embedding_function([image])[0]

    def get_multimodal_embedding(self, text: str, image: Union[Image.Image, BinaryIO]) -> List[float]:
        """Get multimodal embedding by combining text and image features."""
        return self.multimodal_embedding_function([(text, image)])[0]
