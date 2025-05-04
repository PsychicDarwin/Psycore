"""
ChromaDB Handler for Psycore

This module provides a handler class for interacting with ChromaDB
for multimodal vector storage and retrieval with Langchain integration.
"""

import os
import base64
import uuid
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, BinaryIO
import chromadb
from chromadb.config import Settings
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from chromadb.api.types import EmbeddingFunction
from src.credential_manager.LocalCredentials import LocalCredentials
from langchain.text_splitter import TokenTextSplitter


logger = logging.getLogger(__name__)

class ChromaHandler:
    """Handler for ChromaDB operations related to multimodal document embeddings."""
    
    def __init__(self, 
            clip_model_name: str = "openai/clip-vit-base-patch32",
            clip_processor_name: str = "openai/clip-vit-base-patch32",
            collection_name: str = "unified_embeddings"):
        
        """
        Initialize the ChromaDB handler.
        
        Args:
            clip_model_name: Name of the CLIP model to use
            clip_processor_name: Name of the CLIP processor to use
            collection_name: Name of the ChromaDB collection to use
        """
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.HttpClient(
            host=LocalCredentials.get_credential('CHROMADB_HOST').secret_key,
            port=LocalCredentials.get_credential('CHROMADB_PORT').secret_key,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # Initialize CLIP model and processor for multimodal embeddings
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor_name)
    
        
        # Initialize single collection
        self.collection_name = collection_name
        self._init_collection()
        self.chunk_size, self.chunk_overlap = 512, 24
        logger.info("ChromaHandler initialization complete")

    def _init_collection(self):
        """Initialize the ChromaDB collection."""
        try:
            # Get embedding dimension from CLIP model
            # CLIP base model has 512 dimensions for text and 512 for images
            # For multimodal, we'll concatenate them to get 1024 dimensions
            text_dimension = 512
            image_dimension = 512
            multimodal_dimension = text_dimension + image_dimension
            
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(self.collection_name)
            except:
                pass
            
            # Create single collection with proper metadata
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": multimodal_dimension
                }
             )
        
        except Exception as e:
            logger.error(f"Error initializing ChromaDB collection: {e}")
            raise

    def prepare_text_embedding(self, text: str, metadata: Dict[str, Any] = None):
        """Prepare and store text embeddings in the collection."""
        stripped_text = text.strip()
        # Chunk text into doc with langchain
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        chunks = text_splitter.split_text(stripped_text)
        for i, chunk in enumerate(chunks):
            inputs = self.clip_processor(text=[chunk], return_tensors="pt", padding=True, truncation=True, max_length= self.chunk_size)
            with torch.no_grad():
                embedding = self.clip_model.get_text_features(**inputs)
            embedding = embedding[0] / embedding.norm()
            new_metadata = metadata.copy() if metadata else {}
            new_metadata['total_chunks'] = len(chunks)
            new_metadata['chunk_index'] = i
            new_metadata['content_type'] = 'text'
            print(embedding.tolist())
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                ids=[f"text_{uuid.uuid4()}"],
                metadatas=[new_metadata]
            )
        return chunks
    
    def prepare_image_embedding(self, image: Image.Image, metadata: Dict[str, Any] = None, document_data = None, id = None):
        """Prepare and store image embeddings in the collection."""
        inputs = self.clip_processor(images=image, return_tensors="pt", padding=True, truncation=True, max_length= self.chunk_size)
        with torch.no_grad():
            embedding = self.clip_model.get_image_features(**inputs)
        embedding = embedding[0] / embedding.norm()
        documents = document_data
        if documents is None:
            documents = [f"Image file: {image.filename}"]
        ids = id
        if ids is None:
            ids = [f"image_{uuid.uuid4()}"]
        
        new_metadata = metadata.copy() if metadata else {}
        new_metadata['content_type'] = 'image'
        
        self.collection.add(
            documents=documents,
            embeddings=[embedding.tolist()],
            ids=ids,
            metadatas=[new_metadata]
        )

    def search(self, query_text: str, n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None):
        """
        Search for similar items in the collection.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            Search results
        """
        return self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )

    def delete_item(self, item_id: str):
        """
        Delete an item from the collection.
        
        Args:
            item_id: ID of the item to delete
        """
        self.collection.delete(ids=[item_id])

    def get_clip_text_embedding(self, text: str) -> List[float]:
        """Get CLIP text embedding for a given text."""
        return self.text_embedding_function([text])[0]

    def get_clip_image_embedding(self, image: Union[Image.Image, BinaryIO]) -> List[float]:
        """Get CLIP image embedding for a given image."""
        return self.image_embedding_function([image])[0]

    def get_multimodal_embedding(self, text: str, image: Union[Image.Image, BinaryIO]) -> List[float]:
        """Get multimodal embedding by combining text and image features."""
        return self.multimodal_embedding_function([(text, image)])[0]
