"""
Document loading and processing module for RAG systems.
"""
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.documents import Document

class DocumentProcessor:
    """
    A class for loading and processing documents for RAG systems.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 24):
        """
        Initialize the DocumentProcessor.
        
        Args:
            chunk_size: The size of text chunks for splitting documents
            chunk_overlap: The overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def split_documents(self, documents: List[Document], limit: Optional[int] = None) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to split
            limit: Optional limit on the number of documents to process
            
        Returns:
            List of split Document objects
        """
        if limit:
            documents = documents[:limit]
        return self.text_splitter.split_documents(documents)
    
    def process_pdf(self, pdf_path: str, limit: Optional[int] = None) -> List[Document]:
        """
        Load and process a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            limit: Optional limit on the number of documents to process
            
        Returns:
            List of processed Document objects
        """
        raw_documents = self.load_pdf(pdf_path)
        return self.split_documents(raw_documents, limit)
