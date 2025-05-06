"""
Vector store module for Deep Researcher.
Handles document storage and retrieval using ChromaDB with proper persistence.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

from backend.config import (
    VECTORSTORE_DIR,
    OPENAI_API_KEY,
    VECTORSTORE_CONFIGS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Handles document storage and retrieval using vector embeddings."""
    
    def __init__(self):
        """Initialize the vector store with proper configuration."""
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
                
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            
            # Ensure directory exists
            os.makedirs(VECTORSTORE_CONFIGS["persist_directory"], exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path=VECTORSTORE_CONFIGS["persist_directory"])
            
            # Initialize vector store with basic configuration
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=VECTORSTORE_CONFIGS["collection_name"],
                embedding_function=self.embeddings
            )
            
            logger.info(f"Initialized vector store with collection: {VECTORSTORE_CONFIGS['collection_name']}")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def _process_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process metadata to ensure all values are of supported types."""
        processed = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                # Convert list to comma-separated string
                processed[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, (str, int, float, bool)):
                processed[key] = value
            else:
                # Convert other types to string
                processed[key] = str(value)
        return processed
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store with proper error handling.
        
        Args:
            documents: List of Document objects to add
        """
        try:
            if not documents:
                logger.warning("No documents to add to vector store")
                return
                
            # Process metadata for each document
            processed_docs = []
            for doc in documents:
                try:
                    processed_metadata = self._process_metadata(doc.metadata)
                    processed_doc = Document(
                        page_content=doc.page_content,
                        metadata=processed_metadata
                    )
                    processed_docs.append(processed_doc)
                except Exception as e:
                    logger.error(f"Error processing document metadata: {str(e)}")
                    continue
            
            if not processed_docs:
                logger.warning("No valid documents to add after processing")
                return
            
            # Add processed documents to vector store
            self.vectorstore.add_documents(processed_docs)
            
            logger.info(f"Added {len(processed_docs)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar documents with proper error handling.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store with proper error handling.
        
        Returns:
            Dictionary containing vector store statistics
        """
        try:
            collection = self.vectorstore._collection
            if not collection:
                return {
                    "total_documents": 0,
                    "dimensions": 0,
                    "status": "empty"
                }
            
            stats = {
                "total_documents": collection.count(),
                "dimensions": VECTORSTORE_CONFIGS["embedding_dimensions"],
                "status": "active"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {
                "total_documents": 0,
                "dimensions": 0,
                "status": "error",
                "error": str(e)
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Alias for get_stats to maintain compatibility with KnowledgeBase.
        
        Returns:
            Dictionary containing vector store statistics
        """
        return self.get_stats()
    
    def clear(self) -> None:
        """Clear all documents from the vector store."""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                persist_directory=VECTORSTORE_CONFIGS["persist_directory"],
                collection_name=VECTORSTORE_CONFIGS["collection_name"],
                embedding_function=self.embeddings
            )
            logger.info("Vector store cleared")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise