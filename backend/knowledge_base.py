"""
Knowledge base module for managing document storage and retrieval.
"""

import logging
import os
import shutil
import tempfile
from typing import List, Dict, Any
from pathlib import Path
from chromadb import PersistentClient
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from backend.config import (
    OPENAI_API_KEY,
    VECTORSTORE_CONFIGS
)

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Manages document storage and retrieval using ChromaDB."""
    
    def __init__(self):
        """Initialize the knowledge base with ChromaDB."""
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set")
                
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
                model="text-embedding-3-small"
            )
            
            # Initialize ChromaDB client with settings
            self.settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
            
            # Create a temporary directory for ChromaDB
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temporary directory for ChromaDB: {self.temp_dir}")
            
            # Initialize vector store directly
            self.collection_name = VECTORSTORE_CONFIGS["collection_name"]
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.temp_dir
            )
            
            logger.info(f"Initialized vector store with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
            raise
    
    def _delete_vectorstore(self) -> None:
        """Delete the vector store collection."""
        try:
            # Delete the collection by recreating the vector store
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
            
            # Reinitialize vector store with empty collection
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.temp_dir
            )
            
        except Exception as e:
            logger.error(f"Error deleting vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents to add
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return
                
            # Add documents to vector store
            self.vectorstore.add_documents(documents)
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search the knowledge base for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} relevant documents for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base collection.
        
        Returns:
            Dictionary containing collection statistics
        """
        try:
            stats = {
                "total_documents": len(self.vectorstore._collection.get()["ids"]),
                "dimensions": VECTORSTORE_CONFIGS["embedding_dimensions"],
                "status": "active"
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "total_documents": 0,
                "dimensions": 0,
                "status": "error",
                "error": str(e)
            }
    
    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        try:
            self._delete_vectorstore()
            logger.info("Knowledge base cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise
    
    def __del__(self):
        """Clean up temporary directory when the object is destroyed."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()
    stats = kb.get_collection_stats()
    print("Knowledge base stats:", stats) 