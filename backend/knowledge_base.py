"""
Knowledge base module for storing and retrieving research documents.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from backend.config import (
    OPENAI_API_KEY,
    VECTORSTORE_DIR,
    VECTORSTORE_CONFIGS
)
from backend.gather import DataGatherer, ArxivResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBase:
    """Handles storage and retrieval of research documents."""
    
    def __init__(self):
        """Initialize the knowledge base with vector store."""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY
            )
            
            # Create ChromaDB client settings
            client_settings = chromadb.config.Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=VECTORSTORE_CONFIGS["persist_directory"]
            )
            
            # Initialize vector store
            self.vectorstore = Chroma(
                collection_name=VECTORSTORE_CONFIGS["collection_name"],
                embedding_function=self.embeddings,
                persist_directory=VECTORSTORE_CONFIGS["persist_directory"],
                client_settings=client_settings,
                collection_metadata={"hnsw:space": "cosine", "hnsw:construction_ef": VECTORSTORE_CONFIGS["hnsw_config"]["ef_construction"], "hnsw:search_ef": VECTORSTORE_CONFIGS["hnsw_config"]["ef_search"], "hnsw:M": VECTORSTORE_CONFIGS["hnsw_config"]["M"]}
            )
            
            logger.info(f"Initialized vector store with collection: {VECTORSTORE_CONFIGS['collection_name']}")
            
        except Exception as e:
            logger.error(f"Error initializing knowledge base: {str(e)}")
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
            
            # Persist changes if auto_persist is enabled
            if VECTORSTORE_CONFIGS["auto_persist"]:
                self.vectorstore.persist()
            
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
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary containing knowledge base statistics
        """
        try:
            collection = self.vectorstore._collection
            stats = {
                "total_documents": collection.count(),
                "collection_name": VECTORSTORE_CONFIGS["collection_name"],
                "embedding_dimensions": VECTORSTORE_CONFIGS["embedding_dimensions"]
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {str(e)}")
            raise
    
    def clear(self) -> None:
        """Clear all documents from the knowledge base."""
        try:
            # Get all document IDs
            collection = self.vectorstore._collection
            all_ids = collection.get()["ids"]
            
            if all_ids:
                # Delete all documents
                collection.delete(ids=all_ids)
                
            # Recreate the collection
            self.vectorstore = Chroma(
                collection_name=VECTORSTORE_CONFIGS["collection_name"],
                embedding_function=self.embeddings,
                persist_directory=VECTORSTORE_CONFIGS["persist_directory"]
            )
            
            logger.info("Knowledge base cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    kb = KnowledgeBase()
    stats = kb.get_stats()
    print("Knowledge base stats:", stats) 