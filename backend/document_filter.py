"""
Document filtering module for removing irrelevant documents from the knowledge base.
"""

import logging
from typing import List, Dict, Any, Set
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from backend.config import OPENAI_API_KEY, BASE_MODEL
import json

logger = logging.getLogger(__name__)

class DocumentFilter:
    """Filters out irrelevant documents from the knowledge base based on expanded research query."""
    
    def __init__(self):
        """Initialize the document filter."""
        self.llm = ChatOpenAI(
            model=BASE_MODEL,
            temperature=0.0,  # Use 0 temperature for consistent relevance judgments
            openai_api_key=OPENAI_API_KEY
        )
    
    def _evaluate_document_relevance(self, document: Document, expanded_query: Dict[str, Any]) -> bool:
        """
        Evaluate if a document is relevant to the expanded research query.
        
        Args:
            document: The document to evaluate
            expanded_query: Dictionary containing the expanded query details
            
        Returns:
            Boolean indicating if document is relevant
        """
        try:
            # Extract key information from document
            content = document.page_content
            metadata = document.metadata
            
            # Construct prompt for relevance evaluation
            prompt = f"""Evaluate if this document is relevant to the research query.

Research Query: {expanded_query.get('original_query', '')}
Research Focus Areas: {json.dumps(expanded_query.get('focus_areas', []), indent=2)}
Key Topics: {json.dumps(expanded_query.get('key_topics', []), indent=2)}

Document Title: {metadata.get('title', 'No title')}
Document Content: {content[:1000]}  # Limit content length for evaluation

Evaluation Criteria:
1. Document directly addresses one or more focus areas
2. Content is specifically related to key topics
3. Information contributes meaningful insights to the research
4. Document contains concrete, relevant data or findings
5. Content aligns with the research scope

Respond with a JSON object containing:
{
    "is_relevant": boolean,
    "confidence_score": float (0-1),
    "reasoning": string
}"""

            # Get relevance evaluation from LLM
            response = self.llm.invoke(prompt).content
            
            try:
                evaluation = json.loads(response)
                return evaluation.get('is_relevant', False)
            except json.JSONDecodeError:
                logger.error(f"Error parsing LLM response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {str(e)}")
            return False
    
    def filter_documents(self, documents: List[Document], expanded_query: Dict[str, Any], knowledge_base) -> None:
        """
        Filter out irrelevant documents from the knowledge base.
        
        Args:
            documents: List of documents to evaluate
            expanded_query: Dictionary containing expanded query details
            knowledge_base: KnowledgeBase instance to update
        """
        try:
            if not documents:
                logger.warning("No documents to filter")
                return
                
            logger.info(f"Starting document filtering for {len(documents)} documents")
            
            # Track documents to remove
            documents_to_remove = []
            
            # Evaluate each document
            for doc in documents:
                if not self._evaluate_document_relevance(doc, expanded_query):
                    documents_to_remove.append(doc)
            
            # Remove irrelevant documents from knowledge base
            if documents_to_remove:
                # Get unique document IDs to remove
                seen_ids = set()
                unique_ids_to_remove = []
                
                for doc in documents_to_remove:
                    doc_id = doc.metadata.get('id')
                    if doc_id and doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        unique_ids_to_remove.append(doc_id)
                
                # Remove documents from vector store
                if unique_ids_to_remove:
                    try:
                        # Check if documents exist in collection before deleting
                        existing_ids = set()
                        for doc_id in unique_ids_to_remove:
                            try:
                                # Try to get the document to verify it exists
                                knowledge_base.vectorstore._collection.get(ids=[doc_id])
                                existing_ids.add(doc_id)
                            except Exception:
                                continue
                        
                        # Only delete documents that exist
                        if existing_ids:
                            knowledge_base.vectorstore._collection.delete(ids=list(existing_ids))
                            
                    except Exception as e:
                        logger.error(f"Error removing documents from vector store: {str(e)}")
                    
                logger.info(f"Removed {len(documents_to_remove)} irrelevant documents from knowledge base")
            else:
                logger.info("No irrelevant documents found")
            
        except Exception as e:
            logger.error(f"Error filtering documents: {str(e)}")
            raise
    
    def get_filtering_stats(self, original_count: int, final_count: int) -> Dict[str, Any]:
        """
        Get statistics about the filtering process.
        
        Args:
            original_count: Number of documents before filtering
            final_count: Number of documents after filtering
            
        Returns:
            Dictionary containing filtering statistics
        """
        return {
            "original_document_count": original_count,
            "final_document_count": final_count,
            "documents_removed": original_count - final_count,
            "removal_percentage": round((original_count - final_count) / original_count * 100, 2) if original_count > 0 else 0
        } 