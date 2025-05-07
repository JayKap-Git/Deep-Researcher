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

IMPORTANT: Your response must be valid JSON with the following structure:
{{
    "is_relevant": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "explanation for the decision"
}}"""

            # Get relevance evaluation from LLM
            response = self.llm.invoke(
                prompt,
                response_format={ "type": "json_object" }  # Force JSON response
            ).content
            
            try:
                evaluation = json.loads(response)
                
                # Validate evaluation structure
                if not isinstance(evaluation, dict):
                    logger.error("Evaluation response is not a dictionary")
                    return False
                    
                if 'is_relevant' not in evaluation:
                    logger.error("Evaluation missing is_relevant field")
                    return False
                    
                if not isinstance(evaluation['is_relevant'], bool):
                    logger.error("is_relevant field is not a boolean")
                    return False
                    
                # Log the evaluation for debugging
                logger.debug(f"Document evaluation: {json.dumps(evaluation, indent=2)}")
                
                return evaluation['is_relevant']
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing evaluation response: {str(e)}")
                logger.error(f"Raw response: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating document relevance: {str(e)}")
            return False
    
    def filter_documents(self, documents: List[Document], expanded_query: Dict[str, Any], knowledge_base) -> List[Document]:
        """
        Filter out irrelevant documents from the knowledge base.
        
        Args:
            documents: List of documents to evaluate
            expanded_query: Dictionary containing expanded query details
            knowledge_base: KnowledgeBase instance to update
            
        Returns:
            List of relevant documents
        """
        try:
            if not documents:
                logger.warning("No documents to filter")
                return []
                
            if not knowledge_base or not hasattr(knowledge_base, 'vectorstore'):
                logger.error("Invalid knowledge base instance")
                return documents
                
            logger.info(f"Starting document filtering for {len(documents)} documents")
            
            # Track relevant documents
            relevant_documents = []
            documents_to_remove = []
            
            # Evaluate each document
            for doc in documents:
                if self._evaluate_document_relevance(doc, expanded_query):
                    relevant_documents.append(doc)
                else:
                    documents_to_remove.append(doc)
            
            # Remove irrelevant documents from knowledge base
            if documents_to_remove:
                # Get unique document IDs to remove
                seen_ids = set()
                unique_ids_to_remove = []
                
                for doc in documents_to_remove:
                    doc_id = doc.metadata.get('id')
                    if doc_id and doc_id not in seen_ids:
                        # Verify the document exists in the vector store before adding to removal list
                        try:
                            if hasattr(knowledge_base.vectorstore, '_collection'):
                                # Check if ID exists in collection
                                if doc_id in knowledge_base.vectorstore._collection.get():
                                    seen_ids.add(doc_id)
                                    unique_ids_to_remove.append(doc_id)
                            else:
                                seen_ids.add(doc_id)
                                unique_ids_to_remove.append(doc_id)
                        except Exception as e:
                            logger.warning(f"Error checking document existence: {str(e)}")
                            continue
                
                # Remove documents from vector store
                if unique_ids_to_remove:
                    try:
                        # Check if vector store has delete method
                        if hasattr(knowledge_base.vectorstore, 'delete'):
                            # Delete in batches to handle potential failures
                            batch_size = 10
                            for i in range(0, len(unique_ids_to_remove), batch_size):
                                batch = unique_ids_to_remove[i:i + batch_size]
                                try:
                                    knowledge_base.vectorstore.delete(ids=batch)
                                except Exception as e:
                                    logger.warning(f"Error deleting batch {i//batch_size + 1}: {str(e)}")
                        elif hasattr(knowledge_base.vectorstore, '_collection'):
                            # Try to delete using collection in batches
                            batch_size = 10
                            for i in range(0, len(unique_ids_to_remove), batch_size):
                                batch = unique_ids_to_remove[i:i + batch_size]
                                try:
                                    knowledge_base.vectorstore._collection.delete(ids=batch)
                                except Exception as e:
                                    logger.warning(f"Error deleting batch {i//batch_size + 1} from collection: {str(e)}")
                        else:
                            logger.warning("Vector store does not support document deletion")
                            
                    except Exception as e:
                        logger.error(f"Error removing documents from vector store: {str(e)}")
                    
                logger.info(f"Removed {len(documents_to_remove)} irrelevant documents from knowledge base")
            else:
                logger.info("No irrelevant documents found")
            
            # Get filtering stats
            stats = self.get_filtering_stats(len(documents), len(relevant_documents))
            logger.info("Document filtering stats:")
            logger.info(f"- Original document count: {stats['original_document_count']}")
            logger.info(f"- Final document count: {stats['final_document_count']}")
            logger.info(f"- Documents removed: {stats['documents_removed']}")
            logger.info(f"- Removal percentage: {stats['removal_percentage']}%")
            
            return relevant_documents
            
        except Exception as e:
            logger.error(f"Error filtering documents: {str(e)}")
            return documents  # Return original documents in case of error
    
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