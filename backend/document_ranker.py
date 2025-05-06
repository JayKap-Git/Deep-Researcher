"""
Document ranking module for the Deep Researcher application.

This module ranks documents based on their relevance to the research query using LLM.
"""

import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentRanker:
    """Handles ranking of documents based on relevance to the query."""
    
    def __init__(self):
        """Initialize the document ranker."""
        self.llm = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
    
    def _format_document(self, doc: Document, index: int) -> str:
        """Format a document for the ranking prompt."""
        metadata = doc.metadata
        return f"""
Document {index}:
Title: {metadata.get('title', 'Untitled')}
Source: {metadata.get('source', 'Unknown')}
URL: {metadata.get('url', 'N/A')}
Content: {doc.page_content[:500]}...  # First 500 characters
"""
    
    def rank_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rank documents based on relevance to the query.
        
        Args:
            query: The research query
            documents: List of documents to rank
            
        Returns:
            List of ranked documents
        """
        try:
            if not documents:
                logger.warning("No documents to rank")
                return []
            
            # Format documents for the prompt
            formatted_docs = "\n".join(
                self._format_document(doc, i) 
                for i, doc in enumerate(documents, 1)
            )
            
            # Create ranking prompt with explicit JSON formatting instructions
            prompt = f"""
You are a document ranking expert. Your task is to rank the following documents based on their relevance to this research query.

Query: {query}

Documents:
{formatted_docs}

For each document, provide:
1. A relevance score (0-100)
2. A brief explanation of why it's relevant (or not)
3. Key points from the document that relate to the query

IMPORTANT: Return ONLY a valid JSON object with this exact structure:
{{
    "rankings": [
        {{
            "document_index": 1,
            "relevance_score": 85,
            "reasoning": "Explains the core concepts clearly with examples",
            "key_points": ["Point 1", "Point 2", "Point 3"]
        }}
    ]
}}

Do not include any additional text, explanations, or markdown formatting. Return ONLY the JSON object.
"""
            
            # Get ranking from LLM
            response = self.llm.invoke(prompt)
            
            try:
                # Clean the response content to ensure it's valid JSON
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                rankings = json.loads(content)
                if "rankings" not in rankings:
                    logger.error("Invalid response format: missing 'rankings' key")
                    return []
                
                # Add document metadata to rankings
                ranked_docs = []
                for rank in rankings["rankings"]:
                    doc_index = rank["document_index"] - 1  # Convert to 0-based index
                    if 0 <= doc_index < len(documents):
                        doc = documents[doc_index]
                        # Add ranking information to metadata
                        doc.metadata.update({
                            "relevance_score": rank["relevance_score"],
                            "reasoning": rank["reasoning"],
                            "key_points": rank["key_points"]
                        })
                        ranked_docs.append(doc)
                
                return ranked_docs
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing LLM response: {str(e)}")
                logger.error(f"Response content: {response.content}")
                return []
            
        except Exception as e:
            logger.error(f"Error ranking documents: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Test with sample documents
    ranker = DocumentRanker()
    
    test_query = "What is quantum computing?"
    test_documents = [
        Document(
            page_content="Quantum computing uses quantum bits or qubits to perform calculations. Unlike classical bits, qubits can exist in superposition.",
            metadata={"title": "Introduction to Quantum Computing", "source": "arxiv", "url": "http://example.com/1"}
        ),
        Document(
            page_content="Classical computers use bits that are either 0 or 1. They process information sequentially.",
            metadata={"title": "Classical Computing Basics", "source": "web", "url": "http://example.com/2"}
        )
    ]
    
    rankings = ranker.rank_documents(test_query, test_documents)
    for doc in rankings:
        print(f"Title: {doc.metadata.get('title', 'Untitled')}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"URL: {doc.metadata.get('url', 'N/A')}")
        print(f"Relevance Score: {doc.metadata.get('relevance_score', 'N/A')}")
        print(f"Reasoning: {doc.metadata.get('reasoning', 'N/A')}")
        print(f"Key Points: {', '.join(doc.metadata.get('key_points', ['N/A']))}")
        print() 