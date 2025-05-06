"""
Data gathering module for the Deep Researcher application.
Focuses exclusively on gathering research papers from arXiv.
"""

import os
import time
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import arxiv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from backend.config import (
    OPENAI_API_KEY,
    BASE_MODEL,
    MODEL_TEMPERATURE,
    RATE_LIMITS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArxivResult:
    """Represents an arXiv paper result."""
    def __init__(self, title: str, authors: List[str], abstract: str, pdf_url: str, published: str, arxiv_id: str):
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.pdf_url = pdf_url
        self.published = published
        self.arxiv_id = arxiv_id

class RateLimiter:
    """Handles rate limiting for arXiv API calls."""
    
    def __init__(self, requests_per_minute: int, retry_after: int):
        self.requests_per_minute = requests_per_minute
        self.retry_after = retry_after
        self.requests = []
        
    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        self.requests = [req for req in self.requests if now - req < 60]
        
        if len(self.requests) >= self.requests_per_minute:
            sleep_time = 60 - (now - self.requests[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                self.requests = []
        
        self.requests.append(now)

class DataGatherer:
    """Handles gathering research data exclusively from arXiv."""
    
    def __init__(self):
        """Initialize the data gatherer."""
        self.llm = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        
        # Initialize rate limiter for arXiv
        self.arxiv_limiter = RateLimiter(
            RATE_LIMITS["arxiv"]["requests_per_minute"],
            RATE_LIMITS["arxiv"]["retry_after"]
        )
    
    def _shorten_query(self, query: str, max_length: int = 300) -> str:
        """Shorten query to meet arXiv API limits while preserving meaning."""
        if len(query) <= max_length:
            return query
            
        # Split into words and keep the most important ones
        words = query.split()
        if len(words) <= 10:  # minimum words to keep
            return query
            
        # Keep the first and last few words to preserve context
        shortened = words[:5] + words[-5:]
        return " ".join(shortened)

    def search_arxiv(self, query: str, max_results: int = 5) -> List[ArxivResult]:
        """Search arXiv for research papers."""
        try:
            self.arxiv_limiter.wait_if_needed()
            
            # Shorten query if needed
            query = self._shorten_query(query)
            logger.info(f"Searching arXiv with query: {query}")
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in client.results(search):
                results.append(ArxivResult(
                    title=result.title,
                    authors=[author.name for author in result.authors],
                    abstract=result.summary,
                    pdf_url=result.pdf_url,
                    published=result.published.strftime("%Y-%m-%d") if result.published else "Unknown",
                    arxiv_id=result.entry_id.split('/')[-1]
                ))
            
            logger.info(f"Found {len(results)} results for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {str(e)}")
            return []
    
    def gather_data(self, topics: Dict[str, Any]) -> List[Document]:
        """
        Gather research data from arXiv based on the generated topics and search queries.
        
        Args:
            topics: Dictionary containing topics organized by section and their search queries
            
        Returns:
            List of gathered documents
        """
        try:
            all_documents = []
            
            # Process each section and its topics
            for section, section_topics in topics["topics_by_section"].items():
                logger.info(f"Gathering data for section: {section}")
                
                for topic in section_topics:
                    search_query = topics["search_queries"].get(topic)
                    if not search_query:
                        logger.warning(f"No search query found for topic: {topic}")
                        continue
                        
                    logger.info(f"Gathering data for topic: {topic}")
                    logger.info(f"Using search query: {search_query}")
                    
                    # Gather from arXiv
                    arxiv_results = self.search_arxiv(search_query)
                    for result in arxiv_results:
                        doc = Document(
                            page_content=result.abstract,
                            metadata={
                                "source": "arxiv",
                                "title": result.title,
                                "authors": ", ".join(result.authors),
                                "url": result.pdf_url,
                                "section": section,
                                "topic": topic,
                                "published": result.published,
                                "arxiv_id": result.arxiv_id
                            }
                        )
                        all_documents.append(doc)
            
            # Process and rank all gathered documents
            processed_docs = self.process_documents(all_documents)
            
            logger.info(f"Successfully gathered {len(processed_docs)} documents across {len(topics['topics_by_section'])} sections")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error gathering data: {str(e)}")
            raise

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and validate gathered documents."""
        if not documents:
            logger.warning("No documents to process")
            return []
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list")
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All documents must be instances of Document")

        try:
            processed_docs = []
            for doc in documents:
                # Validate document content
                if not doc.page_content:
                    logger.warning("Skipping document with empty content")
                    continue
                
                # Add processing metadata
                doc.metadata["processed"] = True
                doc.metadata["processing_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                processed_docs.append(doc)
            
            logger.info(f"Processed {len(processed_docs)} valid documents")
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error in process_documents: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    gatherer = DataGatherer()
    results = gatherer.gather_data({
        "topics_by_section": {
            "Introduction": ["quantum computing basics", "recent developments"],
            "Technical Advances": ["quantum hardware", "quantum algorithms"]
        },
        "search_queries": {
            "quantum computing basics": "quantum computing fundamentals review",
            "recent developments": "latest quantum computing breakthroughs 2023",
            "quantum hardware": "quantum processor development superconducting",
            "quantum algorithms": "quantum algorithm optimization NISQ"
        }
    }) 