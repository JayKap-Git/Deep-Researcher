"""Tests for the research gatherer module."""

import pytest
from backend.gather import DataGatherer
from backend.config import BASE_MODEL, MODEL_TEMPERATURE, SEARCH_CONFIGS
from langchain_core.documents import Document

@pytest.fixture
def gatherer():
    """Create a research gatherer instance for testing."""
    return DataGatherer()

@pytest.fixture
def test_documents():
    """Create test documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test1", "title": "ML Basics"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "test2", "title": "DL Overview"}
        )
    ]

def test_gatherer_initialization(gatherer):
    """Test research gatherer initialization."""
    assert gatherer.model is not None
    assert gatherer.model.model_name == BASE_MODEL
    assert gatherer.model.temperature == MODEL_TEMPERATURE
    assert gatherer.config == SEARCH_CONFIGS

def test_search_arxiv(gatherer):
    """Test arXiv search functionality."""
    query = "machine learning"
    results = gatherer.search_arxiv(query)
    
    assert results is not None
    assert isinstance(results, list)
    if results:  # If we got results
        assert all(isinstance(doc, Document) for doc in results)
        assert all("arxiv" in doc.metadata["source"] for doc in results)

def test_search_web(gatherer):
    """Test web search functionality."""
    query = "artificial intelligence"
    results = gatherer.search_web(query)
    
    assert results is not None
    assert isinstance(results, list)
    if results:  # If we got results
        assert all(isinstance(doc, Document) for doc in results)
        assert all("url" in doc.metadata for doc in results)

def test_search_google(gatherer):
    """Test Google search functionality."""
    query = "neural networks"
    results = gatherer.search_google(query)
    
    assert results is not None
    assert isinstance(results, list)
    if results:  # If we got results
        assert all(isinstance(doc, Document) for doc in results)
        assert all("url" in doc.metadata for doc in results)

def test_gather_from_all_sources(gatherer):
    """Test gathering from all sources."""
    query = "quantum computing"
    results = gatherer.gather_from_all_sources(query)
    
    assert results is not None
    assert isinstance(results, list)
    if results:  # If we got results
        assert all(isinstance(doc, Document) for doc in results)

def test_gather_with_empty_query(gatherer):
    """Test gathering with empty query."""
    with pytest.raises(ValueError):
        gatherer.gather_from_all_sources("")

def test_gather_with_invalid_query(gatherer):
    """Test gathering with invalid query."""
    with pytest.raises(ValueError):
        gatherer.gather_from_all_sources(None)

def test_document_processing(gatherer, test_documents):
    """Test document processing functionality."""
    processed_docs = gatherer.process_documents(test_documents)
    
    assert processed_docs is not None
    assert isinstance(processed_docs, list)
    assert len(processed_docs) == len(test_documents)
    assert all(isinstance(doc, Document) for doc in processed_docs)
    assert all("processed" in doc.metadata for doc in processed_docs) 