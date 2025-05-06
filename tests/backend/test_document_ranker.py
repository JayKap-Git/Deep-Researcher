"""Tests for the document ranker module."""

import pytest
from backend.document_ranker import DocumentRanker
from langchain_core.documents import Document
from backend.config import BASE_MODEL, MODEL_TEMPERATURE

@pytest.fixture
def ranker():
    """Create a document ranker instance for testing."""
    return DocumentRanker()

@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test1", "title": "ML Basics"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "test2", "title": "DL Overview"}
        ),
        Document(
            page_content="Natural language processing is a field of AI.",
            metadata={"source": "test3", "title": "NLP Intro"}
        )
    ]

def test_ranker_initialization(ranker):
    """Test document ranker initialization."""
    assert ranker is not None
    assert ranker.model is not None
    assert ranker.model.model_name == BASE_MODEL
    assert ranker.model.temperature == MODEL_TEMPERATURE

def test_rank_documents(ranker, test_documents):
    """Test document ranking functionality."""
    query = "What is machine learning?"
    ranked_docs = ranker.rank_documents(query, test_documents)
    
    assert ranked_docs is not None
    assert isinstance(ranked_docs, list)
    assert len(ranked_docs) == len(test_documents)
    assert all(isinstance(doc, Document) for doc in ranked_docs)
    assert all("relevance_score" in doc.metadata for doc in ranked_docs)

def test_rank_documents_with_empty_query(ranker, test_documents):
    """Test ranking with empty query."""
    with pytest.raises(ValueError):
        ranker.rank_documents("", test_documents)

def test_rank_documents_with_invalid_query(ranker, test_documents):
    """Test ranking with invalid query."""
    with pytest.raises(ValueError):
        ranker.rank_documents(None, test_documents)

def test_rank_documents_with_empty_documents(ranker):
    """Test ranking with empty documents list."""
    with pytest.raises(ValueError):
        ranker.rank_documents("test query", [])

def test_rank_documents_with_invalid_documents(ranker):
    """Test ranking with invalid documents."""
    with pytest.raises(ValueError):
        ranker.rank_documents("test query", None)

def test_rank_documents_with_single_document(ranker):
    """Test ranking with single document."""
    doc = Document(
        page_content="Test content",
        metadata={"source": "test", "title": "Test Doc"}
    )
    ranked_docs = ranker.rank_documents("test query", [doc])
    
    assert len(ranked_docs) == 1
    assert "relevance_score" in ranked_docs[0].metadata

def test_rank_documents_relevance_scores(ranker, test_documents):
    """Test that relevance scores are properly assigned."""
    query = "machine learning"
    ranked_docs = ranker.rank_documents(query, test_documents)
    
    # Check that scores are between 0 and 1
    for doc in ranked_docs:
        score = doc.metadata["relevance_score"]
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    # Check that documents are sorted by relevance
    scores = [doc.metadata["relevance_score"] for doc in ranked_docs]
    assert scores == sorted(scores, reverse=True)

def test_rank_documents_with_similar_content(ranker):
    """Test ranking with documents containing similar content."""
    docs = [
        Document(
            page_content="Machine learning is important",
            metadata={"source": "test1", "title": "Doc 1"}
        ),
        Document(
            page_content="Machine learning is very important",
            metadata={"source": "test2", "title": "Doc 2"}
        ),
        Document(
            page_content="Machine learning is extremely important",
            metadata={"source": "test3", "title": "Doc 3"}
        )
    ]
    
    ranked_docs = ranker.rank_documents("machine learning", docs)
    assert len(ranked_docs) == len(docs)
    assert all("relevance_score" in doc.metadata for doc in ranked_docs) 