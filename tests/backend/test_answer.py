"""Tests for the research answerer module."""

import pytest
from backend.research_answer import ResearchAnswerGenerator
from backend.config import BASE_MODEL, MODEL_TEMPERATURE
from langchain_core.documents import Document

@pytest.fixture
def answerer():
    """Create a research answerer instance for testing."""
    return ResearchAnswerGenerator()

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

def test_answerer_initialization(answerer):
    """Test research answerer initialization."""
    assert answerer.model is not None
    assert answerer.model.model_name == BASE_MODEL
    assert answerer.model.temperature == MODEL_TEMPERATURE

def test_generate_answer(answerer, test_documents):
    """Test answer generation."""
    query = "What is machine learning?"
    answer = answerer.generate_answer(query, test_documents)
    
    assert answer is not None
    assert isinstance(answer, dict)
    assert "answer" in answer
    assert "sources" in answer
    assert isinstance(answer["answer"], str)
    assert isinstance(answer["sources"], list)
    assert len(answer["sources"]) > 0

def test_generate_answer_with_empty_query(answerer, test_documents):
    """Test answer generation with empty query."""
    with pytest.raises(ValueError):
        answerer.generate_answer("", test_documents)

def test_generate_answer_with_invalid_query(answerer, test_documents):
    """Test answer generation with invalid query."""
    with pytest.raises(ValueError):
        answerer.generate_answer(None, test_documents)

def test_generate_answer_with_empty_documents(answerer):
    """Test answer generation with empty documents."""
    query = "What is machine learning?"
    with pytest.raises(ValueError):
        answerer.generate_answer(query, [])

def test_generate_answer_with_invalid_documents(answerer):
    """Test answer generation with invalid documents."""
    query = "What is machine learning?"
    with pytest.raises(ValueError):
        answerer.generate_answer(query, None)

def test_answer_structure(answerer, test_documents):
    """Test the structure of the generated answer."""
    query = "What is the relationship between machine learning and deep learning?"
    answer = answerer.generate_answer(query, test_documents)
    
    # Check required fields
    required_fields = ["answer", "sources", "confidence"]
    for field in required_fields:
        assert field in answer
    
    # Check answer content
    assert isinstance(answer["answer"], str)
    assert len(answer["answer"]) > 0
    
    # Check sources
    assert isinstance(answer["sources"], list)
    assert len(answer["sources"]) > 0
    for source in answer["sources"]:
        assert "title" in source
        assert "url" in source
        assert "relevance" in source
    
    # Check confidence
    assert isinstance(answer["confidence"], float)
    assert 0 <= answer["confidence"] <= 1 