"""Tests for the knowledge base module."""

import pytest
from langchain_core.documents import Document
from backend.knowledge_base import KnowledgeBase
from backend.vectorstore import VectorStore

@pytest.fixture
def knowledge_base():
    """Create a knowledge base instance for testing."""
    kb = KnowledgeBase()
    yield kb
    # Cleanup after tests
    kb.vector_store.clear()

@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        Document(
            page_content="This is a test document about machine learning.",
            metadata={"source": "test", "title": "ML Basics"}
        ),
        Document(
            page_content="Another test document about deep learning.",
            metadata={"source": "test", "title": "DL Basics"}
        )
    ]

def test_knowledge_base_initialization(knowledge_base):
    """Test knowledge base initialization."""
    assert knowledge_base.vector_store is not None
    assert isinstance(knowledge_base.vector_store, VectorStore)

def test_add_documents(knowledge_base, test_documents):
    """Test adding documents to the knowledge base."""
    knowledge_base.add_documents(test_documents)
    stats = knowledge_base.get_stats()
    assert stats["total_documents"] == len(test_documents)

def test_search(knowledge_base, test_documents):
    """Test searching the knowledge base."""
    knowledge_base.add_documents(test_documents)
    results = knowledge_base.search("machine learning", k=1)
    assert len(results) == 1
    assert "machine learning" in results[0].page_content.lower()

def test_get_stats(knowledge_base, test_documents):
    """Test getting knowledge base statistics."""
    knowledge_base.add_documents(test_documents)
    stats = knowledge_base.get_stats()
    assert "total_documents" in stats
    assert "dimensions" in stats
    assert "status" in stats

def test_clear(knowledge_base, test_documents):
    """Test clearing the knowledge base."""
    knowledge_base.add_documents(test_documents)
    knowledge_base.clear()
    stats = knowledge_base.get_stats()
    assert stats["total_documents"] == 0 