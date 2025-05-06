"""Tests for the vectorstore module."""

import pytest
import os
from pathlib import Path
from langchain_core.documents import Document
from backend.vectorstore import VectorStore
from backend.config import VECTORSTORE_CONFIGS

@pytest.fixture
def vector_store():
    """Create a vector store instance for testing."""
    store = VectorStore()
    yield store
    # Cleanup after tests
    store.clear()

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

def test_vector_store_initialization(vector_store):
    """Test vector store initialization."""
    assert vector_store.vectorstore is not None
    assert vector_store.embeddings is not None

def test_add_documents(vector_store, test_documents):
    """Test adding documents to the vector store."""
    vector_store.add_documents(test_documents)
    stats = vector_store.get_stats()
    assert stats["total_documents"] == len(test_documents)

def test_similarity_search(vector_store, test_documents):
    """Test similarity search functionality."""
    vector_store.add_documents(test_documents)
    results = vector_store.similarity_search("machine learning", k=1)
    assert len(results) == 1
    assert "machine learning" in results[0].page_content.lower()

def test_clear(vector_store, test_documents):
    """Test clearing the vector store."""
    vector_store.add_documents(test_documents)
    vector_store.clear()
    stats = vector_store.get_stats()
    assert stats["total_documents"] == 0

def test_get_stats(vector_store, test_documents):
    """Test getting vector store statistics."""
    vector_store.add_documents(test_documents)
    stats = vector_store.get_stats()
    assert "total_documents" in stats
    assert "dimensions" in stats
    assert "status" in stats 