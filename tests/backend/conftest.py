"""Test configuration file with common fixtures and settings."""

import pytest
import os
from unittest.mock import Mock, patch
from dotenv import load_dotenv
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

@pytest.fixture(scope="session")
def mock_arxiv_search():
    """Mock arXiv search response."""
    return [
        Document(
            page_content="Machine learning paper content",
            metadata={"source": "arxiv", "title": "ML Paper", "url": "http://arxiv.org/1"}
        ),
        Document(
            page_content="Deep learning paper content",
            metadata={"source": "arxiv", "title": "DL Paper", "url": "http://arxiv.org/2"}
        )
    ]

@pytest.fixture(scope="session")
def mock_web_search():
    """Mock web search response."""
    return [
        Document(
            page_content="Web article about AI",
            metadata={"source": "web", "title": "AI Article", "url": "http://example.com/1"}
        ),
        Document(
            page_content="Web article about ML",
            metadata={"source": "web", "title": "ML Article", "url": "http://example.com/2"}
        )
    ]

@pytest.fixture(scope="session")
def mock_google_search():
    """Mock Google search response."""
    return [
        Document(
            page_content="Google result about AI",
            metadata={"source": "google", "title": "AI Result", "url": "http://google.com/1"}
        ),
        Document(
            page_content="Google result about ML",
            metadata={"source": "google", "title": "ML Result", "url": "http://google.com/2"}
        )
    ]

@pytest.fixture(scope="session", autouse=True)
def setup_mocks(mock_arxiv_search, mock_web_search, mock_google_search):
    """Set up mock implementations for external services."""
    with patch('backend.gather.DataGatherer.search_arxiv', return_value=mock_arxiv_search), \
         patch('backend.gather.DataGatherer.search_web', return_value=mock_web_search), \
         patch('backend.gather.DataGatherer.search_google', return_value=mock_google_search):
        yield

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration settings."""
    return {
        "test_data_dir": "tests/data",
        "max_test_documents": 5,
        "test_timeout": 30,  # seconds
        "test_queries": [
            "What is machine learning?",
            "Explain quantum computing",
            "What are neural networks?",
            "Describe artificial intelligence",
            "What is deep learning?"
        ]
    }

@pytest.fixture(scope="session")
def test_documents():
    """Create a set of test documents."""
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
            page_content="Quantum computing uses quantum bits or qubits.",
            metadata={"source": "test3", "title": "QC Introduction"}
        ),
        Document(
            page_content="Neural networks are inspired by biological neurons.",
            metadata={"source": "test4", "title": "NN Basics"}
        ),
        Document(
            page_content="AI systems can learn from data and make decisions.",
            metadata={"source": "test5", "title": "AI Overview"}
        )
    ]

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up the test environment."""
    # Create test data directory if it doesn't exist
    os.makedirs("tests/data", exist_ok=True)
    
    # Set up any other test environment configurations
    yield
    
    # Clean up after tests
    # Remove test data directory if empty
    try:
        os.rmdir("tests/data")
    except OSError:
        pass  # Directory not empty or doesn't exist 