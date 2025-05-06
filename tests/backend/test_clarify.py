"""Tests for the query clarifier module."""

import pytest
from backend.clarify import QueryClarifier
from backend.config import BASE_MODEL, MODEL_TEMPERATURE

@pytest.fixture
def clarifier():
    """Create a query clarifier instance for testing."""
    return QueryClarifier()

def test_clarifier_initialization(clarifier):
    """Test query clarifier initialization."""
    assert clarifier.llm is not None
    assert clarifier.llm.model_name == BASE_MODEL
    assert clarifier.llm.temperature == MODEL_TEMPERATURE

def test_clarify_query(clarifier):
    """Test basic query clarification."""
    query = "Tell me about AI"
    clarified = clarifier.clarify(query)
    assert clarified is not None
    assert isinstance(clarified, str)
    assert len(clarified) > len(query)  # Clarified query should be more detailed

def test_interactive_clarify(clarifier):
    """Test interactive query clarification."""
    query = "Tell me about AI"
    clarified = clarifier.interactive_clarify(query, max_turns=2)
    assert clarified is not None
    assert isinstance(clarified, str)
    assert len(clarified) > len(query)

def test_clarify_with_empty_query(clarifier):
    """Test clarification with empty query."""
    with pytest.raises(ValueError):
        clarifier.clarify("")

def test_clarify_with_invalid_query(clarifier):
    """Test clarification with invalid query."""
    with pytest.raises(ValueError):
        clarifier.clarify(None)

def test_interactive_clarify_with_max_turns(clarifier):
    """Test interactive clarification with max turns."""
    query = "Tell me about AI"
    clarified = clarifier.interactive_clarify(query, max_turns=1)
    assert clarified is not None
    assert isinstance(clarified, str) 