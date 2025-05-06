"""Tests for the research planner module."""

import pytest
from backend.planner import ResearchPlanner
from backend.config import BASE_MODEL, MODEL_TEMPERATURE

@pytest.fixture
def planner():
    """Create a research planner instance for testing."""
    return ResearchPlanner()

def test_planner_initialization(planner):
    """Test research planner initialization."""
    assert planner.model is not None
    assert planner.model.model_name == BASE_MODEL
    assert planner.model.temperature == MODEL_TEMPERATURE

def test_generate_outline(planner):
    """Test generating a research outline."""
    query = "Investigate the impact of AI on healthcare"
    outline = planner.generate_outline(query)
    
    assert outline is not None
    assert isinstance(outline, dict)
    assert "title" in outline
    assert "sections" in outline
    assert len(outline["sections"]) > 0
    
    # Check section structure
    section = outline["sections"][0]
    assert "title" in section
    assert "content" in section
    assert isinstance(section["content"], dict)

def test_generate_outline_with_empty_query(planner):
    """Test outline generation with empty query."""
    with pytest.raises(ValueError):
        planner.generate_outline("")

def test_generate_outline_with_invalid_query(planner):
    """Test outline generation with invalid query."""
    with pytest.raises(ValueError):
        planner.generate_outline(None)

def test_outline_structure(planner):
    """Test the structure of the generated outline."""
    query = "Analyze the future of renewable energy"
    outline = planner.generate_outline(query)
    
    # Check required fields
    required_fields = ["title", "sections", "key_questions", "required_sources"]
    for field in required_fields:
        assert field in outline
    
    # Check sections
    assert len(outline["sections"]) >= 3  # Should have at least intro, body, conclusion
    for section in outline["sections"]:
        assert "title" in section
        assert "content" in section
        assert isinstance(section["content"], dict)
    
    # Check key questions
    assert len(outline["key_questions"]) > 0
    assert all(isinstance(q, str) for q in outline["key_questions"])
    
    # Check required sources
    assert len(outline["required_sources"]) > 0
    assert all(isinstance(s, str) for s in outline["required_sources"]) 