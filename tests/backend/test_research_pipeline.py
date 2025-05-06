"""Tests for the research pipeline module."""

import pytest
from backend.research_answer import ResearchAnswerGenerator
from backend.gather import DataGatherer
from backend.clarify import QueryClarifier
from backend.planner import ResearchPlanner
from backend.knowledge_base import KnowledgeBase
from backend.config import BASE_MODEL, MODEL_TEMPERATURE

class ResearchPipeline:
    """Research pipeline that coordinates all components."""
    
    def __init__(self):
        """Initialize the research pipeline."""
        self.clarifier = QueryClarifier()
        self.planner = ResearchPlanner()
        self.gatherer = DataGatherer()
        self.answerer = ResearchAnswerGenerator()
        self.knowledge_base = KnowledgeBase()

    def clarify_query(self, query: str) -> str:
        """Clarify the research query."""
        return self.clarifier.clarify_query(query)

    def plan_research(self, query: str) -> dict:
        """Generate a research plan."""
        return self.planner.generate_outline(query)

    def gather_documents(self, query: str) -> list:
        """Gather documents from various sources."""
        return self.gatherer.gather_from_all_sources(query)

    def generate_answer(self, query: str, documents: list) -> dict:
        """Generate a comprehensive answer."""
        return self.answerer.generate_answer(query, documents)

    def run(self, query: str) -> dict:
        """Run the complete research pipeline."""
        if not query:
            raise ValueError("Query cannot be empty")
        if query is None:
            raise ValueError("Query cannot be None")

        clarified_query = self.clarify_query(query)
        outline = self.plan_research(clarified_query)
        documents = self.gather_documents(clarified_query)
        answer = self.generate_answer(clarified_query, documents)

        return {
            "clarified_query": clarified_query,
            "outline": outline,
            "documents": documents,
            "answer": answer
        }

@pytest.fixture
def pipeline():
    """Create a research pipeline instance for testing."""
    return ResearchPipeline()

def test_pipeline_initialization(pipeline):
    """Test research pipeline initialization."""
    assert pipeline.clarifier is not None
    assert pipeline.planner is not None
    assert pipeline.gatherer is not None
    assert pipeline.answerer is not None
    assert pipeline.knowledge_base is not None

def test_run_pipeline(pipeline):
    """Test running the complete research pipeline."""
    query = "What is the impact of AI on healthcare?"
    result = pipeline.run(query)
    
    assert result is not None
    assert isinstance(result, dict)
    assert "clarified_query" in result
    assert "outline" in result
    assert "documents" in result
    assert "answer" in result
    
    # Check clarified query
    assert isinstance(result["clarified_query"], str)
    assert len(result["clarified_query"]) > len(query)
    
    # Check outline
    assert isinstance(result["outline"], dict)
    assert "title" in result["outline"]
    assert "sections" in result["outline"]
    
    # Check documents
    assert isinstance(result["documents"], list)
    if result["documents"]:  # If we got documents
        assert all(hasattr(doc, "page_content") for doc in result["documents"])
    
    # Check answer
    assert isinstance(result["answer"], dict)
    assert "answer" in result["answer"]
    assert "sources" in result["answer"]

def test_run_pipeline_with_empty_query(pipeline):
    """Test running pipeline with empty query."""
    with pytest.raises(ValueError):
        pipeline.run("")

def test_run_pipeline_with_invalid_query(pipeline):
    """Test running pipeline with invalid query."""
    with pytest.raises(ValueError):
        pipeline.run(None)

def test_pipeline_components(pipeline):
    """Test individual pipeline components."""
    query = "What are the latest developments in quantum computing?"
    
    # Test query clarification
    clarified = pipeline.clarify_query(query)
    assert isinstance(clarified, str)
    assert len(clarified) > len(query)
    
    # Test research planning
    outline = pipeline.plan_research(clarified)
    assert isinstance(outline, dict)
    assert "title" in outline
    assert "sections" in outline
    
    # Test document gathering
    documents = pipeline.gather_documents(clarified)
    assert isinstance(documents, list)
    if documents:  # If we got documents
        assert all(hasattr(doc, "page_content") for doc in documents)
    
    # Test answer generation
    answer = pipeline.generate_answer(clarified, documents)
    assert isinstance(answer, dict)
    assert "answer" in answer
    assert "sources" in answer

def test_pipeline_with_knowledge_base(pipeline):
    """Test pipeline with knowledge base integration."""
    query = "What is machine learning?"
    
    # Run pipeline
    result = pipeline.run(query)
    
    # Check knowledge base integration
    assert pipeline.knowledge_base is not None
    stats = pipeline.knowledge_base.get_stats()
    assert isinstance(stats, dict)
    assert "total_documents" in stats
    assert "dimensions" in stats
    assert "status" in stats 