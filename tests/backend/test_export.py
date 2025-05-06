"""Tests for the export module."""

import pytest
import os
from pathlib import Path
from backend.export import ReportExporter
from backend.config import EXPORTS_DIR
from langchain_core.documents import Document

@pytest.fixture
def exporter():
    """Create a report exporter instance for testing."""
    return ReportExporter()

@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        Document(
            page_content="Test document content 1",
            metadata={"source": "test1", "title": "Test Doc 1"}
        ),
        Document(
            page_content="Test document content 2",
            metadata={"source": "test2", "title": "Test Doc 2"}
        )
    ]

@pytest.fixture
def test_report_data():
    """Create test report data."""
    return {
        "title": "Test Research Report",
        "query": "What is machine learning?",
        "sections": [
            {
                "title": "Introduction",
                "content": "Introduction content"
            },
            {
                "title": "Main Content",
                "content": "Main content"
            }
        ],
        "sources": [
            {
                "title": "Source 1",
                "url": "http://example.com/1"
            },
            {
                "title": "Source 2",
                "url": "http://example.com/2"
            }
        ]
    }

def test_exporter_initialization(exporter):
    """Test exporter initialization."""
    assert exporter is not None
    assert hasattr(exporter, 'export_pdf')
    assert hasattr(exporter, 'export_markdown')

def test_export_pdf(exporter, test_report_data):
    """Test PDF export functionality."""
    # Create exports directory if it doesn't exist
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    
    # Export PDF
    pdf_path = exporter.export_pdf(test_report_data)
    
    # Verify PDF was created
    assert os.path.exists(pdf_path)
    assert pdf_path.endswith('.pdf')
    
    # Clean up
    os.remove(pdf_path)

def test_export_markdown(exporter, test_report_data):
    """Test Markdown export functionality."""
    # Create exports directory if it doesn't exist
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    
    # Export Markdown
    md_path = exporter.export_markdown(test_report_data)
    
    # Verify Markdown was created
    assert os.path.exists(md_path)
    assert md_path.endswith('.md')
    
    # Clean up
    os.remove(md_path)

def test_export_with_invalid_data(exporter):
    """Test export with invalid data."""
    with pytest.raises(ValueError):
        exporter.export_pdf(None)
    
    with pytest.raises(ValueError):
        exporter.export_pdf({})
    
    with pytest.raises(ValueError):
        exporter.export_markdown(None)
    
    with pytest.raises(ValueError):
        exporter.export_markdown({})

def test_export_with_missing_fields(exporter):
    """Test export with missing required fields."""
    invalid_data = {
        "title": "Test Report",
        # Missing query
        "sections": [],
        "sources": []
    }
    
    with pytest.raises(ValueError):
        exporter.export_pdf(invalid_data)
    
    with pytest.raises(ValueError):
        exporter.export_markdown(invalid_data)

def test_export_directory_creation(exporter, test_report_data):
    """Test export directory creation."""
    # Remove exports directory if it exists
    if os.path.exists(EXPORTS_DIR):
        os.rmdir(EXPORTS_DIR)
    
    # Export should create directory
    pdf_path = exporter.export_pdf(test_report_data)
    
    # Verify directory was created
    assert os.path.exists(EXPORTS_DIR)
    
    # Clean up
    os.remove(pdf_path)
    os.rmdir(EXPORTS_DIR) 