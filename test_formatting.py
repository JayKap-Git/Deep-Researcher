"""
Test script for checking report formatting.
"""

import logging
from backend.report_generator import ReportGenerator
from backend.knowledge_base import KnowledgeBase
from backend.export import ReportExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_report():
    """Create a sample report for testing formatting."""
    return {
        'title': 'Test Report: Quantum Computing Advances',
        'date': '2024-05-07',
        'sections': [
            {
                'section_title': 'Introduction to Quantum Computing',
                'content': 'Quantum computing represents a revolutionary approach to computation that harnesses the principles of quantum mechanics. This section provides an overview of the fundamental concepts.',
                'subsections': [
                    {
                        'subsection_title': 'Basic Quantum Principles',
                        'content': 'Quantum computing relies on quantum bits (qubits) which can exist in multiple states simultaneously due to superposition. This property allows quantum computers to perform certain calculations exponentially faster than classical computers.'
                    },
                    {
                        'subsection_title': 'Current Challenges',
                        'content': 'Despite recent advances, quantum computers face several challenges including decoherence, error correction, and scalability issues. Researchers are actively working on solutions to these problems.'
                    }
                ],
                'citations': [
                    'Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information',
                    'Preskill, J. (2018). Quantum Computing in the NISQ era and beyond'
                ]
            },
            {
                'section_title': 'Recent Developments',
                'content': 'The field of quantum computing has seen significant progress in recent years, with several breakthrough achievements in hardware and software development.',
                'subsections': [
                    {
                        'subsection_title': 'Hardware Advances',
                        'content': 'IBM and Google have achieved major milestones in quantum processor development, with IBM unveiling its 127-qubit Eagle processor and Google demonstrating quantum supremacy.'
                    }
                ],
                'citations': [
                    'Arute, F. et al. (2019). Quantum supremacy using a programmable superconducting processor',
                    'IBM Research (2021). IBM Eagle Quantum Processor Announcement'
                ]
            }
        ]
    }

def main():
    """Test the report formatting."""
    try:
        # Create sample report
        report = create_sample_report()
        
        # Initialize exporter
        exporter = ReportExporter()
        
        # Format report content
        formatted_content = exporter._format_report_content(report)
        
        # Export in different formats
        logger.info("Exporting report in different formats...")
        
        # Export PDF
        pdf_path = exporter.export_pdf(formatted_content, [], "test_format")
        logger.info(f"PDF exported to: {pdf_path}")
        
        # Export DOCX
        docx_path = exporter.export_docx(report, "test_format")
        logger.info(f"DOCX exported to: {docx_path}")
        
        # Export HTML
        html_path = exporter.export_html({"content": formatted_content, "citations": []}, "test_format")
        logger.info(f"HTML exported to: {html_path}")
        
        logger.info("Report formatting test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in formatting test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 