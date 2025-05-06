"""
Main application module for the research pipeline.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from backend.config import (
    OPENAI_API_KEY,
    VECTORSTORE_DIR,
    VECTORSTORE_CONFIGS,
    EXPORTS_DIR
)
from backend.planner import ResearchPlanner
from backend.gather import DataGatherer
from backend.knowledge_base import KnowledgeBase
from backend.report_generator import ReportGenerator
from backend.export import ReportExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchPipeline:
    """Main research pipeline class that orchestrates the entire process."""
    
    def __init__(self):
        """Initialize the research pipeline components."""
        try:
            logger.info("Initializing research pipeline components...")
            self.planner = ResearchPlanner()
            self.gatherer = DataGatherer()
            self.kb = KnowledgeBase()
            self.report_generator = ReportGenerator()
            self.exporter = ReportExporter()
            logger.info("Research pipeline components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing research pipeline: {str(e)}")
            raise
    
    def clear_knowledge_base(self) -> None:
        """Clear the knowledge base and reinitialize it."""
        try:
            self.kb.clear()
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a research query through the pipeline.
        
        Args:
            query: The research query to process
            
        Returns:
            Dictionary containing the results of the pipeline
        """
        try:
            results = {
                "query": query,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "outputs": {}
            }
            
            # Step 1: Clarify query
            logger.info("\nStep 1: Clarifying query...")
            clarified_query = self.planner.clarify_query(query)
            results["outputs"]["clarified_query"] = clarified_query
            logger.info(f"Clarified query: {clarified_query}")
            
            # Step 2: Generate research outline
            logger.info("\nStep 2: Generating research outline...")
            outline = self.planner.generate_outline(clarified_query)
            results["outputs"]["outline"] = outline
            logger.info("Generated outline:")
            logger.info(f"Title: {outline['title']}")
            logger.info("Sections:")
            for section in outline["sections"]:
                logger.info(f"- {section.get('section_title', 'Untitled Section')}")
                if "subsections" in section:
                    for subsection in section["subsections"]:
                        logger.info(f"  * {subsection.get('subsection_title', 'Untitled Subsection')}")
                        if "content" in subsection:
                            logger.info(f"    {subsection['content']}")
            
            # Step 3: Generate research topics
            logger.info("\nStep 3: Generating research topics...")
            topics = self.planner.generate_research_topics(outline)
            topics["original_query"] = query  # Add original query for document filtering
            results["outputs"]["topics"] = topics
            
            # Step 4: Gather documents
            logger.info("\nStep 4: Gathering documents...")
            documents = self.gatherer.gather_data(topics, self.kb)
            
            # Store documents in knowledge base
            if documents:
                self.kb.add_documents(documents)
                results["outputs"]["document_count"] = len(documents)
                logger.info(f"Added {len(documents)} documents to knowledge base")
            else:
                logger.warning("No documents found")
                results["outputs"]["document_count"] = 0
            
            # Step 5: Generate report
            logger.info("\nStep 5: Generating report...")
            report = self.report_generator.generate_report(outline, clarified_query)
            results["outputs"]["report"] = report
            
            # Log report details
            logger.info("\nReport generated successfully!")
            logger.info(f"Report title: {report['title']}")
            logger.info(f"Number of sections: {len(report['sections'])}")
            logger.info(f"Number of citations: {len(report['citations'])}")
            
            # Step 6: Export report
            logger.info("\nStep 6: Exporting report...")
            
            # Format report content for export
            report_content = f"# {report['title']}\n\n"
            for section in report['sections']:
                report_content += f"## {section['title']}\n\n"
                if 'content' in section:
                    report_content += f"{section['content']}\n\n"
                if 'subsections' in section:
                    for subsection in section['subsections']:
                        report_content += f"### {subsection['title']}\n\n"
                        if 'content' in subsection:
                            report_content += f"{subsection['content']}\n\n"
            
            # Export to different formats
            try:
                filename = report['title'].lower().replace(' ', '_')
                pdf_path = self.exporter.export_pdf(report_content, report['citations'], filename)
                docx_path = self.exporter.export_docx(report, filename)
                html_path = self.exporter.export_html(report, filename)
                
                results["outputs"]["export_paths"] = {
                    "pdf": pdf_path,
                    "docx": docx_path,
                    "html": html_path
                }
                
                logger.info(f"Report exported to PDF: {pdf_path}")
                logger.info(f"Report exported to DOCX: {docx_path}")
                logger.info(f"Report exported to HTML: {html_path}")
                
            except Exception as e:
                logger.error(f"Error exporting report: {str(e)}")
                results["status"] = "error"
                results["error"] = f"Export error: {str(e)}"
            
            return results
            
        except Exception as e:
            logger.error(f"Error in research pipeline: {str(e)}")
            return {
                "query": query,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

def main():
    """Main entry point for the research pipeline application."""
    try:
        # Initialize pipeline
        pipeline = ResearchPipeline()
        
        # Clear knowledge base
        pipeline.clear_knowledge_base()
        
        # Get research query
        print("\nWelcome to the Research Pipeline!")
        print("Please enter your research query (e.g., 'What are the latest advancements in quantum computing?')")
        query = input("Your query: ")
        
        # Process query
        results = pipeline.process_query(query)
        
        # Display results
        if results["status"] == "success":
            print("\nResearch pipeline completed successfully!")
            print(f"\nReport exported to:")
            for format, path in results["outputs"]["export_paths"].items():
                print(f"- {format.upper()}: {path}")
        else:
            print(f"\nError in research pipeline: {results.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main() 