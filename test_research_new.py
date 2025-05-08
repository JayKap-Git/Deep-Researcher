"""
Test script for the research pipeline.
"""

import os
import logging
import shutil
import stat
from datetime import datetime
from pathlib import Path
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
from backend.clarify import QueryClarifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory_permissions(directory: str):
    """Ensure the directory has proper read/write permissions."""
    try:
        if os.path.exists(directory):
            os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            for root, dirs, files in os.walk(directory):
                for d in dirs:
                    os.chmod(os.path.join(root, d), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                for f in files:
                    os.chmod(os.path.join(root, f), stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
        logger.info(f"Set permissions for directory: {directory}")
    except Exception as e:
        logger.error(f"Error setting directory permissions: {str(e)}")
        raise

def clear_vectorstore_dir():
    """Clear the vectorstore directory to ensure a fresh start."""
    try:
        if os.path.exists(VECTORSTORE_DIR):
            ensure_directory_permissions(VECTORSTORE_DIR)
            shutil.rmtree(VECTORSTORE_DIR)
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        ensure_directory_permissions(VECTORSTORE_DIR)
        logger.info("Vectorstore directory cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing vectorstore directory: {str(e)}")
        raise

def clear_knowledge_base():
    """Clear the knowledge base and reinitialize it."""
    try:
        # Clear vectorstore directory first
        clear_vectorstore_dir()
        
        # Initialize knowledge base
        kb = KnowledgeBase()
        
        # Try to clear the knowledge base
        kb.clear()
        logger.info("Knowledge base cleared successfully")
        
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise

def get_clarified_query() -> str:
    """
    Get a clarified research query through interactive conversation.
    
    Returns:
        str: The final clarified query
    """
    clarifier = QueryClarifier()
    print("\nWelcome to the Research Pipeline!")
    print("Please enter your research query (e.g., 'What are the latest advancements in quantum computing?')")
    
    initial_query = input("Your query: ").strip()
    if not initial_query:
        print("Query cannot be empty. Please try again.")
        return get_clarified_query()
    
    logger.info(f"\nInitial query: {initial_query}")
    
    # Use interactive clarification
    final_query = clarifier.interactive_clarify(initial_query)
    
    if not final_query:
        print("\nQuery clarification was incomplete or cancelled.")
        print("Would you like to try again? (yes/no)")
        if input().lower().startswith('y'):
            return get_clarified_query()
        else:
            raise ValueError("Research cancelled by user")
    
    return final_query

def main():
    """Test the research pipeline."""
    try:
        # Initialize components
        logger.info("Initializing research pipeline components...")
        planner = ResearchPlanner()
        gatherer = DataGatherer()
        kb = KnowledgeBase()
        report_generator = ReportGenerator()
        exporter = ReportExporter()
        
        # Clear knowledge base
        clear_knowledge_base()
        
        # Get and clarify research query
        query = get_clarified_query()
        logger.info(f"\nTesting research pipeline with clarified query: {query}")
        
        # Step 1: Generate research outline
        logger.info("\nStep 1: Generating research outline...")
        outline = planner.generate_outline(query)
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
        
        # Step 2: Generate research topics
        logger.info("\nStep 2: Generating research topics...")
        topics = planner.generate_research_topics(outline)
        
        # Add original query to topics for document filtering
        topics["original_query"] = query
        
        # Step 3: Gather documents
        logger.info("\nStep 3: Gathering documents...")
        documents = gatherer.gather_data(topics, kb)  # Pass knowledge base instance
        
        # Store documents in knowledge base
        if documents:
            kb.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to knowledge base")
        else:
            logger.warning("No documents found")
        
        # Step 4: Generate report
        # logger.info("\nStep 4: Generating report...")
        # report = report_generator.generate_report(outline, kb)
        
        # # Log report details
        # logger.info("\nReport generated successfully!")
        # logger.info(f"Report title: {report['title']}")
        # logger.info(f"Number of sections: {len(report['sections'])}")
        # logger.info(f"Number of citations: {len(report.get('citations', []))}")
        
        # # Print report sections
        # logger.info("\nReport sections:")
        # for section in report['sections']:
        #     logger.info(f"\nSection: {section['section_title']}")
        #     if 'content' in section:
        #         logger.info(f"Content: {section['content']}")
        #     if 'subsections' in section:
        #         for subsection in section['subsections']:
        #             logger.info(f"\n  Subsection: {subsection['subsection_title']}")
        #             if 'content' in subsection:
        #                 logger.info(f"  Content: {subsection['content']}")
        
        # # Step 5: Export report
        # logger.info("\nStep 5: Exporting report...")
        
        # # Format report content for export
        # report_content = f"# {report['title']}\n\n"
        # for section in report['sections']:
        #     report_content += f"## {section['section_title']}\n\n"
        #     if 'content' in section:
        #         report_content += f"{section['content']}\n\n"
        #     if 'subsections' in section:
        #         for subsection in section['subsections']:
        #             report_content += f"### {subsection['subsection_title']}\n\n"
        #             if 'content' in subsection:
        #                 report_content += f"{subsection['content']}\n\n"
        
        # Export to different formats
        try:
            filename = report['title'].lower().replace(' ', '_')
            pdf_path = exporter.export_pdf(report_content, report.get('citations', []), filename)
            logger.info(f"Report exported to PDF: {pdf_path}")
            
            docx_path = exporter.export_docx(report, filename)
            logger.info(f"Report exported to DOCX: {docx_path}")
            
            html_path = exporter.export_html(report, filename)
            logger.info(f"Report exported to HTML: {html_path}")
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
        
        print("\nResearch pipeline test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in research pipeline test: {str(e)}")
        raise

if __name__ == "__main__":
    main() 