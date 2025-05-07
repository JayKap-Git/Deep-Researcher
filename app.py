"""
Streamlit application for the research pipeline.
"""

import os
import logging
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import time

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

class ResearchPipeline:
    """Main research pipeline class that orchestrates the entire process."""
    
    def __init__(self):
        """Initialize the research pipeline components."""
        try:
            self.planner = ResearchPlanner()
            self.gatherer = DataGatherer()
            self.kb = KnowledgeBase()
            self.report_generator = ReportGenerator()
            self.exporter = ReportExporter()
            self.clarifier = QueryClarifier()
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
    
    def process_query(self, query: str, progress_bar) -> Dict[str, Any]:
        """
        Process a research query through the pipeline.
        
        Args:
            query: The research query to process
            progress_bar: Streamlit progress bar object
            
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
            
            # Step 1: Clarify query (20% of progress)
            progress_bar.progress(0)
            st.write("🔍 Clarifying your query...")
            clarified_query, is_complete, questions = self.clarifier.clarify_query(query)
            
            if not is_complete:
                st.warning("Query needs more clarification. Please provide more details about:")
                for q in questions:
                    st.write(f"- {q}")
                return {
                    "query": query,
                    "status": "needs_clarification",
                    "timestamp": datetime.now().isoformat(),
                    "questions": questions
                }
            
            results["outputs"]["clarified_query"] = clarified_query
            progress_bar.progress(20)
            
            # Step 2: Generate research outline (40% of progress)
            st.write("📝 Generating research outline...")
            outline = self.planner.generate_outline(clarified_query)
            results["outputs"]["outline"] = outline
            progress_bar.progress(40)
            
            # Step 3: Generate research topics and gather documents (60% of progress)
            st.write("📚 Gathering relevant documents...")
            topics = self.planner.generate_research_topics(outline)
            topics["original_query"] = query
            results["outputs"]["topics"] = topics
            
            documents = self.gatherer.gather_data(topics, self.kb)
            if documents:
                self.kb.add_documents(documents)
                results["outputs"]["document_count"] = len(documents)
            else:
                results["outputs"]["document_count"] = 0
            progress_bar.progress(60)
            
            # Step 4: Generate report (80% of progress)
            st.write("📊 Generating comprehensive report...")
            report = self.report_generator.generate_report(outline, self.kb)
            results["outputs"]["report"] = report
            progress_bar.progress(80)
            
            # Step 5: Export report (100% of progress)
            st.write("💾 Exporting report in multiple formats...")
            filename = report['title'].lower().replace(' ', '_')
            
            # Format report content
            report_content = f"# {report['title']}\n\n"
            for section in report['sections']:
                report_content += f"## {section['section_title']}\n\n"
                if 'content' in section:
                    report_content += f"{section['content']}\n\n"
                if 'subsections' in section:
                    for subsection in section['subsections']:
                        report_content += f"### {subsection['subsection_title']}\n\n"
                        if 'content' in subsection:
                            report_content += f"{subsection['content']}\n\n"
            
            try:
                pdf_path = self.exporter.export_pdf(report_content, report.get('citations', []), filename)
                docx_path = self.exporter.export_docx(report, filename)
                html_path = self.exporter.export_html(report, filename)
                
                results["outputs"]["export_paths"] = {
                    "pdf": pdf_path,
                    "docx": docx_path,
                    "html": html_path
                }
            except Exception as e:
                logger.error(f"Error exporting report: {str(e)}")
                results["status"] = "error"
                results["error"] = f"Export error: {str(e)}"
            
            progress_bar.progress(100)
            return results
            
        except Exception as e:
            logger.error(f"Error in research pipeline: {str(e)}")
            return {
                "query": query,
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

def display_outline(outline):
    """Display the research outline in a structured format."""
    st.subheader("Research Outline")
    st.write(f"**Title:** {outline['title']}")
    
    for section in outline["sections"]:
        st.write(f"### {section.get('section_title', 'Untitled Section')}")
        if "subsections" in section:
            for subsection in section["subsections"]:
                st.write(f"#### {subsection.get('subsection_title', 'Untitled Subsection')}")
                if "content" in subsection:
                    st.write(subsection['content'])

def display_report(report):
    """Display the generated report in a structured format."""
    st.subheader("Generated Report")
    st.write(f"## {report['title']}")
    
    for section in report['sections']:
        st.write(f"### {section['section_title']}")
        if 'content' in section:
            st.write(section['content'])
        if 'subsections' in section:
            for subsection in section['subsections']:
                st.write(f"#### {subsection['subsection_title']}")
                if 'content' in subsection:
                    st.write(subsection['content'])

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Research Pipeline",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Research Pipeline")
    st.write("""
    Welcome to the Research Pipeline! This tool helps you conduct comprehensive research 
    by analyzing your query, gathering relevant information, and generating a detailed report.
    """)
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = ResearchPipeline()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'needs_clarification' not in st.session_state:
        st.session_state.needs_clarification = False
    if 'clarification_questions' not in st.session_state:
        st.session_state.clarification_questions = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['type'] == 'user':
            st.write(f"You: {message['content']}")
        else:
            st.write(f"Assistant: {message['content']}")
    
    # Create a form for input
    with st.form(key='query_form'):
        if st.session_state.needs_clarification:
            st.write("Please provide more details about:")
            for q in st.session_state.clarification_questions:
                st.write(f"- {q}")
            query = st.text_area(
                "Your clarification",
                placeholder="Please provide your response to the questions above...",
                height=150,
                key="clarification_input"
            )
        else:
            query = st.text_area(
                "Enter your research query",
                placeholder="e.g., What are the latest advancements in quantum computing?",
                height=150,
                key="query_input"
            )
        
        submit_button = st.form_submit_button(
            "Submit",
            type="primary",
            use_container_width=True
        )
    
    if submit_button:
        if not query:
            st.error("Please enter a query or response.")
            return
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            'type': 'user',
            'content': query
        })
        
        try:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            # Process the query
            results = st.session_state.pipeline.process_query(query, progress_bar)
            
            if results["status"] == "needs_clarification":
                # Update session state for clarification
                st.session_state.needs_clarification = True
                st.session_state.clarification_questions = results["questions"]
                
                # Add clarification questions to chat history
                questions_text = "\n".join([f"- {q}" for q in results["questions"]])
                st.session_state.chat_history.append({
                    'type': 'assistant',
                    'content': f"I need some clarification:\n{questions_text}"
                })
                
                # Force a rerun by refreshing the page
                st.rerun()
                
            elif results["status"] == "success":
                # Reset clarification state
                st.session_state.needs_clarification = False
                st.session_state.clarification_questions = []
                
                st.success("Research completed successfully! 🎉")
                
                # Clear the conversation state for a new query
                st.session_state.chat_history = []
                
                # Display tabs for different sections
                tab1, tab2, tab3 = st.tabs(["Outline", "Report", "Downloads"])
                
                with tab1:
                    display_outline(results["outputs"]["outline"])
                
                with tab2:
                    display_report(results["outputs"]["report"])
                
                with tab3:
                    st.subheader("Download Reports")
                    for format, path in results["outputs"]["export_paths"].items():
                        with open(path, 'rb') as f:
                            st.download_button(
                                f"Download {format.upper()} Report",
                                f,
                                file_name=os.path.basename(path),
                                mime=f"application/{format}"
                            )
            else:
                st.error(f"Error in research pipeline: {results.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 