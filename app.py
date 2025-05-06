"""
Deep Researcher Web Application
This module provides a web interface for the research pipeline.
"""

import streamlit as st
import os
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from datetime import datetime

# Import backend modules
from backend.clarify import QueryClarifier
from backend.planner import ResearchPlanner
from backend.gather import DataGatherer
from backend.knowledge_base import KnowledgeBase
from backend.research_answer import ResearchAnswerGenerator
from backend.export import ReportExporter
from backend.config import (
    OPENAI_API_KEY,
    UPLOADS_DIR,
    EXPORTS_DIR,
    BASE_MODEL,
    MODEL_TEMPERATURE,
    VECTORSTORE_DIR,
    VECTORSTORE_CONFIGS
)
from langchain.schema import Document
from backend.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize session state
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'current_step': 0,
        'initial_query': '',
        'clarified_query': '',
        'outline': None,
        'sectioned_documents': None,
        'report': None,
        'citations': None,
        'selected_model': None,
        'temperature': None
    }

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize components
clarifier = QueryClarifier()
planner = ResearchPlanner()
gatherer = DataGatherer()
knowledge_base = KnowledgeBase()
answerer = ResearchAnswerGenerator()
exporter = ReportExporter()
report_generator = ReportGenerator()

def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

def process_user_input(user_input):
    current_step = st.session_state.pipeline_state['current_step']
    
    if current_step == 0:  # Initial query
        if user_input.strip():
            st.session_state.pipeline_state['initial_query'] = user_input
            st.session_state.pipeline_state['current_step'] = 1
            display_message("assistant", "I'll help you research that topic. Let me clarify your query first...")
            try:
                clarified_query = planner.clarify_query(user_input)
                if clarified_query:
                    st.session_state.pipeline_state['clarified_query'] = clarified_query
                    st.session_state.pipeline_state['current_step'] = 2
                    display_message("assistant", f"Here's my understanding of your research query:\n\n{clarified_query}\n\nI'll now select the best model for your research.")
                else:
                    display_message("assistant", "I'm having trouble clarifying your query. Could you please rephrase it?")
            except Exception as e:
                display_message("assistant", f"Error during query clarification: {str(e)}")
        else:
            display_message("assistant", "Please enter a research query.")
    
    elif current_step == 2:  # Model selection
        try:
            planner.select_research_model(st.session_state.pipeline_state['clarified_query'])
            st.session_state.pipeline_state['current_step'] = 3
            display_message("assistant", "I've selected the appropriate model. Let me create a research outline...")
            try:
                outline = planner.generate_outline(st.session_state.pipeline_state['clarified_query'])
                st.session_state.pipeline_state['outline'] = outline
                display_message("assistant", f"Here's the research outline I've created:\n\n{json.dumps(outline, indent=2)}\n\nShall I proceed with gathering data? (Type 'yes' to continue)")
            except Exception as e:
                display_message("assistant", f"Error during research planning: {str(e)}")
        except Exception as e:
            display_message("assistant", f"Error during model selection: {str(e)}")
    
    elif current_step == 3:  # Research planning
        if user_input.lower() == 'yes':
            st.session_state.pipeline_state['current_step'] = 4
            display_message("assistant", "I'll start gathering research data from various sources...")
            try:
                with st.spinner("Gathering data..."):
                    sectioned_documents = gatherer.gather_data(
                        st.session_state.pipeline_state['clarified_query'],
                        st.session_state.pipeline_state['outline']
                    )
                    st.session_state.pipeline_state['sectioned_documents'] = sectioned_documents
                    
                    # Display document counts
                    source_counts = {}
                    for docs in sectioned_documents.values():
                        for doc in docs:
                            source = doc.metadata.get("source", "unknown")
                            source_counts[source] = source_counts.get(source, 0) + 1
                    
                    sources_text = "\n".join([f"- {source}: {count} documents" for source, count in source_counts.items()])
                    display_message("assistant", f"I've gathered data from the following sources:\n\n{sources_text}\n\nShall I proceed with building the knowledge base? (Type 'yes' to continue)")
            except Exception as e:
                display_message("assistant", f"Error during data gathering: {str(e)}")
    
    elif current_step == 4:  # Data gathering
        if user_input.lower() == 'yes':
            st.session_state.pipeline_state['current_step'] = 5
            display_message("assistant", "I'll now build the knowledge base...")
            try:
                with st.spinner("Building knowledge base..."):
                    all_documents = []
                    for docs in st.session_state.pipeline_state['sectioned_documents'].values():
                        all_documents.extend(docs)
                    
                    knowledge_base.add_documents(all_documents)
                    stats = knowledge_base.get_stats()
                    
                    display_message("assistant", f"Knowledge base built successfully!\n\nStatistics:\n- Total documents: {stats.get('total_documents', 0)}\n- Vector store size: {stats.get('vector_store_size', 0)}\n\nShall I generate the research report? (Type 'yes' to continue)")
            except Exception as e:
                display_message("assistant", f"Error during knowledge base building: {str(e)}")
    
    elif current_step == 5:  # Knowledge base building
        if user_input.lower() == 'yes':
            st.session_state.pipeline_state['current_step'] = 6
            display_message("assistant", "I'll now generate the research report...")
            try:
                with st.spinner("Generating report..."):
                    report, citations = report_generator.generate_report(
                        st.session_state.pipeline_state['outline'],
                        st.session_state.pipeline_state['clarified_query']
                    )
                    st.session_state.pipeline_state['report'] = report
                    st.session_state.pipeline_state['citations'] = citations
                    
                    display_message("assistant", f"Here's your research report:\n\n{report}\n\nCitations:\n\n" + 
                                  "\n".join([f"[{citation['index']}] {citation['title']}\n    URL: {citation['url']}" for citation in citations]) +
                                  "\n\nWould you like me to export this report? (Type 'yes' to export)")
            except Exception as e:
                display_message("assistant", f"Error during report generation: {str(e)}")
    
    elif current_step == 6:  # Report generation
        if user_input.lower() == 'yes':
            st.session_state.pipeline_state['current_step'] = 7
            display_message("assistant", "I'll now export the report...")
            try:
                with st.spinner("Exporting report..."):
                    pdf_path = exporter.export_pdf(
                        report_content=st.session_state.report,
                        citations=st.session_state.citations,
                        filename=st.session_state.report['title'].lower().replace(' ', '_')
                    )
                    display_message("assistant", f"Report exported successfully to: {pdf_path}\n\nWould you like to start a new research project? (Type 'yes' to start new research)")
            except Exception as e:
                display_message("assistant", f"Error during report export: {str(e)}")
    
    elif current_step == 7:  # Report export
        if user_input.lower() == 'yes':
            st.session_state.pipeline_state = {
                'current_step': 0,
                'initial_query': '',
                'clarified_query': '',
                'outline': None,
                'sectioned_documents': None,
                'report': None,
                'citations': None,
                'selected_model': None,
                'temperature': None
            }
            display_message("assistant", "Great! I'm ready to help you with a new research project. What would you like to research?")

def clear_knowledge_base():
    """Clear the knowledge base and reinitialize it."""
    try:
        kb = KnowledgeBase()
        kb.clear()
        return True
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        return False

def main():
    st.set_page_config(
        page_title="Research Pipeline",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Research Pipeline")
    st.markdown("""
    This application helps you conduct comprehensive research on any topic. 
    Enter your research query below to get started.
    """)

    # Initialize session state
    if 'research_complete' not in st.session_state:
        st.session_state.research_complete = False
    if 'report' not in st.session_state:
        st.session_state.report = None

    # Research query input
    query = st.text_area(
        "Enter your research query",
        placeholder="e.g., What are the latest advancements in quantum computing?",
        height=100
    )

    if st.button("Start Research", type="primary"):
        if not query:
            st.error("Please enter a research query")
            return

        with st.spinner("Initializing research pipeline..."):
            # Initialize components
            planner = ResearchPlanner()
            gatherer = DataGatherer()
            kb = KnowledgeBase()
            report_generator = ReportGenerator()
            exporter = ReportExporter()

            # Clear knowledge base
            if clear_knowledge_base():
                st.success("Knowledge base cleared successfully")

        # Create progress container
        progress_container = st.container()
        with progress_container:
            st.markdown("### Research Progress")

            # Step 1: Clarify query
            with st.spinner("Clarifying query..."):
                clarified_query = planner.clarify_query(query)
                st.info(f"Clarified query: {clarified_query}")

            # Step 2: Generate research outline
            with st.spinner("Generating research outline..."):
                outline = planner.generate_outline(clarified_query)
                st.success("Research outline generated")
                
                # Display outline
                st.markdown("#### Research Outline")
                st.markdown(f"**Title:** {outline['title']}")
                for section in outline["sections"]:
                    st.markdown(f"- {section.get('section_title', 'Untitled Section')}")
                    if "subsections" in section:
                        for subsection in section["subsections"]:
                            st.markdown(f"  * {subsection.get('subsection_title', 'Untitled Subsection')}")

            # Step 3: Generate research topics
            with st.spinner("Generating research topics..."):
                topics = planner.generate_research_topics(outline)
                st.success("Research topics generated")

            # Step 4: Gather documents
            with st.spinner("Gathering documents..."):
                documents = gatherer.gather_data(topics)
                if documents:
                    kb.add_documents(documents)
                    st.success(f"Added {len(documents)} documents to knowledge base")
                else:
                    st.warning("No documents found")

            # Step 5: Generate report
            with st.spinner("Generating report..."):
                report = report_generator.generate_report(outline, clarified_query)
                st.session_state.report = report
                st.session_state.research_complete = True
                st.success("Report generated successfully!")

    # Display report if research is complete
    if st.session_state.research_complete and st.session_state.report:
        st.markdown("---")
        st.markdown("## 📄 Generated Report")
        
        report = st.session_state.report
        
        # Display report content
        st.markdown(f"# {report['title']}")
        
        for section in report['sections']:
            st.markdown(f"## {section['title']}")
            if 'content' in section:
                st.markdown(section['content'])
            
            if 'subsections' in section:
                for subsection in section['subsections']:
                    st.markdown(f"### {subsection['title']}")
                    if 'content' in subsection:
                        st.markdown(subsection['content'])

        # Export options
        st.markdown("### Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export as PDF"):
                with st.spinner("Exporting to PDF..."):
                    try:
                        exporter = ReportExporter()
                        pdf_path = exporter.export_pdf(
                            report_content=st.session_state.report,
                            citations=st.session_state.citations,
                            filename=st.session_state.report['title'].lower().replace(' ', '_')
                        )
                        st.success(f"Report exported to PDF: {pdf_path}")
                    except Exception as e:
                        st.error(f"Error exporting to PDF: {str(e)}")

        with col2:
            if st.button("Export as DOCX"):
                with st.spinner("Exporting to DOCX..."):
                    try:
                        exporter = ReportExporter()
                        docx_path = exporter.export_docx(
                            report=st.session_state.report,
                            filename=st.session_state.report['title'].lower().replace(' ', '_')
                        )
                        st.success(f"Report exported to DOCX: {docx_path}")
                    except Exception as e:
                        st.error(f"Error exporting to DOCX: {str(e)}")

        with col3:
            if st.button("Export as HTML"):
                with st.spinner("Exporting to HTML..."):
                    try:
                        exporter = ReportExporter()
                        html_path = exporter.export_html(
                            report=st.session_state.report,
                            filename=st.session_state.report['title'].lower().replace(' ', '_')
                        )
                        st.success(f"Report exported to HTML: {html_path}")
                    except Exception as e:
                        st.error(f"Error exporting to HTML: {str(e)}")

if __name__ == "__main__":
    main() 