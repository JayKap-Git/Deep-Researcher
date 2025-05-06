"""
Report generator module for creating research reports from gathered data.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from backend.knowledge_base import KnowledgeBase
from backend.config import OPENAI_API_KEY
import openai

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates research reports from gathered data and outlines."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.kb = KnowledgeBase()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.all_citations = []
        
    def _generate_section_content(self, section: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate content for a section using relevant documents from knowledge base.
        
        Args:
            section: Section dictionary containing title and content
            query: The original research query
            
        Returns:
            Dictionary containing generated content and citations
        """
        # Get relevant documents for this section
        section_query = f"{section.get('section_title', '')} {section.get('content', '')}"
        relevant_docs = self.kb.search(section_query, k=5)
        
        if not relevant_docs:
            logger.warning(f"No relevant documents found for section: {section.get('section_title', '')}")
            return {
                'content': '',
                'citations': []
            }
            
        # Prepare context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate content using LLM
        prompt = f"""Based on the following research documents and the section description, generate a detailed section for a research report.
        
Section Title: {section.get('section_title', '')}
Section Description: {section.get('content', '')}

Research Documents:
{context}

Please write a comprehensive section that:
1. Addresses the section's topic thoroughly
2. Uses information from the provided documents
3. Includes proper citations in the format [citation number]
4. Maintains academic writing style
5. Is well-structured and flows logically

Write the section content:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a research report writer. Write detailed, well-cited sections based on provided research documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Extract citations from content
            citations = []
            for doc in relevant_docs:
                if doc.page_content in content:
                    citations.append({
                        'text': doc.page_content,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'url': doc.metadata.get('url', '')
                    })
            
            return {
                'content': content,
                'citations': citations
            }
            
        except Exception as e:
            logger.error(f"Error generating content for section: {str(e)}")
            return {
                'content': '',
                'citations': []
            }
    
    def _process_section(self, section: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Process a section and its subsections recursively.
        
        Args:
            section: Section dictionary
            query: Original research query
            
        Returns:
            Processed section with content and citations
        """
        # Generate content for main section
        section_content = self._generate_section_content(section, query)
        
        # Process subsections if they exist
        subsections = []
        if 'subsections' in section:
            for subsection in section['subsections']:
                subsection_content = self._generate_section_content(subsection, query)
                subsections.append({
                    'title': subsection.get('subsection_title', ''),
                    'content': subsection_content['content'],
                    'citations': subsection_content['citations']
                })
                # Add subsection citations to main list
                self.all_citations.extend(subsection_content['citations'])
        
        # Add section citations to main list
        self.all_citations.extend(section_content['citations'])
        
        return {
            'title': section.get('section_title', ''),
            'content': section_content['content'],
            'citations': section_content['citations'],
            'subsections': subsections
        }
    
    def generate_report(self, outline: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate a complete research report from the outline.
        
        Args:
            outline: Research outline containing sections and subsections
            query: Original research query
            
        Returns:
            Complete report with content and citations
        """
        logger.info("Starting report generation...")
        self.all_citations = []  # Reset citations list
        
        # Process each section
        sections = []
        for section in outline['sections']:
            logger.info(f"Processing section: {section.get('section_title', '')}")
            processed_section = self._process_section(section, query)
            sections.append(processed_section)
        
        # Create the report
        report = {
            'title': outline['title'],
            'sections': sections,
            'citations': self.all_citations
        }
        
        logger.info("Report generation completed successfully")
        return report 