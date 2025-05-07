"""
Report generator module for creating research reports from gathered data.
"""

import logging
from typing import Dict, List, Any, Set
from datetime import datetime
from backend.knowledge_base import KnowledgeBase
from backend.config import OPENAI_API_KEY, BASE_MODEL
import openai
import textwrap
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates research reports from gathered data and outlines."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.kb = KnowledgeBase()
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.all_citations = []
        self.citation_map = {}  # Maps citation text to citation number
        self.max_section_length = 1000  # Maximum words per section
        self.max_citations_per_section = 5  # Maximum citations per section
        self.processed_sections = set()  # Track processed sections to avoid duplicates
        self.used_citations = defaultdict(set)  # Track citations used in each section
        self.section_hierarchy = {}  # Track section hierarchy and relationships
        self.technical_terms = set()  # Track technical terms for glossary
        self.chat_history = []  # Track chat history for context
        
    def _add_to_chat_history(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})
        
    def _get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        return self.chat_history.copy()
    
    def _validate_content(self, content: str) -> str:
        """
        Validate and clean content to ensure it's relevant and properly formatted.
        
        Args:
            content: The content to validate
            
        Returns:
            Cleaned and validated content
        """
        # Remove any duplicate paragraphs
        paragraphs = content.split('\n\n')
        unique_paragraphs = []
        seen = set()
        
        for para in paragraphs:
            # Clean paragraph
            para = para.strip()
            if not para:
                continue
                
            # Check for duplicates
            para_hash = hash(para.lower())
            if para_hash not in seen:
                seen.add(para_hash)
                unique_paragraphs.append(para)
        
        # Rejoin paragraphs
        content = '\n\n'.join(unique_paragraphs)
        
        # Remove any standalone citations
        content = re.sub(r'\[\d+\]\s*$', '', content)
        
        # Ensure proper spacing around citations
        content = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', content)
        
        # Remove duplicate section headers
        content = re.sub(r'(#{1,6}\s+.*?)\n+\1', r'\1', content)
        
        # Fix citation formatting
        content = re.sub(r'\[(\d+)\]\s*\[(\d+)\]', r'[\1,\2]', content)  # Combine adjacent citations
        content = re.sub(r'\[(\d+),\s*(\d+)\]', r'[\1,\2]', content)  # Fix spacing in combined citations
        
        # Standardize heading case
        content = re.sub(r'^(#{1,6})\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 
                        lambda m: f"{m.group(1)} {m.group(2).title()}", 
                        content, flags=re.MULTILINE)
        
        return content
    
    def _format_section_content(self, content: str, indent_level: int = 0) -> str:
        """
        Format section content with proper indentation and wrapping.
        
        Args:
            content: The content to format
            indent_level: The indentation level (0 for main sections, 1 for subsections)
            
        Returns:
            Formatted content with proper indentation
        """
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        
        # Format each paragraph with proper indentation and wrapping
        formatted_paragraphs = []
        indent = '    ' * indent_level  # 4 spaces per indent level
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Break long sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                formatted_sentences = []
                
                for sentence in sentences:
                    # Wrap text to 80 characters, accounting for indentation
                    wrapped = textwrap.fill(
                        sentence,
                        width=80 - len(indent),
                        initial_indent=indent if not formatted_sentences else indent,
                        subsequent_indent=indent
                    )
                    formatted_sentences.append(wrapped)
                
                formatted_paragraphs.append('\n'.join(formatted_sentences))
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _extract_technical_terms(self, content: str) -> Set[str]:
        """
        Extract technical terms from content for glossary.
        
        Args:
            content: The content to analyze
            
        Returns:
            Set of technical terms
        """
        # Common technical term patterns
        patterns = [
            r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b',  # Acronyms and proper nouns
            r'\b\w+(?:-\w+)+\b',  # Hyphenated terms
            r'\b\w+(?:\.\w+)+\b',  # Terms with dots
        ]
        
        terms = set()
        for pattern in patterns:
            matches = re.finditer(pattern, content)
            terms.update(match.group() for match in matches)
        
        return terms
    
    def _generate_section_content(self, section: Dict[str, Any], query: str, is_subsection: bool = False) -> Dict[str, Any]:
        """
        Generate content for a section using relevant documents from knowledge base.
        
        Args:
            section: Section dictionary containing title and content
            query: The original research query
            is_subsection: Whether this is a subsection (to avoid repetitive structure)
            
        Returns:
            Dictionary containing generated content and citations
        """
        # Check if section has already been processed
        section_key = f"{section.get('section_title', '')}_{section.get('content', '')}"
        if section_key in self.processed_sections:
            logger.warning(f"Duplicate section detected: {section.get('section_title', '')}")
            return {
                'content': '',
                'citations': []
            }
        
        self.processed_sections.add(section_key)
        
        # Get relevant documents for this section
        section_query = f"{section.get('section_title', '')} {section.get('content', '')}"
        relevant_docs = self.kb.search(section_query, k=self.max_citations_per_section)
        
        if not relevant_docs:
            logger.warning(f"No relevant documents found for section: {section.get('section_title', '')}")
            return {
                'content': '',
                'citations': []
            }
            
        # Prepare context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Add section context to chat history
        self._add_to_chat_history("system", f"Generating content for section: {section.get('section_title', '')}")
        self._add_to_chat_history("system", f"Section description: {section.get('content', '')}")
        
        # Generate content using LLM with chat history
        messages = [
            {"role": "system", "content": "You are a research report writer. Write focused, well-cited sections based on provided research documents. Use collective citations, avoid repetition, and maintain proper formatting. Include quantitative data and stay on topic. Define technical terms and maintain consistent terminology."},
            *self._get_chat_history(),
            {"role": "user", "content": f"""Based on the following research documents and the section description, generate content for a research report.
        
Section Title: {section.get('section_title', '')}
Section Description: {section.get('content', '')}

Research Documents:
{context}

Please write content that:
1. Focuses ONLY on the specific topic of this section
2. Uses information from the provided documents
3. Includes proper citations in the format [citation number]
4. Maintains academic writing style
5. Is well-structured and flows logically
6. Avoids repeating conclusions or key points
7. Uses collective citations where appropriate (e.g., [1,2,3] instead of [1][2][3])
8. Uses proper paragraph structure with clear topic sentences
9. Maintains consistent formatting and indentation
10. Stays within {self.max_section_length} words
11. Uses at most {self.max_citations_per_section} citations
12. Includes quantitative data where available
13. Avoids off-topic content
14. Maintains focus on the main research question
15. Uses proper academic citation format (Author, Year, Title, Journal, DOI/URL)
16. Defines technical terms on first use
17. Includes specific metrics and baseline comparisons
18. Uses consistent terminology throughout
19. Follows IMRaD structure (Introduction, Methods, Results, and Discussion)
20. Maintains clear section hierarchy

{'IMPORTANT: This is a subsection. Write focused content without Introduction/Results/Discussion structure. Focus on the specific topic and integrate it with the main section.' if is_subsection else 'IMPORTANT: This is a main section. Write comprehensive content that flows with the overall report structure.'}

Write the section content:"""}
        ]

        try:
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            
            # Add generated content to chat history
            self._add_to_chat_history("assistant", content)
            
            # Extract technical terms
            self.technical_terms.update(self._extract_technical_terms(content))
            
            # Validate and clean content
            content = self._validate_content(content)
            
            # Extract and deduplicate citations
            citations = []
            for doc in relevant_docs:
                if doc.page_content in content:
                    citation_text = doc.page_content
                    if citation_text not in self.citation_map:
                        self.citation_map[citation_text] = len(self.citation_map) + 1
                    citation_number = self.citation_map[citation_text]
                    citations.append({
                        'text': citation_text,
                        'source': doc.metadata.get('source', 'Unknown'),
                        'url': doc.metadata.get('url', ''),
                        'doi': doc.metadata.get('doi', ''),
                        'number': citation_number
                    })
                    self.used_citations[section_key].add(citation_number)
            
            return {
                'content': content,
                'citations': citations
            }
            
        except Exception as e:
            logger.error(f"Error generating section content: {str(e)}")
            return {
                'content': '',
                'citations': []
            }
    
    def _process_section(self, section: Dict[str, Any], query: str, is_subsection: bool = False) -> Dict[str, Any]:
        """
        Process a section and its subsections recursively.
        
        Args:
            section: Section dictionary
            query: Original research query
            is_subsection: Whether this is a subsection
            
        Returns:
            Processed section with content and citations
        """
        # Generate content for main section
        section_content = self._generate_section_content(section, query, is_subsection)
        
        # Process subsections if they exist
        subsections = []
        if 'subsections' in section:
            for subsection in section['subsections']:
                subsection_content = self._generate_section_content(subsection, query, True)
                subsections.append({
                    'title': subsection.get('subsection_title', ''),
                    'content': subsection_content['content'],
                    'citations': subsection_content['citations']
                })
        
        return {
            'title': section.get('section_title', ''),
            'content': section_content['content'],
            'citations': section_content['citations'],
            'subsections': subsections
        }
    
    def _format_references(self, citations: List[Dict[str, Any]]) -> str:
        """
        Format the references section with deduplicated citations.
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            Formatted references section
        """
        # Sort citations by number
        sorted_citations = sorted(citations, key=lambda x: x['number'])
        
        # Format references with proper indentation
        references = "## References\n\n"
        for citation in sorted_citations:
            # Format the citation number and source
            references += f"{citation['number']}. {citation['source']}\n"
            
            # Format DOI with proper indentation if it exists
            if citation.get('doi'):
                references += f"    DOI: {citation['doi']}\n"
            
            # Format URL with proper indentation if it exists
            if citation.get('url'):
                references += f"    URL: {citation['url']}\n"
            
            references += "\n"
        
        return references
    
    def _format_glossary(self) -> str:
        """
        Format the glossary section with technical terms.
        
        Returns:
            Formatted glossary section
        """
        if not self.technical_terms:
            return ""
            
        glossary = "## Glossary\n\n"
        for term in sorted(self.technical_terms):
            glossary += f"**{term}**: [Definition to be added]\n\n"
        
        return glossary
    
    def _format_report_section(self, section: Dict[str, Any], level: int = 0) -> str:
        """
        Format a report section with proper indentation and structure.
        
        Args:
            section: Section dictionary
            level: Current section level (0 for main sections)
            
        Returns:
            Formatted section content
        """
        # Format section header
        header = f"{'#' * (level + 1)} {section['title']}\n\n"
        
        # Format section content
        content = self._format_section_content(section['content'], level)
        
        # Format subsections
        subsections = ""
        if 'subsections' in section:
            for subsection in section['subsections']:
                subsections += self._format_report_section(subsection, level + 1)
        
        return f"{header}{content}\n\n{subsections}"
    
    def generate_report(self, outline: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Generate a complete research report from the outline and query.
        
        Args:
            outline: The research outline containing sections and subsections
            query: The original research query
            
        Returns:
            Dictionary containing the complete report with all sections and citations
        """
        # Clear chat history at the start of a new report
        self.chat_history = []
        
        # Add initial context to chat history
        self._add_to_chat_history("system", f"Generating research report for query: {query}")
        self._add_to_chat_history("system", f"Report title: {outline.get('title', 'Untitled Report')}")
        
        report = {
            'title': outline.get('title', 'Untitled Report'),
            'sections': [],
            'citations': [],
            'glossary': set(),
            'generated_at': datetime.now().isoformat()
        }
        
        # First, generate introduction
        intro_section = {
            'section_title': 'Introduction',
            'content': f'Introduction to the research topic: {query}'
        }
        intro_content = self._generate_section_content(intro_section, query, False)
        report['sections'].append({
            'title': 'Introduction',
            'content': intro_content['content'],
            'citations': intro_content['citations'],
            'subsections': []
        })
        
        # Process each main section in the outline
        for section in outline.get('sections', []):
            # Skip if this is the introduction section
            if section.get('section_title', '').lower() == 'introduction':
                continue
                
            # Process main section
            section_content = self._process_section(section, query, False)
            report['sections'].append(section_content)
            
            # Process subsections if they exist
            if 'subsections' in section:
                for subsection in section['subsections']:
                    subsection_content = self._process_section(subsection, query, True)
                    if 'subsections' not in section_content:
                        section_content['subsections'] = []
                    section_content['subsections'].append(subsection_content)
        
        # Finally, generate conclusion
        conclusion_section = {
            'section_title': 'Conclusion',
            'content': f'Conclusion and future directions for: {query}'
        }
        conclusion_content = self._generate_section_content(conclusion_section, query, False)
        report['sections'].append({
            'title': 'Conclusion',
            'content': conclusion_content['content'],
            'citations': conclusion_content['citations'],
            'subsections': []
        })
        
        # Add all citations to the report
        report['citations'] = self.all_citations
        
        # Add glossary
        report['glossary'] = sorted(list(self.technical_terms))
        
        return report 