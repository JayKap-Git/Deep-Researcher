"""
Report generator module for creating structured research reports.
"""

import logging
from typing import Dict, List, Any, Optional
import openai
from backend.config import OPENAI_API_KEY, BASE_MODEL
from backend.research_answer import ResearchAnswerGenerator
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates structured research reports from knowledge base content."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.research_generator = ResearchAnswerGenerator()
        self.max_sections = 5  # Maximum number of main sections
        self.max_subsections = 3  # Maximum number of subsections per section
        
    def generate_outline(self, query: str, kb) -> Dict[str, Any]:
        """
        Generate a structured outline for the research report.
        
        Args:
            query: The research query
            kb: Knowledge base instance
            
        Returns:
            Dictionary containing the report outline
        """
        # Get relevant documents for outline generation
        relevant_docs = kb.search(query, k=10)
        if not relevant_docs:
            logger.warning("No relevant documents found for outline generation")
            return {}
            
        # Prepare context from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Generate outline using LLM
        messages = [
            {"role": "system", "content": "You are a research report outline generator. Create a structured outline for a research report based on the provided documents and query. Follow IMRaD structure and include relevant subsections. Your response must be valid JSON."},
            {"role": "user", "content": f"""Based on the following research documents and query, generate a structured outline for a research report.

Query: {query}

Research Documents:
{context}

Please create an outline that:
1. Follows IMRaD structure (Introduction, Methods, Results, and Discussion)
2. Includes relevant subsections for each main section
3. Maintains logical flow and organization
4. Covers all key aspects of the research topic
5. Is well-structured and hierarchical
6. Includes a clear introduction and conclusion
7. Has at most {self.max_sections} main sections
8. Has at most {self.max_subsections} subsections per section
9. Uses clear and descriptive section titles
10. Maintains academic writing style

IMPORTANT: Your response must be valid JSON with the following structure:
{{
    "title": "Report Title",
    "sections": [
        {{
            "section_title": "Section Title",
            "content": "Brief description of section content",
            "subsections": [
                {{
                    "subsection_title": "Subsection Title",
                    "content": "Brief description of subsection content"
                }}
            ]
        }}
    ]
}}

Generate the outline:"""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                response_format={ "type": "json_object" }  # Force JSON response
            )
            
            try:
                outline = json.loads(response.choices[0].message.content)
                
                # Validate outline structure
                if not isinstance(outline, dict):
                    logger.error("Generated outline is not a dictionary")
                    return {}
                    
                if 'title' not in outline or 'sections' not in outline:
                    logger.error("Generated outline missing required fields")
                    return {}
                    
                if not isinstance(outline['sections'], list):
                    logger.error("Sections field is not a list")
                    return {}
                    
                # Validate each section
                for section in outline['sections']:
                    if not isinstance(section, dict):
                        logger.error("Invalid section format")
                        return {}
                    if 'section_title' not in section:
                        section['section_title'] = section.get('title', 'Untitled Section')
                    if 'content' not in section:
                        section['content'] = ''
                    if 'subsections' in section and not isinstance(section['subsections'], list):
                        section['subsections'] = []
                        
                return outline
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                return {}
            
        except Exception as e:
            logger.error(f"Error generating outline: {str(e)}")
            return {}
    
    def generate_report(self, outline: Dict[str, Any], kb) -> Dict[str, Any]:
        """
        Generate a complete research report.
        
        Args:
            outline: The research outline containing sections and subsections
            kb: Knowledge base instance
            
        Returns:
            Dictionary containing the complete report
        """
        if not outline:
            logger.warning("Empty outline provided")
            return {}
            
        if not kb:
            logger.error("Knowledge base instance is required")
            return {}
            
        try:
            # Process each section
            processed_sections = []
            all_content = []  # Collect all content for technical term extraction
            
            for section in outline.get('sections', []):
                # Ensure section has required fields
                if 'section_title' not in section:
                    section['section_title'] = section.get('title', 'Untitled Section')
                if 'content' not in section:
                    section['content'] = ''
                    
                # Process subsections
                if 'subsections' in section:
                    for subsection in section['subsections']:
                        if 'subsection_title' not in subsection:
                            subsection['subsection_title'] = subsection.get('title', 'Untitled Subsection')
                        if 'content' not in subsection:
                            subsection['content'] = ''
                
                try:
                    processed_section = self.process_section(section, kb, outline.get('title', ''))
                    processed_sections.append(processed_section)
                    
                    # Collect content for technical term extraction
                    all_content.append(processed_section.get('content', ''))
                    for subsection in processed_section.get('subsections', []):
                        all_content.append(subsection.get('content', ''))
                        
                except Exception as e:
                    logger.error(f"Error processing section {section.get('section_title')}: {str(e)}")
                    continue
            
            # Extract technical terms from all content at once
            combined_content = "\n\n".join(all_content)
            technical_terms = self.research_generator.extract_all_technical_terms(combined_content)
            
            # Generate glossary from collected terms
            glossary = self._generate_glossary(technical_terms)
            
            # Generate references
            references = self._generate_references()
            
            # Compile final report
            report = {
                'title': outline.get('title', 'Untitled Report'),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'query': outline.get('title', ''),
                'sections': processed_sections,
                'glossary': glossary,
                'references': references
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {}
    
    def process_section(self, section: Dict[str, Any], kb, query: str) -> Dict[str, Any]:
        """
        Process a section and its subsections.
        
        Args:
            section: Section information
            kb: Knowledge base instance
            query: The research query
            
        Returns:
            Processed section with content
        """
        logger.info(f"\nProcessing main section: {section.get('section_title', '')}")
        # Generate content for main section
        content = self.research_generator.generate_section_content(section, kb, query, is_subsection=False)
        
        # Process subsections if they exist
        processed_subsections = []
        if 'subsections' in section:
            logger.info(f"Found {len(section['subsections'])} subsections to process")
            for subsection in section['subsections']:
                # Ensure subsection has required fields
                if not isinstance(subsection, dict):
                    subsection = {'subsection_title': subsection}
                elif 'subsection_title' not in subsection:
                    continue
                    
                logger.info(f"\n  Processing subsection: {subsection.get('subsection_title', '')}")
                # Create a simplified subsection structure without nested subsections
                simple_subsection = {
                    'section_title': subsection['subsection_title'],
                    'content': subsection.get('content', '')
                }
                
                logger.info("  Creating simplified subsection structure (no nested subsections allowed)")
                # Generate content for subsection
                subsection_content = self.research_generator.generate_section_content(
                    simple_subsection,
                    kb,
                    query,
                    is_subsection=True  # Mark as subsection to enforce different constraints
                )
                
                if subsection_content:
                    logger.info("  Successfully generated subsection content")
                else:
                    logger.warning("  Failed to generate valid subsection content")
                
                processed_subsections.append({
                    'subsection_title': subsection['subsection_title'],
                    'content': subsection_content
                })
            
        return {
            'section_title': section.get('section_title', ''),
            'content': content,
            'subsections': processed_subsections
        }
    
    def _generate_glossary(self, technical_terms: List[str]) -> List[Dict[str, str]]:
        """
        Generate a glossary from collected technical terms.
        
        Args:
            technical_terms: List of technical terms
            
        Returns:
            List of glossary entries
        """
        glossary = []
        for term in technical_terms:
            # Generate definition using LLM
            messages = [
                {"role": "system", "content": "You are a technical term definition generator. Generate clear and concise definitions for technical terms."},
                {"role": "user", "content": f"Generate a clear and concise definition for the technical term: {term}"}
            ]
            
            try:
                response = self.client.chat.completions.create(
                    model=BASE_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                
                definition = response.choices[0].message.content.strip()
                glossary.append({
                    'term': term,
                    'definition': definition
                })
                
            except Exception as e:
                logger.error(f"Error generating definition for term {term}: {str(e)}")
                continue
                
        return glossary
    
    def _generate_references(self) -> List[Dict[str, str]]:
        """
        Generate a references list from collected citations.
        
        Returns:
            List of references
        """
        references = []
        for citation_text, citation_number in self.research_generator.citation_map.items():
            # Extract reference information using LLM
            messages = [
                {"role": "system", "content": "You are a reference formatter. Extract and format reference information from citation text."},
                {"role": "user", "content": f"Extract and format reference information from the following citation text:\n\n{citation_text}"}
            ]
            
            try:
                response = self.client.chat.completions.create(
                    model=BASE_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=200
                )
                
                reference = response.choices[0].message.content.strip()
                references.append({
                    'number': citation_number,
                    'text': reference
                })
                
            except Exception as e:
                logger.error(f"Error generating reference for citation {citation_number}: {str(e)}")
                continue
                
        return sorted(references, key=lambda x: x['number'])
    
    def export_report(self, report: Dict[str, Any], format: str = 'json', output_dir: str = 'reports') -> Optional[str]:
        """
        Export the report in the specified format.
        
        Args:
            report: The report to export
            format: Export format ('json' or 'docx')
            output_dir: Directory to save the exported file
            
        Returns:
            Path to the exported file if successful, None otherwise
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{timestamp}"
        
        try:
            if format.lower() == 'json':
                output_path = os.path.join(output_dir, f"{filename}.json")
                with open(output_path, 'w') as f:
                    json.dump(report, f, indent=2)
                    
            elif format.lower() == 'docx':
                output_path = os.path.join(output_dir, f"{filename}.docx")
                # TODO: Implement DOCX export
                logger.warning("DOCX export not yet implemented")
                return None
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting report: {str(e)}")
            return None

def main():
    """Example usage of the ReportGenerator."""
    from backend.knowledge_base import KnowledgeBase
    
    # Initialize knowledge base
    kb = KnowledgeBase()
    
    # Initialize report generator
    generator = ReportGenerator()
    
    # Generate report
    query = "What are the latest developments in quantum computing?"
    outline = generator.generate_outline(query, kb)
    report = generator.generate_report(outline, kb)
    
    # Export report
    if report:
        output_path = generator.export_report(report, format='json')
        if output_path:
            print(f"Report exported to: {output_path}")

if __name__ == "__main__":
    main() 