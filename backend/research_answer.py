"""
Research Answer Generator

This module generates detailed research answers by processing sections and subsections
from gathered research data and using LLM to synthesize comprehensive responses.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE
import openai
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAnswerGenerator:
    """Generates comprehensive research answers based on gathered documents."""
    
    def __init__(self):
        """Initialize the research answer generator."""
        self.model = ChatOpenAI(
            model=BASE_MODEL,
            temperature=MODEL_TEMPERATURE,
            openai_api_key=OPENAI_API_KEY
        )
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.chat_history = []  # Track chat history for context
        self.citation_map = {}  # Maps citation text to citation number
        self.used_citations = defaultdict(set)  # Track citations used in each section
        self.technical_terms = set()  # Track technical terms for glossary
        self.max_section_length = 3000  # Maximum length of a section in words
        self.max_citations_per_section = 15  # Maximum number of citations per section
        logger.info(f"Initialized ResearchAnswerGenerator with model: {BASE_MODEL} and temperature: {MODEL_TEMPERATURE}")

    def generate_answer(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate a comprehensive answer based on the query and documents."""
        try:
            # Input validation
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            if not documents or not isinstance(documents, list):
                raise ValueError("Documents must be a non-empty list")
            
            if not all(isinstance(doc, Document) for doc in documents):
                raise ValueError("All documents must be instances of Document")
            
            # Filter out empty documents
            valid_documents = [doc for doc in documents if doc.page_content.strip()]
            if not valid_documents:
                raise ValueError("No valid documents with content found")

            # Extract relevant information from documents
            document_texts = [doc.page_content for doc in valid_documents]
            document_sources = [doc.metadata.get("source", "Unknown") for doc in valid_documents]

            # Generate answer using the language model
            prompt = self._create_answer_prompt(query, document_texts)
            response = self.model.predict(prompt)

            # Process and format the response
            answer = self._process_response(response)
            sources = self._format_sources(valid_documents)
            confidence = self._calculate_confidence(response, valid_documents)

            # Validate the generated answer
            if not answer or len(answer.strip()) < 50:
                logger.warning("Generated answer is too short or empty")
                raise ValueError("Generated answer is too short or empty")

            return {
                "answer": answer,
                "sources": sources,
                "confidence": confidence
            }

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def _create_answer_prompt(self, query: str, document_texts: List[str]) -> str:
        """Create a prompt for answer generation."""
        context = "\n\n".join(document_texts)
        return f"""Based on the following research documents, please provide a comprehensive answer to the question: {query}

Research Documents:
{context}

Please provide a detailed answer that:
1. Directly addresses the question
2. Synthesizes information from multiple sources
3. Highlights key findings and insights
4. Maintains academic rigor and accuracy
5. Cites specific sources when making claims

Answer:"""

    def _process_response(self, response: str) -> str:
        """Process and format the model's response."""
        # Clean up and format the response
        lines = response.strip().split("\n")
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Format source information from documents."""
        sources = []
        for doc in documents:
            source = {
                "title": doc.metadata.get("title", "Untitled"),
                "url": doc.metadata.get("url", ""),
                "relevance": doc.metadata.get("relevance", 1.0)
            }
            sources.append(source)
        return sources

    def _calculate_confidence(self, response: str, documents: List[Document]) -> float:
        """Calculate confidence score for the generated answer."""
        # Simple confidence calculation based on:
        # 1. Number of sources used
        # 2. Length and detail of the answer
        # 3. Presence of specific citations
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on number of sources
        source_factor = min(len(documents) / 5, 1.0)  # Max out at 5 sources
        confidence += source_factor * 0.2
        
        # Adjust based on answer length
        length_factor = min(len(response) / 1000, 1.0)  # Max out at 1000 chars
        confidence += length_factor * 0.2
        
        # Adjust based on citations
        citation_count = response.count("according to") + response.count("suggests") + response.count("shows")
        citation_factor = min(citation_count / 5, 1.0)  # Max out at 5 citations
        confidence += citation_factor * 0.1
        
        return min(confidence, 1.0)  # Ensure confidence doesn't exceed 1.0

    def _add_to_chat_history(self, role: str, content: str):
        """Add a message to the chat history."""
        self.chat_history.append({"role": role, "content": content})
        
    def _get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history."""
        return self.chat_history.copy()
    
    def _validate_content(self, content: str) -> bool:
        """
        Validate the generated content.
        
        Args:
            content: The content to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        if not content:
            logger.warning("Empty content")
            return False
            
        # Check content length
        word_count = len(content.split())
        if word_count > self.max_section_length:
            logger.warning(f"Content exceeds maximum length: {word_count} words")
            return False
            
        # Check for citations
        citations = re.findall(r'\[\d+\]', content)
        if len(set(citations)) > self.max_citations_per_section:
            logger.warning(f"Content exceeds maximum unique citations: {len(set(citations))} citations")
            return False
            
        # Check for minimum content
        if word_count < 50:  # Reduced minimum word count
            logger.warning(f"Content too short: {word_count} words")
            return False
            
        # Check for balanced citations
        if len(citations) < 1:  # Reduced minimum citation requirement
            logger.warning("Content has too few citations")
            return False
            
        return True
        
    def _extract_technical_terms(self, content: str):
        """Extract technical terms from content."""
        # Use LLM to identify technical terms
        messages = [
            {"role": "system", "content": "You are a technical term extractor. Identify technical terms and jargon in the given text."},
            {"role": "user", "content": f"Extract technical terms and jargon from the following text:\n\n{content}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            
            terms = response.choices[0].message.content.strip().split('\n')
            self.technical_terms.update(term.strip() for term in terms if term.strip())
            
        except Exception as e:
            logger.error(f"Error extracting technical terms: {str(e)}")
            
    def generate_section_content(self, section: Dict[str, Any], kb, query: str, is_subsection: bool = False) -> str:
        """
        Generate content for a section.
        
        Args:
            section: Section information
            kb: Knowledge base instance
            query: The research query
            is_subsection: Whether this is a subsection
            
        Returns:
            Generated section content
        """
        section_title = section.get('section_title', '')
        section_content = section.get('content', '')
        
        # Construct section query
        section_query = f"{query} {section_title} {section_content}"
        
        # Get relevant documents
        relevant_docs = kb.search(section_query, k=self.max_citations_per_section)
        if not relevant_docs:
            logger.warning(f"No relevant documents found for section: {section_title}")
            return ""
            
        # Prepare context from relevant documents
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # Add citations to map
        for i, doc in enumerate(relevant_docs, 1):
            citation_text = f"[{i}] {doc.metadata.get('title', '')} ({doc.metadata.get('year', '')})"
            self.citation_map[citation_text] = i
            
        # Generate content using LLM
        messages = [
            {"role": "system", "content": """You are a research content generator specializing in academic writing. Your task is to generate well-structured, 
academic content for research report sections. You should:
1. Use formal academic language
2. Integrate information from multiple sources
3. Use proper citation format [1], [2], etc.
4. Maintain logical flow and coherence
5. Be precise and specific
6. Use technical terms appropriately
7. Stay within word limits
8. Ensure balanced use of citations
9. Focus on key findings and insights
10. Maintain objectivity"""},
            {"role": "user", "content": f"""Generate content for the following section of a research report.

Section Title: {section_title}
Section Description: {section_content}
Research Query: {query}

Available Documents for Citation:
{context}

Requirements:
1. Generate well-structured academic content
2. Use citations in the format [1], [2], etc.
3. Each major claim should be supported by at least one citation
4. Use between {max(1, self.max_citations_per_section//3)} and {self.max_citations_per_section} unique citations
5. Keep the content between {max(50, self.max_section_length//2)} and {self.max_section_length} words
6. Maintain academic writing style
7. Focus on synthesizing information from multiple sources
8. Use technical terms appropriately
9. Ensure logical flow between paragraphs
10. End with a brief summary or conclusion

Generate the section content:"""}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content.strip()
            
            # Validate content
            if not self._validate_content(content):
                logger.warning(f"Generated content for section {section_title} failed validation")
                # Try one more time with adjusted parameters
                messages[1]["content"] += "\n\nNote: Previous attempt failed validation. Please ensure proper citation usage and content length."
                response = self.client.chat.completions.create(
                    model=BASE_MODEL,
                    messages=messages,
                    temperature=0.5,  # Reduce temperature for more focused output
                    max_tokens=2000
                )
                content = response.choices[0].message.content.strip()
                if not self._validate_content(content):
                    return ""
                
            # Extract technical terms
            self._extract_technical_terms(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error generating content for section {section_title}: {str(e)}")
            return ""
            
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
        # Generate content for main section
        content = self.generate_section_content(section, kb, query)
        
        # Process subsections
        processed_subsections = []
        for subsection in section.get('subsections', []):
            subsection_content = self.generate_section_content(subsection, kb, query, is_subsection=True)
            processed_subsections.append({
                'subsection_title': subsection.get('subsection_title', ''),
                'content': subsection_content
            })
            
        return {
            'section_title': section.get('section_title', ''),
            'content': content,
            'subsections': processed_subsections
        }
    
    def clear_history(self):
        """Clear the chat history and other tracking data."""
        self.chat_history = []
        self.citation_map = {}
        self.used_citations = defaultdict(set)
        self.technical_terms = set()

def main():
    """Example usage of the ResearchAnswerGenerator."""
    try:
        # Example data
        query = "How effective are transformer-based models in biomedical text analysis?"
        outline = {
            "sections": [
                {
                    "Introduction": {
                        "description": "Overview of transformer models and their applications in biomedicine"
                    }
                },
                {
                    "Methodology": {
                        "description": "Approaches to evaluating transformer models",
                        "subsections": [
                            {
                                "Evaluation Metrics": "Metrics used to assess model performance"
                            },
                            {
                                "Datasets": "Common biomedical datasets used for evaluation"
                            }
                        ]
                    }
                },
                {
                    "Results": {
                        "description": "Performance analysis of transformer models",
                        "subsections": [
                            {
                                "Accuracy": "Quantitative performance metrics"
                            },
                            {
                                "Limitations": "Challenges and limitations identified"
                            }
                        ]
                    }
                }
            ]
        }
        
        # Example documents
        documents = [
            Document(
                page_content="Sample document content",
                metadata={
                    "title": "Sample Paper",
                    "url": "https://example.com",
                    "source": "web"
                }
            )
        ]
        
        # Generate answer
        generator = ResearchAnswerGenerator()
        answer = generator.generate_answer(query, documents)
        
        print("\n=== Research Answer ===")
        print(answer["answer"])
        print("\n=== Sources ===")
        for source in answer["sources"]:
            print(f"[{source['title']}] {source['url']}")
        print(f"\nConfidence: {answer['confidence']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 