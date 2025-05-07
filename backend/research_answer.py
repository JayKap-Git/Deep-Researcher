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
        self.max_citations_per_section = 20  # Maximum number of citations per section
        self.max_citations_per_subsection = 10  # Maximum number of citations per subsection
        self.max_subsection_length = 1000  # Maximum length of a subsection in words
        self.min_subsection_length = 150  # Minimum length of a subsection in words
        self.extract_terms = False  # Flag to control technical term extraction
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
    
    def _validate_content(self, content: str, is_subsection: bool = False) -> bool:
        """
        Validate the generated content.
        
        Args:
            content: The content to validate
            is_subsection: Whether this is a subsection content
            
        Returns:
            True if content is valid, False otherwise
        """
        if not content:
            logger.warning("Empty content")
            return False
            
        # Check content length
        word_count = len(content.split())
        max_length = self.max_subsection_length if is_subsection else self.max_section_length
        min_length = self.min_subsection_length if is_subsection else 50
        
        if word_count > max_length:
            logger.warning(f"Content exceeds maximum length: {word_count} words")
            return False
            
        if word_count < min_length:
            logger.warning(f"Content too short: {word_count} words")
            return False
            
        # Check for citations
        citations = re.findall(r'\[\d+\]', content)
        max_citations = self.max_citations_per_subsection if is_subsection else self.max_citations_per_section
        
        if len(set(citations)) > max_citations:
            logger.warning(f"Content exceeds maximum unique citations: {len(set(citations))} citations")
            return False
            
        if len(citations) < 1:
            logger.warning("Content has too few citations")
            return False

        # Additional validation for subsections
        if is_subsection:
            # Split content into paragraphs
            paragraphs = [p.strip().lower() for p in content.split('\n\n') if p.strip()]
            
            if len(paragraphs) > 3:
                logger.warning("Subsection has too many paragraphs")
                return False
                
            if len(paragraphs) < 1:
                logger.warning("Subsection has too few paragraphs")
                return False

            # Check first paragraph for introduction-like content
            first_para = paragraphs[0]
            intro_indicators = [
                'introduction', 'this section', 'in this part', 'overview',
                'we will discuss', 'this subsection', 'this paper', 'this report',
                'in the following', 'here we', 'we present', 'we describe',
                'we explore', 'we examine', 'we investigate', 'we analyze',
                'we review', 'we study', 'we focus on', 'we consider'
            ]
            if any(indicator in first_para for indicator in intro_indicators):
                logger.warning("Subsection contains introduction-like content")
                return False

            # Check last paragraph for conclusion-like content
            last_para = paragraphs[-1]
            conclusion_indicators = [
                'conclusion', 'in summary', 'to summarize', 'finally',
                'in conclusion', 'thus', 'therefore', 'overall',
                'to conclude', 'in closing', 'ultimately', 'in the end',
                'taken together', 'as shown above', 'as discussed',
                'as demonstrated', 'as illustrated', 'as presented',
                'as outlined', 'as described', 'as mentioned',
                'looking ahead', 'future work', 'future research',
                'moving forward', 'next steps'
            ]
            if any(indicator in last_para for indicator in conclusion_indicators):
                logger.warning("Subsection contains conclusion-like content")
                return False

            # Check for meta-references and transitions
            meta_indicators = [
                'as mentioned earlier', 'as discussed above',
                'as shown previously', 'in the previous section',
                'in the next section', 'following section',
                'preceding section', 'subsequent section',
                'above discussion', 'below we', 'later we',
                'furthermore', 'moreover', 'additionally',
                'in addition', 'besides', 'also'
            ]
            for para in paragraphs:
                if any(indicator in para for indicator in meta_indicators):
                    logger.warning("Subsection contains meta-references or transitions")
                    return False
            
            # Check for balanced citation distribution
            citations_per_para = []
            current_para_citations = 0
            for line in content.split('\n'):
                current_para_citations += len(re.findall(r'\[\d+\]', line))
                if not line.strip():  # Empty line indicates paragraph break
                    if current_para_citations > 0:
                        citations_per_para.append(current_para_citations)
                    current_para_citations = 0
            if current_para_citations > 0:  # Add citations from last paragraph
                citations_per_para.append(current_para_citations)
            
            if not citations_per_para:
                logger.warning("No citations found in paragraphs")
                return False
            
            # Check if citations are too concentrated in one paragraph
            max_citations_per_para = max(citations_per_para)
            total_citations = sum(citations_per_para)
            if max_citations_per_para > total_citations * 0.7:  # More than 70% citations in one paragraph
                logger.warning("Citations are not well distributed across paragraphs")
                return False
            
        return True
        
    def _extract_technical_terms(self, content: str):
        """Extract technical terms from content."""
        # Skip technical term extraction if flag is False
        if not self.extract_terms:
            return set()
            
        try:
            messages = [
                {"role": "system", "content": "You are a technical term extractor. Identify technical terms and jargon in the given text."},
                {"role": "user", "content": f"Extract technical terms and jargon from the following text:\n\n{content}"}
            ]
            
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            terms = set(response.choices[0].message.content.split('\n'))
            return {term.strip() for term in terms if term.strip()}
            
        except Exception as e:
            logger.error(f"Error extracting technical terms: {str(e)}")
            return set()

    def extract_all_technical_terms(self, report_content: str):
        """Extract technical terms from the entire report at once."""
        self.extract_terms = True  # Enable term extraction temporarily
        all_terms = self._extract_technical_terms(report_content)
        self.technical_terms.update(all_terms)
        self.extract_terms = False  # Disable term extraction again
        return self.technical_terms

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
        
        logger.info(f"Generating content for {'subsection' if is_subsection else 'section'}: {section_title}")
        
        # Construct section query
        if is_subsection:
            # For subsections, focus query on the specific topic
            section_query = f"{section_title}: {section_content}"
            logger.info("Using focused subsection query without broader context")
        else:
            # For main sections, include broader context
            section_query = f"{query}: {section_title}: {section_content}"
            logger.info("Using full section query with broader context")
        
        # Get relevant documents
        k = self.max_citations_per_subsection if is_subsection else self.max_citations_per_section
        logger.info(f"Using maximum {k} citations for {'subsection' if is_subsection else 'section'}")
        
        relevant_docs = kb.search(section_query, k=k)
        if not relevant_docs:
            logger.warning(f"No relevant documents found for {'subsection' if is_subsection else 'section'}: {section_title}")
            return ""
            
        logger.info(f"Found {len(relevant_docs)} relevant documents")
            
        # Prepare context from relevant documents
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # Add citations to map
        for i, doc in enumerate(relevant_docs, 1):
            citation_text = f"[{i}] {doc.metadata.get('title', '')} ({doc.metadata.get('year', '')})"
            self.citation_map[citation_text] = i

        # Different prompts for main sections and subsections
        if is_subsection:
            logger.info("Using subsection-specific prompt with strict formatting rules")
            system_content = """You are a research content generator specializing in academic writing. Your task is to generate focused, direct content for a subsection.

IMPORTANT: This is a SUBSECTION, not a full section. Keep content focused and avoid any section-like structure.

STRICT FORMATTING RULES:
1. NO introductions or conclusions
2. NO overview statements
3. NO transitions between topics
4. NO meta-references
5. NO broad context setting
6. NO phrases like "as mentioned", "furthermore", "moreover"
7. NO references to other sections
8. NO future work discussions
9. NO subsection divisions

CONTENT REQUIREMENTS:
1. Start DIRECTLY with specific, relevant information
2. Use precise, technical language
3. Support claims with citations [1], [2]
4. Focus ONLY on the specific topic
5. Keep content concise and focused
6. Present information in a logical sequence
7. Use academic tone
8. Distribute citations evenly

STRUCTURE:
- Write 2-3 focused paragraphs
- Each paragraph should directly address the topic
- Start with specific information, not context
- Support claims with evidence
- NO concluding remarks
- NO transition sentences"""

            user_content = f"""Generate focused subsection content about: {section_title}

Topic Details: {section_content}

Available Sources:
{context}

Requirements:
1. 2-3 paragraphs maximum
2. {max(1, self.max_citations_per_subsection//4)}-{self.max_citations_per_subsection} citations
3. {self.min_subsection_length}-{self.max_subsection_length} words
4. Start with specific information
5. End with substantive content
6. Distribute citations evenly
7. NO transitions
8. NO meta-references
9. NO introductions or conclusions
10. NO subsection divisions

Generate the subsection content:"""

        else:
            # Keep existing main section prompt
            system_content = """You are a research content generator specializing in academic writing. Your task is to generate well-structured, 
academic content for main sections of a research report. You should:
1. Use formal academic language
2. Integrate information from multiple sources
3. Use proper citation format [1], [2], etc.
4. Maintain logical flow and coherence
5. Be precise and specific
6. Use technical terms appropriately
7. Stay within word limits
8. Ensure balanced use of citations
9. Focus on key findings and insights
10. Include proper section structure"""

            user_content = f"""Generate content for the following main section of a research report.

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
9. Ensure logical flow between paragraphs"""
            
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=BASE_MODEL,
                messages=messages,
                temperature=0.4 if is_subsection else 0.7,  # Lower temperature for subsections
                max_tokens=800 if is_subsection else 2000  # Fewer tokens for subsections
            )
            
            content = response.choices[0].message.content.strip()
            
            # Validate content
            if not self._validate_content(content, is_subsection):
                logger.warning(f"Generated content for {'subsection' if is_subsection else 'section'} {section_title} failed validation")
                logger.info("Attempting content generation again with adjusted parameters")
                # Try one more time with adjusted parameters
                messages[1]["content"] += "\n\nNote: Previous attempt failed validation. Please ensure proper formatting and content requirements are met."
                response = self.client.chat.completions.create(
                    model=BASE_MODEL,
                    messages=messages,
                    temperature=0.3 if is_subsection else 0.5,  # Even lower temperature for retry
                    max_tokens=600 if is_subsection else 1500
                )
                content = response.choices[0].message.content.strip()
                if not self._validate_content(content, is_subsection):
                    logger.warning("Second attempt at content generation also failed validation")
                    return ""
                else:
                    logger.info("Second attempt at content generation succeeded")
            else:
                logger.info("Content generation succeeded on first attempt")
                
            return content
            
        except Exception as e:
            logger.error(f"Error generating content for {'subsection' if is_subsection else 'section'} {section_title}: {str(e)}")
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