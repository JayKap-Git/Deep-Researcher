"""
Research Answer Generator

This module generates detailed research answers by processing sections and subsections
from gathered research data and using LLM to synthesize comprehensive responses.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from backend.config import OPENAI_API_KEY, BASE_MODEL, MODEL_TEMPERATURE

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
        logger.info(f"Initialized ResearchAnswerGenerator with model: {BASE_MODEL} and temperature: {MODEL_TEMPERATURE}")

    def generate_answer(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Generate a comprehensive answer based on the query and documents."""
        if not query:
            raise ValueError("Query cannot be empty")
        if query is None:
            raise ValueError("Query cannot be None")
        if not documents:
            raise ValueError("Documents list cannot be empty")
        if documents is None:
            raise ValueError("Documents cannot be None")
        if not isinstance(documents, list):
            raise ValueError("Documents must be a list")
        if not all(isinstance(doc, Document) for doc in documents):
            raise ValueError("All documents must be instances of Document")

        # Process documents and generate answer
        try:
            # Extract relevant information from documents
            document_texts = [doc.page_content for doc in documents]
            document_sources = [doc.metadata.get("source", "Unknown") for doc in documents]

            # Generate answer using the language model
            prompt = self._create_answer_prompt(query, document_texts)
            response = self.model.predict(prompt)

            # Process and format the response
            answer = self._process_response(response)
            sources = self._format_sources(documents)
            confidence = self._calculate_confidence(response, documents)

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