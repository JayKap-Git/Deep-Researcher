# Deep Researcher

A comprehensive research assistant that helps gather, analyze, and synthesize information from multiple sources to generate detailed research reports.

## Features

- **Query Clarification**: Interactive clarification of research queries to ensure precise understanding
- **Multi-Source Data Gathering**: Collects information from:
  - arXiv research papers
  - Web articles (via DuckDuckGo, Google, and Tavily)
  - User-uploaded documents
- **Knowledge Base**: Maintains a vectorized knowledge base for efficient information retrieval
- **Research Planning**: Generates structured research outlines
- **Report Generation**: Synthesizes comprehensive research reports with proper citations
- **Export Options**: Exports reports in various formats (PDF, DOCX, etc.)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deep-researcher.git
cd deep-researcher
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
```

## Usage

1. Run the research pipeline:
```bash
python test_research_new.py
```

2. Enter your research query when prompted.

3. The system will:
   - Clarify your query through interactive conversation
   - Generate a research outline
   - Gather relevant information from multiple sources
   - Build a knowledge base
   - Generate a comprehensive report
   - Export the report in your preferred format

## Project Structure

```
backend/
├── __init__.py
├── answer.py           # Answer generation and synthesis
├── clarify.py          # Query clarification
├── config.py           # Configuration settings
├── export.py           # Report export functionality
├── gather.py           # Data gathering from multiple sources
├── knowledge_base.py   # Knowledge base management
├── model_selector.py   # Model selection for different tasks
├── planner.py          # Research planning and outlining
├── research_answer.py  # Research-specific answer generation
└── vectorstore.py      # Vector storage implementation
```

## Dependencies

- OpenAI API for language models
- DuckDuckGo, Google, and Tavily for web search
- arXiv API for research papers
- ChromaDB for vector storage
- Various document processing libraries (PyPDF2, python-docx)
- LangChain for LLM integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for their language models
- DuckDuckGo, Google, and Tavily for search APIs
- arXiv for research paper access
- The open-source community for various libraries and tools 