"""
Configuration settings for the Deep Researcher application.
"""

import os
import logging
import logging.config
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any
import chromadb

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model settings
BASE_MODEL = "gpt-4o-mini"  # Default model
MODEL_TEMPERATURE = 0.7     # Default temperature for model responses

# Rate limits
RATE_LIMITS = {
    "arxiv": {
        "requests_per_minute": 30,  # ArXiv's guideline
        "retry_after": 60          # Wait 60 seconds before retrying
    }
}

# File paths
BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [UPLOADS_DIR, EXPORTS_DIR, VECTORSTORE_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o777)  # Set full permissions

# Validate configuration
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Model configurations
BASE_MODEL = "gpt-4o-mini"
MODEL_TEMPERATURE = 0.7

# Search configurations
SEARCH_CONFIGS = {
    "max_results_per_source": 3,
    "max_query_length": 200,
    "min_query_length": 10,
    "max_web_results": 5,
    "max_google_results": 5,
    "max_arxiv_results": 5
}

# Vector store configurations
VECTORSTORE_CONFIGS = {
    "collection_name": "research_documents",
    "embedding_dimensions": 1536,
    "persist_directory": str(VECTORSTORE_DIR)
}

# PDF Export configurations
PDF_OPTIONS = {
    "page_size": "A4",
    "margin": 72,  # 1 inch in points
    "font_size": 11,
    "title_font_size": 16,
    "heading_font_size": 14,
    "line_spacing": 1.2,
    "font_name": "Helvetica"
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "research.log",
            "formatter": "standard"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard"
        }
    },
    "root": {
        "handlers": ["file", "console"],
        "level": "INFO"
    }
}

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

def validate_config() -> Dict[str, Any]:
    """
    Validate and return the current configuration status.
    
    Returns:
        Dict containing validation results and warnings
    """
    validation = {
        "api_keys": {
            "openai": bool(OPENAI_API_KEY)
        },
        "directories": {
            "uploads": UPLOADS_DIR.exists(),
            "exports": EXPORTS_DIR.exists(),
            "vectorstore": VECTORSTORE_DIR.exists(),
            "logs": LOGS_DIR.exists()
        },
        "warnings": []
    }
    
    # Check for missing API keys
    if not validation["api_keys"]["openai"]:
        validation["warnings"].append("OpenAI API key is missing")
    
    return validation 