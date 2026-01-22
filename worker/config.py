"""Worker configuration."""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import shared configurations from app
from ..app import config as app_config

# ============================================
# Worker Specific Configuration
# ============================================
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
WORKER_POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))  # seconds
WORKER_MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "3"))

# ============================================
# Full-Text Search Configuration
# ============================================
MEILISEARCH_HOST = os.getenv("MEILISEARCH_HOST", "http://localhost:7700")
MEILISEARCH_API_KEY = os.getenv("MEILISEARCH_API_KEY")

# ============================================
# LLM Configuration (for extractors)
# ============================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, bailian
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Dynamically select LLM API key based on provider
if LLM_PROVIDER == "openai":
    LLM_API_KEY = app_config.OPENAI_API_KEY
elif LLM_PROVIDER == "bailian":
    LLM_API_KEY = app_config.BAILIAN_API_KEY
else:
    LLM_API_KEY = None  # anthropic or other providers

# ============================================
# Reranker Configuration
# ============================================
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "cohere")  # cohere, sentence-transformers
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ============================================
# Document Processing Configuration
# ============================================
# PDF Processing
MAX_PAGES_PER_PART = int(os.getenv("MAX_PAGES_PER_PART", "100"))
PAGE_OVERLAP = int(os.getenv("PAGE_OVERLAP", "2"))

# Reader Configuration
DOCUMENT_READER = os.getenv("DOCUMENT_READER", "docling")  # docling, markitdown, mineru

# Processing Settings (used in document_processor.py)
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
NUM_THREADS = int(os.getenv("NUM_THREADS", "8"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "1200"))
TABLE_MAX_TOKENS = int(os.getenv("TABLE_MAX_TOKENS", "1500"))
TARGET_TOKEN_SIZE = int(os.getenv("TARGET_TOKEN_SIZE", "800"))
NUM_KEYWORDS = int(os.getenv("NUM_KEYWORDS", "5"))

# ============================================
# Reuse from App Config
# ============================================
VECTOR_STORE_TYPE = app_config.VECTOR_STORE_TYPE
VECTOR_STORE_HOST = app_config.VECTOR_STORE_HOST
VECTOR_STORE_PORT = app_config.VECTOR_STORE_PORT

EMBEDDING_URI = app_config.EMBEDDING_URI
OPENAI_API_KEY = app_config.OPENAI_API_KEY

# ============================================
# HuggingFace Configuration
# ============================================
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
