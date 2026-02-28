"""Worker configuration."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# Load .env file with explicit path resolution
# ============================================
# Get the directory where this config.py is located
CONFIG_DIR = Path(__file__).parent.resolve()
WORKER_DIR = CONFIG_DIR
RAG_SERVICE_DIR = WORKER_DIR.parent
ENV_FILE = RAG_SERVICE_DIR / ".env"

# Load .env file from rag-service directory
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
    print(f"[Worker Config] Loaded .env from: {ENV_FILE}")
else:
    print(f"[Worker Config] ERROR: .env file not found at: {ENV_FILE}")
    print("[Worker Config] Please create .env file before starting the worker.")
    sys.exit(1)


def require_env(key: str, default_value: str = None) -> str:
    """
    Require an environment variable to be set.
    
    Args:
        key: Environment variable name
        default_value: Optional default value (if None, the variable is required)
        
    Returns:
        The environment variable value
        
    Raises:
        SystemExit: If required variable is not set
    """
    value = os.getenv(key, default_value)
    if value is None:
        print(f"[Worker Config] ERROR: Required environment variable '{key}' is not set.")
        print(f"[Worker Config] Please add '{key}' to your .env file.")
        sys.exit(1)
    return value

# ============================================
# API Server Configuration (read from .env, same as app)
# ============================================
API_HOST = require_env("API_HOST", "0.0.0.0")
API_PORT = int(require_env("API_PORT", "8000"))

# ============================================
# Worker Specific Configuration
# ============================================
_worker_api_host = "localhost" if API_HOST == "0.0.0.0" else API_HOST
API_BASE_URL = f"http://{_worker_api_host}:{API_PORT}"

WORKER_POLL_INTERVAL = int(require_env("WORKER_POLL_INTERVAL", "5"))  # seconds
WORKER_MAX_RETRIES = int(require_env("WORKER_MAX_RETRIES", "3"))

# ============================================
# Full-Text Search Configuration
# ============================================
MEILISEARCH_HOST = require_env("MEILISEARCH_HOST", "http://localhost:7700")
MEILISEARCH_API_KEY = os.getenv("MEILISEARCH_API_KEY")  # Optional

# ============================================
# Embedding Configuration (read from .env, same as app)
# ============================================
EMBEDDING_URI = require_env("EMBEDDING_URI", "openai/text-embedding-3-small")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")  # Optional
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Optional

# ============================================
# LLM Configuration (for extractors)
# ============================================
LLM_PROVIDER = require_env("LLM_PROVIDER", "openai")  # openai, anthropic, bailian
LLM_MODEL = require_env("LLM_MODEL", "gpt-4o-mini")

# Dynamically select LLM API key based on provider
if LLM_PROVIDER == "openai":
    LLM_API_KEY = OPENAI_API_KEY
elif LLM_PROVIDER == "bailian":
    LLM_API_KEY = BAILIAN_API_KEY
else:
    LLM_API_KEY = None  # anthropic or other providers

# ============================================
# Reranker Configuration
# ============================================
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "cohere")  # cohere, sentence-transformers

# ============================================
# Document Processing Configuration
# ============================================
# PDF Processing
MAX_PAGES_PER_PART = int(require_env("MAX_PAGES_PER_PART", "100"))

# Reader Configuration
DOCUMENT_READER = require_env("DOCUMENT_READER", "docling")  # docling, markitdown, mineru

# Processing Settings (used in document_processor.py)
USE_GPU = require_env("USE_GPU", "true").lower() == "true"
NUM_THREADS = int(require_env("NUM_THREADS", "8"))
MAX_CHUNK_TOKENS = int(require_env("MAX_CHUNK_TOKENS", "1200"))
TABLE_MAX_TOKENS = int(require_env("TABLE_MAX_TOKENS", "1500"))
TARGET_TOKEN_SIZE = int(require_env("TARGET_TOKEN_SIZE", "800"))
NUM_KEYWORDS = int(require_env("NUM_KEYWORDS", "5"))

# ============================================
# Vector Store Configuration (read from .env, same as app)
# ============================================
VECTOR_STORE_TYPE = require_env("VECTOR_STORE_TYPE", "qdrant")
VECTOR_STORE_HOST = require_env("VECTOR_STORE_HOST", "localhost")
VECTOR_STORE_PORT = int(require_env("VECTOR_STORE_PORT", "6333"))
VECTOR_STORE_GRPC_PORT = int(require_env("VECTOR_STORE_GRPC_PORT", "6334"))

# ============================================
# Graph Database Configuration
# ============================================
FALKORDB_HOST = require_env("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(require_env("FALKORDB_PORT", "6379"))

# ============================================
# HuggingFace Configuration
# ============================================
HF_ENDPOINT = require_env("HF_ENDPOINT", "https://hf-mirror.com")
