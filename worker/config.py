"""Worker configuration."""
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ============================================
# API Server Configuration (read from .env, same as app)
# ============================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ============================================
# Worker Specific Configuration
# ============================================
_worker_api_host = "localhost" if API_HOST == "0.0.0.0" else API_HOST
API_BASE_URL = f"http://{_worker_api_host}:{API_PORT}"

WORKER_POLL_INTERVAL = int(os.getenv("WORKER_POLL_INTERVAL", "5"))  # seconds
WORKER_MAX_RETRIES = int(os.getenv("WORKER_MAX_RETRIES", "3"))

# ============================================
# Full-Text Search Configuration
# ============================================
MEILISEARCH_HOST = os.getenv("MEILISEARCH_HOST", "http://localhost:7700")
MEILISEARCH_API_KEY = os.getenv("MEILISEARCH_API_KEY")

# ============================================
# Embedding Configuration (read from .env, same as app)
# ============================================
EMBEDDING_URI = os.getenv("EMBEDDING_URI", "openai/text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ============================================
# LLM Configuration (for extractors)
# ============================================
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, bailian
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

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
# Vector Store Configuration (read from .env, same as app)
# ============================================
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "qdrant")
VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST", "localhost")
VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT", "6333"))
VECTOR_STORE_GRPC_PORT = int(os.getenv("VECTOR_STORE_GRPC_PORT", "6334"))

# ============================================
# Graph Database Configuration
# ============================================
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", "6379"))

# ============================================
# HuggingFace Configuration
# ============================================
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")


# ============================================
# Configuration Validation
# ============================================
def validate_services():
    """Validate critical services and API keys by actually calling them."""
    from rich.console import Console
    import requests
    from openai import OpenAI
    
    console = Console()
    errors = []
    
    # 1. Validate OpenAI API Key
    if OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            client.models.list()
        except Exception as e:
            errors.append(f"OpenAI API Key invalid: {str(e)}")
    else:
        errors.append("OPENAI_API_KEY is not set")
    
    # 2. Validate Vector Store (Qdrant)
    try:
        response = requests.get(f"http://{VECTOR_STORE_HOST}:{VECTOR_STORE_PORT}/collections", timeout=2)
        if response.status_code != 200:
            errors.append(f"Vector Store unreachable: HTTP {response.status_code}")
    except Exception as e:
        errors.append(f"Vector Store unreachable: {str(e)}")
    
    # 3. Validate Meilisearch
    try:
        response = requests.get(f"{MEILISEARCH_HOST}/health", timeout=2)
        if response.status_code != 200:
            errors.append(f"Meilisearch unreachable: HTTP {response.status_code}")
    except Exception as e:
        errors.append(f"Meilisearch unreachable: {str(e)}")
    
    # 4. Check required config
    if not EMBEDDING_URI:
        errors.append("EMBEDDING_URI is not set")
    if not LLM_PROVIDER:
        errors.append("LLM_PROVIDER is not set")
    if not LLM_MODEL:
        errors.append("LLM_MODEL is not set")
    
    if errors:
        console.print("\n[bold red]Configuration/Service Validation Failed:[/bold red]")
        for error in errors:
            console.print(f"  ✗ {error}")
        console.print("\n[yellow]Please check your .env file and ensure all services are running.[/yellow]\n")
        raise RuntimeError(f"Service validation failed: {len(errors)} error(s)")
