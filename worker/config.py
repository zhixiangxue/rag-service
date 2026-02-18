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


# ============================================
# Configuration Summary & Interactive Confirmation
# ============================================
def print_config_summary():
    """Print configuration summary for user confirmation."""
    print("\n" + "=" * 60)
    print("  RAG Worker Configuration Summary")
    print("=" * 60)
    print(f"\n[API Server]")
    print(f"  API_HOST: {API_HOST}")
    print(f"  API_PORT: {API_PORT}")
    print(f"  API_BASE_URL: {API_BASE_URL}")
    print(f"\n[Worker Settings]")
    print(f"  WORKER_POLL_INTERVAL: {WORKER_POLL_INTERVAL}s")
    print(f"  WORKER_MAX_RETRIES: {WORKER_MAX_RETRIES}")
    print(f"\n[Vector Store]")
    print(f"  VECTOR_STORE_TYPE: {VECTOR_STORE_TYPE}")
    print(f"  VECTOR_STORE_HOST: {VECTOR_STORE_HOST}")
    print(f"  VECTOR_STORE_PORT: {VECTOR_STORE_PORT}")
    print(f"  VECTOR_STORE_GRPC_PORT: {VECTOR_STORE_GRPC_PORT}")
    print(f"\n[Meilisearch]")
    print(f"  MEILISEARCH_HOST: {MEILISEARCH_HOST}")
    print(f"\n[Embedding]")
    print(f"  EMBEDDING_URI: {EMBEDDING_URI}")
    print(f"  OPENAI_API_KEY: {'*' * 10} (set)")
    print(f"\n[LLM]")
    print(f"  LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"  LLM_MODEL: {LLM_MODEL}")
    print(f"\n[Document Processing]")
    print(f"  DOCUMENT_READER: {DOCUMENT_READER}")
    print(f"  MAX_PAGES_PER_PART: {MAX_PAGES_PER_PART}")
    print(f"  USE_GPU: {USE_GPU}")
    print(f"  NUM_THREADS: {NUM_THREADS}")
    print(f"  MAX_CHUNK_TOKENS: {MAX_CHUNK_TOKENS}")
    print(f"  TARGET_TOKEN_SIZE: {TARGET_TOKEN_SIZE}")
    print("=" * 60)


def confirm_config() -> bool:
    """
    Interactive confirmation of configuration.

    Returns:
        True if user confirms, False otherwise
    """
    print_config_summary()

    # Check if running in non-interactive mode (CI/CD, etc.)
    if not sys.stdin.isatty():
        print("\n[Worker Config] Running in non-interactive mode, skipping confirmation.")
        return True

    try:
        response = input("\n[Worker Config] Do you want to proceed with this configuration? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            print("[Worker Config] Configuration confirmed. Starting worker...\n")
            return True
        else:
            print("[Worker Config] Configuration rejected. Exiting.")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n[Worker Config] Interrupted. Exiting.")
        return False


# Run confirmation when this module is imported
# Skip confirmation for worker (runs in background/daemon mode)
# Only run confirmation if explicitly executed as main script
if __name__ == "__main__":
    if not confirm_config():
        sys.exit(1)
