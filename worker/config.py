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


def require_env(key: str) -> str:
    """Require an environment variable. Exit if not set."""
    value = os.getenv(key)
    if value is None:
        print(f"[Worker Config] ERROR: Required environment variable '{key}' is not set.")
        print(f"[Worker Config] Please add '{key}' to your .env file.")
        sys.exit(1)
    return value


def _resolve_api_key(provider: str, context: str) -> str:
    """根据 provider 名称自动解析 {PROVIDER}_API_KEY 环境变量。

    规范: provider 名称 = env var 前缀
      deepseek  → DEEPSEEK_API_KEY
      bailian   → BAILIAN_API_KEY
      openai    → OPENAI_API_KEY
    """
    env_key = f"{provider.upper()}_API_KEY"
    key = os.getenv(env_key)
    if not key:
        print(f"[Worker Config] ERROR: {env_key} is not set (required by {context}).")
        print(f"[Worker Config] Please add '{env_key}' to your .env file.")
        sys.exit(1)
    return key

# ============================================
# API Server Configuration
# ============================================
API_HOST = require_env("API_HOST")
API_PORT = int(require_env("API_PORT"))

# ============================================
# Worker Specific Configuration
# ============================================
_worker_api_host = "localhost" if API_HOST == "0.0.0.0" else API_HOST
API_BASE_URL = f"http://{_worker_api_host}:{API_PORT}"

# Service-level access key (must match app's ACCESS_KEY; empty = no auth)
ACCESS_KEY = os.getenv("ACCESS_KEY", "")
API_HEADERS: dict = {"X-Api-Key": ACCESS_KEY} if ACCESS_KEY else {}

# ============================================
# Full-Text Search Configuration
# ============================================
MEILISEARCH_HOST = require_env("MEILISEARCH_HOST")
MEILISEARCH_API_KEY = require_env("MEILISEARCH_API_KEY")

# ============================================
# Embedding Configuration
# ============================================
EMBEDDING_URI = require_env("EMBEDDING_URI")
# Provider name → {PROVIDER}_API_KEY (e.g. bailian → BAILIAN_API_KEY)
_EMBEDDING_PROVIDER = EMBEDDING_URI.split("/")[0].split("@")[0].lower()
EMBEDDING_API_KEY = _resolve_api_key(_EMBEDDING_PROVIDER, f"EMBEDDING_URI={EMBEDDING_URI}")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Optional: legacy, only if RERANKER_URI uses cohere
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # Required when using claude reader

# ============================================
# LLM Configuration (for extractors)
# ============================================
LLM_URI = require_env("LLM_URI")
# Provider name → {PROVIDER}_API_KEY (e.g. deepseek → DEEPSEEK_API_KEY)
_LLM_PROVIDER = LLM_URI.split("/")[0].split("@")[0].lower()
LLM_API_KEY = _resolve_api_key(_LLM_PROVIDER, f"LLM_URI={LLM_URI}")

# ============================================
# Reranker Configuration
# ============================================
RERANKER_URI = require_env("RERANKER_URI")
_RERANKER_PROVIDER = RERANKER_URI.split("/")[0].split("@")[0].lower()
RERANKER_API_KEY = _resolve_api_key(_RERANKER_PROVIDER, f"RERANKER_URI={RERANKER_URI}")

# ============================================
# Document Processing Configuration
# ============================================
MAX_PAGES_PER_PART = int(require_env("MAX_PAGES_PER_PART"))
USE_GPU = require_env("USE_GPU").lower() == "true"
NUM_THREADS = int(require_env("NUM_THREADS"))
MAX_CHUNK_TOKENS = int(require_env("MAX_CHUNK_TOKENS"))
TABLE_MAX_TOKENS = int(require_env("TABLE_MAX_TOKENS"))
TARGET_TOKEN_SIZE = int(require_env("TARGET_TOKEN_SIZE"))
NUM_KEYWORDS = int(require_env("NUM_KEYWORDS"))

# ============================================
# Vector Store Configuration
# ============================================
VECTOR_STORE_TYPE = require_env("VECTOR_STORE_TYPE")
VECTOR_STORE_HOST = require_env("VECTOR_STORE_HOST")
VECTOR_STORE_PORT = int(require_env("VECTOR_STORE_PORT"))
VECTOR_STORE_GRPC_PORT = int(require_env("VECTOR_STORE_GRPC_PORT"))
VECTOR_STORE_API_KEY = os.getenv("VECTOR_STORE_API_KEY") or None  # Optional

# ============================================
# Graph Database Configuration (FalkorDB)
# ============================================
FALKORDB_HOST = require_env("FALKORDB_HOST")
FALKORDB_PORT = int(require_env("FALKORDB_PORT"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD") or None  # Optional

# ============================================
# Redis / Dramatiq Configuration
# ============================================
REDIS_HOST = require_env("REDIS_HOST")
REDIS_PORT = int(require_env("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None  # Optional

# ============================================
# AWS S3 Configuration (used by task_processor to download S3-hosted files)
# ============================================
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# ============================================
# PDF Cache Configuration
# ============================================
ARCHIVES_DIR = Path(require_env("ARCHIVES_DIR")).expanduser()
