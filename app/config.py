"""Application configuration."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# ============================================
# Load .env file with explicit path resolution
# ============================================
# Get the directory where this config.py is located
CONFIG_DIR = Path(__file__).parent.resolve()
RAG_SERVICE_DIR = CONFIG_DIR.parent
ENV_FILE = RAG_SERVICE_DIR / ".env"

# Load .env file from rag-service directory
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
    print(f"[Config] Loaded .env from: {ENV_FILE}")
else:
    print(f"[Config] ERROR: .env file not found at: {ENV_FILE}")
    print("[Config] Please create .env file before starting the service.")
    sys.exit(1)


def require_env(key: str) -> str:
    """Require an environment variable. Exit if not set."""
    value = os.getenv(key)
    if value is None:
        print(f"[Config] ERROR: Required environment variable '{key}' is not set.")
        print(f"[Config] Please add '{key}' to your .env file.")
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
        print(f"[Config] ERROR: {env_key} is not set (required by {context}).")
        print(f"[Config] Please add '{env_key}' to your .env file.")
        sys.exit(1)
    return key

# ============================================
# Database Configuration
# ============================================
# SQLite:  sqlite:///./rag_service.db
# rqlite:  http://host:4001  or  https://host:4001
DATABASE_URI = require_env("DATABASE_URI")

# ============================================
# Vector Store Configuration
# ============================================
VECTOR_STORE_TYPE = require_env("VECTOR_STORE_TYPE")
VECTOR_STORE_HOST = require_env("VECTOR_STORE_HOST")
VECTOR_STORE_PORT = int(require_env("VECTOR_STORE_PORT"))
VECTOR_STORE_GRPC_PORT = int(require_env("VECTOR_STORE_GRPC_PORT"))
VECTOR_STORE_API_KEY = os.getenv("VECTOR_STORE_API_KEY") or None  # Optional

# ============================================
# Full-Text Search Configuration
# ============================================
MEILISEARCH_HOST = require_env("MEILISEARCH_HOST")
MEILISEARCH_API_KEY = require_env("MEILISEARCH_API_KEY")

# ============================================
# Graph Database Configuration (FalkorDB)
# ============================================
FALKORDB_HOST = require_env("FALKORDB_HOST")
FALKORDB_PORT = int(require_env("FALKORDB_PORT"))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD") or None  # Optional

# ============================================
# Embedding Configuration
# ============================================
EMBEDDING_URI = require_env("EMBEDDING_URI")
_EMBEDDING_PROVIDER = EMBEDDING_URI.split("/")[0].split("@")[0].lower()
EMBEDDING_API_KEY = _resolve_api_key(_EMBEDDING_PROVIDER, f"EMBEDDING_URI={EMBEDDING_URI}")

# ============================================
# Reranker Configuration
# ============================================
RERANKER_URI = require_env("RERANKER_URI")
_RERANKER_PROVIDER = RERANKER_URI.split("/")[0].split("@")[0].lower()
RERANKER_API_KEY = _resolve_api_key(_RERANKER_PROVIDER, f"RERANKER_URI={RERANKER_URI}")

# ============================================
# LLM Configuration
# ============================================
LLM_URI = require_env("LLM_URI")
_LLM_PROVIDER = LLM_URI.split("/")[0].split("@")[0].lower()
LLM_API_KEY = _resolve_api_key(_LLM_PROVIDER, f"LLM_URI={LLM_URI}")

# ============================================
# File Storage Configuration
# ============================================
UPLOAD_DIR = require_env("UPLOAD_DIR")
STORAGE_TYPE = require_env("STORAGE_TYPE")  # local, s3
# S3 Configuration (only when STORAGE_TYPE=s3)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
# AWS Credentials for S3 download
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# ============================================
# PDF Cache Configuration
# ============================================
PDF_FILES_DIR = Path(require_env("PDF_FILES_DIR")).expanduser()
LOCATE_CACHE_DIR = Path(require_env("LOCATE_CACHE_DIR")).expanduser()

# ============================================
# API Server Configuration
# ============================================
API_HOST = require_env("API_HOST")
API_PORT = int(require_env("API_PORT"))

# Service-level access key (empty = no auth)
ACCESS_KEY = os.getenv("ACCESS_KEY", "")

# Public-facing host for file URLs (used by distributed workers)
API_PUBLIC_HOST = os.getenv("API_PUBLIC_HOST") or (
    "localhost" if API_HOST == "0.0.0.0" else API_HOST
)

# ============================================
# Evaluation Service Configuration
# ============================================
EVAL_SERVICE_URL = os.getenv("EVAL_SERVICE_URL")  # Optional

# ============================================
# Redis / Dramatiq Configuration
# ============================================
REDIS_HOST = require_env("REDIS_HOST")
REDIS_PORT = int(require_env("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None  # Optional
