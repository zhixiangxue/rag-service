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
        print(f"[Config] ERROR: Required environment variable '{key}' is not set.")
        print(f"[Config] Please add '{key}' to your .env file.")
        sys.exit(1)
    return value

# ============================================
# Database Configuration
# ============================================
DATABASE_PATH = require_env("DATABASE_PATH")

# ============================================
# Vector Store Configuration
# ============================================
VECTOR_STORE_TYPE = require_env("VECTOR_STORE_TYPE")
VECTOR_STORE_HOST = require_env("VECTOR_STORE_HOST")
VECTOR_STORE_PORT = int(require_env("VECTOR_STORE_PORT"))
VECTOR_STORE_GRPC_PORT = int(require_env("VECTOR_STORE_GRPC_PORT"))

# ============================================
# Graph Database Configuration
# ============================================
#FALKORDB_HOST = require_env("FALKORDB_HOST")
#FALKORDB_PORT = int(require_env("FALKORDB_PORT"))

# TODO: Remove these default fallback settings when dataset management is fully implemented
# Currently used as fallback when dataset_id is not found in database
DEFAULT_COLLECTION_NAME = require_env("DEFAULT_COLLECTION_NAME")
DEFAULT_VECTOR_ENGINE = require_env("DEFAULT_VECTOR_ENGINE")

# ============================================
# Embedding Configuration
# ============================================
EMBEDDING_URI = require_env("EMBEDDING_URI")
OPENAI_API_KEY = require_env("OPENAI_API_KEY")
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")  # Optional
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Optional

# ============================================
# LLM Configuration
# ============================================
LLM_PROVIDER = require_env("LLM_PROVIDER")
LLM_MODEL = require_env("LLM_MODEL")

# ============================================
# File Storage Configuration
# ============================================
UPLOAD_DIR = require_env("UPLOAD_DIR")
STORAGE_TYPE = require_env("STORAGE_TYPE")  # local, s3
# S3 Configuration (if using S3)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# AWS Credentials for S3 download
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# ============================================
# API Server Configuration
# ============================================
API_HOST = require_env("API_HOST")
API_PORT = int(require_env("API_PORT"))

# Public-facing host for file URLs (used by distributed workers)
# If not set, defaults to API_HOST (unless API_HOST is 0.0.0.0, then uses localhost)
API_PUBLIC_HOST = os.getenv("API_PUBLIC_HOST") or (
    "localhost" if API_HOST == "0.0.0.0" else API_HOST
)


# ============================================
# Configuration Summary & Interactive Confirmation
# ============================================
def print_config_summary():
    """Print configuration summary for user confirmation."""
    print("\n" + "=" * 60)
    print("  RAG Service Configuration Summary")
    print("=" * 60)
    print(f"\n[Database]")
    print(f"  DATABASE_PATH: {DATABASE_PATH}")
    print(f"\n[Vector Store]")
    print(f"  VECTOR_STORE_TYPE: {VECTOR_STORE_TYPE}")
    print(f"  VECTOR_STORE_HOST: {VECTOR_STORE_HOST}")
    print(f"  VECTOR_STORE_PORT: {VECTOR_STORE_PORT}")
    print(f"  VECTOR_STORE_GRPC_PORT: {VECTOR_STORE_GRPC_PORT}")
    print(f"\n[Embedding]")
    print(f"  EMBEDDING_URI: {EMBEDDING_URI}")
    print(f"  OPENAI_API_KEY: {'*' * 10} (set)")
    print(f"\n[LLM]")
    print(f"  LLM_PROVIDER: {LLM_PROVIDER}")
    print(f"  LLM_MODEL: {LLM_MODEL}")
    print(f"\n[API Server]")
    print(f"  API_HOST: {API_HOST}")
    print(f"  API_PORT: {API_PORT}")
    print(f"\n[File Storage]")
    print(f"  UPLOAD_DIR: {UPLOAD_DIR}")
    print(f"  STORAGE_TYPE: {STORAGE_TYPE}")
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
        print("\n[Config] Running in non-interactive mode, skipping confirmation.")
        return True

    try:
        response = input("\n[Config] Do you want to proceed with this configuration? [Y/n]: ").strip().lower()
        if response in ('', 'y', 'yes'):
            print("[Config] Configuration confirmed. Starting service...\n")
            return True
        else:
            print("[Config] Configuration rejected. Exiting.")
            return False
    except (EOFError, KeyboardInterrupt):
        print("\n[Config] Interrupted. Exiting.")
        return False


# Run confirmation when this module is imported
if __name__ != "__main__":
    if not confirm_config():
        sys.exit(1)
