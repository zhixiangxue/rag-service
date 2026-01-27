"""Application configuration."""
import os
from ..env_loader import load_environment

# Load environment variables from .env file
load_environment()

# ============================================
# Database Configuration
# ============================================
DATABASE_PATH = os.getenv("DATABASE_PATH", "./rag_service.db")

# ============================================
# Vector Store Configuration
# ============================================
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "qdrant")
VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST", "localhost")
VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT", "6333"))

# TODO: Remove these default fallback settings when dataset management is fully implemented
# Currently used as fallback when dataset_id is not found in database
DEFAULT_COLLECTION_NAME = os.getenv("DEFAULT_COLLECTION_NAME", "mortgage_guidelines")
DEFAULT_VECTOR_ENGINE = os.getenv("DEFAULT_VECTOR_ENGINE", "qdrant")

# ============================================
# Embedding Configuration
# ============================================
EMBEDDING_URI = os.getenv("EMBEDDING_URI", "openai/text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ============================================
# File Storage Configuration
# ============================================
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "local")  # local, s3
# S3 Configuration (if using S3)
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "us-east-1")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")

# ============================================
# API Server Configuration
# ============================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
