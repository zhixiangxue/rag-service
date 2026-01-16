"""Health check and statistics API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Dict

from ..schemas import HealthResponse, DatasetStats, ApiResponse
from ..database import get_connection
from .query import get_dataset_info
from .. import config
from qdrant_client import QdrantClient

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and dependencies.
    
    Returns:
        Health status of service and dependencies
    """
    status = "ok"
    dependencies: Dict[str, str] = {}
    
    # Check database
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        dependencies["database"] = "ok"
    except Exception as e:
        dependencies["database"] = f"error: {str(e)}"
        status = "degraded"
    
    # Check vector store
    try:
        client = QdrantClient(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            timeout=10
        )
        client.get_collections()
        dependencies["vector_store"] = "ok"
    except Exception as e:
        dependencies["vector_store"] = f"error: {str(e)}"
        status = "degraded"
    
    return HealthResponse(status=status, dependencies=dependencies)
