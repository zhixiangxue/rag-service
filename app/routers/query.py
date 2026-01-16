"""Query API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Tuple
from cachetools import cached, TTLCache
from cachetools.keys import hashkey
import threading

from ..schemas import QueryRequest, UnitResult, ApiResponse
from ..database import get_connection
from .. import config
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers.basic import VectorRetriever

router = APIRouter(prefix="/datasets", tags=["query"])

# Global cache for dataset info
_dataset_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes TTL
_cache_lock = threading.RLock()


@cached(_dataset_cache, key=lambda dataset_id: hashkey(dataset_id), lock=_cache_lock)
def get_dataset_info(dataset_id: str) -> Tuple[str, str]:
    """Get dataset collection_name and engine from database (with cache).
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Tuple of (collection_name, engine)
        
    Raises:
        HTTPException: If dataset not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, engine FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return row["name"], row["engine"]


def clear_dataset_cache(dataset_id: str):
    """Clear cache for a dataset.
    
    Should be called when dataset is updated or deleted.
    """
    # Generate the same cache key used by @cached decorator
    cache_key = hashkey(dataset_id)
    with _cache_lock:
        _dataset_cache.pop(cache_key, None)


@router.post("/{dataset_id}/query/vector", response_model=ApiResponse[List[UnitResult]])
async def query_vector(dataset_id: str, request: QueryRequest):
    """Vector search for relevant units.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, filters, etc.
    
    Returns:
        List of relevant units with scores
    """
    try:
        # Get dataset info from cache or database
        collection_name, engine = get_dataset_info(dataset_id)
        # Initialize embedder
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        
        # Initialize vector store with timeout
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=embedder,
            timeout=60
        )
        
        # Initialize retriever
        retriever = VectorRetriever(vector_store=vector_store)
        
        # Perform search
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Build response
        unit_results = []
        for result in results:
            # Extract metadata - handle both dict and UnitMetadata object
            metadata = result.metadata if isinstance(result.metadata, dict) else result.metadata.__dict__
            unit_results.append(UnitResult(
                unit_id=result.unit_id,
                doc_id=metadata.get("doc_id", ""),
                score=result.score,
                content=result.content,
                metadata=metadata,
                document_info=None
            ))
        
        return ApiResponse(
            success=True,
            code=200,
            data=unit_results
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@router.post("/{dataset_id}/query/fulltext", response_model=ApiResponse[List[UnitResult]])
async def query_fulltext(dataset_id: str, request: QueryRequest):
    """Full-text search for relevant units.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, etc.
    
    Returns:
        List of relevant units with scores
    """
    # Get dataset info
    collection_name, engine = get_dataset_info(dataset_id)
    
    # TODO: Implement fulltext search with Meilisearch
    raise HTTPException(status_code=501, detail="Fulltext search not implemented yet")


@router.post("/{dataset_id}/query/fusion", response_model=ApiResponse[List[UnitResult]])
async def query_fusion(dataset_id: str, request: QueryRequest):
    """Fusion search combining vector and fulltext search.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, weights, etc.
    
    Returns:
        List of relevant units with combined scores
    """
    # Get dataset info
    collection_name, engine = get_dataset_info(dataset_id)
    
    # TODO: Implement fusion search
    raise HTTPException(status_code=501, detail="Fusion search not implemented yet")
