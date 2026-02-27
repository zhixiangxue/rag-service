"""Query API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Tuple, Dict, Any, Optional
from cachetools import cached, TTLCache
from cachetools.keys import hashkey
import threading

from ..schemas import QueryRequest, UnitResponse, ApiResponse, TreeQueryRequest
from ..database import get_connection
from .. import config
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers.basic import VectorRetriever, FullTextRetriever
from zag.postprocessors import Reranker
from zag.schemas import LODLevel

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
        HTTPException: If dataset not found (only when DEFAULT_COLLECTION_NAME is not configured)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, engine FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        # TODO: Remove this fallback when dataset management is fully implemented in business
        # Currently, the business doesn't have dataset management yet, so we fallback to a
        # hardcoded collection name. This allows the API to work without proper dataset setup.
        # When dataset management is ready, remove the fallback and raise 404 instead.
        if config.DEFAULT_COLLECTION_NAME:
            return config.DEFAULT_COLLECTION_NAME, config.DEFAULT_VECTOR_ENGINE
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


def _rewrite_for_fulltext(query: str) -> str:
    """Rewrite natural language query to keyword form for BM25 fulltext search.

    Falls back to the original query on any error to avoid blocking the pipeline.
    """
    try:
        import chak
        conv = chak.Conversation("openai/gpt-4o-mini", api_key=config.OPENAI_API_KEY)
        prompt = (
            "Extract the most important search keywords from the following question "
            "for a full-text search engine (BM25). "
            "Output only the keywords separated by spaces, no punctuation, no explanation.\n\n"
            f"Question: {query}"
        )
        response = conv.send(prompt)
        keywords = response.content.strip()
        return keywords if keywords else query
    except Exception as e:
        print(f"[WARN] fulltext query rewrite failed, using original: {e}")
        return query


def _rerank_units(query: str, units: list[Any], top_k: Optional[int] = None) -> list[Any]:
    """Apply reranker to units.

    Uses RERANKER_URI from app config. On any error, returns the original
    units to avoid breaking the query pipeline.
    """
    if not units:
        return []

    try:
        reranker = Reranker(config.RERANKER_URI, api_key=config.COHERE_API_KEY)
        return reranker.rerank(query, units, top_k=top_k)
    except Exception as e:
        # Best-effort reranking: log and fall back to original units
        print(f"[WARN] Reranking failed, using original units: {e}")
        return units


async def _perform_vector_query(dataset_id: str, request: QueryRequest) -> ApiResponse[List[UnitResponse]]:
    """Internal function to perform vector search.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, filters, etc.
    
    Returns:
        API response with list of relevant units with scores
    """
    try:
        # Get dataset info from cache or database
        collection_name, engine = get_dataset_info(dataset_id)
        
        # Initialize embedder
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        
        # Initialize vector store
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
        units = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Rerank units (best-effort)
        units = _rerank_units(request.query, units, top_k=request.top_k)
        
        # Build response
        unit_responses: List[UnitResponse] = []
        for unit in units:
            # Extract metadata - handle both dict and UnitMetadata object
            metadata = unit.metadata if isinstance(unit.metadata, dict) else unit.metadata.__dict__
            unit_responses.append(UnitResponse(
                unit_id=unit.unit_id,
                unit_type=unit.unit_type,
                content=unit.content,
                metadata=metadata,
                doc_id=unit.doc_id,
                score=unit.score
            ))
        
        return ApiResponse(
            success=True,
            code=200,
            data=unit_responses
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")


@router.post("/{dataset_id}/query", response_model=ApiResponse[List[UnitResponse]])
async def query(dataset_id: str, request: QueryRequest):
    """Query for relevant units (default: vector search).
    
    This is the default query endpoint that uses vector search.
    For specific search types, use /query/vector, /query/fulltext, or /query/fusion.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, filters, etc.
    
    Returns:
        List of relevant units with scores
    """
    return await _perform_vector_query(dataset_id, request)


@router.post("/{dataset_id}/query/vector", response_model=ApiResponse[List[UnitResponse]])
async def query_vector(dataset_id: str, request: QueryRequest):
    """Vector search for relevant units.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, filters, etc.
    
    Returns:
        List of relevant units with scores
    """
    return await _perform_vector_query(dataset_id, request)


@router.post("/{dataset_id}/query/fulltext", response_model=ApiResponse[List[UnitResponse]])
async def query_fulltext(dataset_id: str, request: QueryRequest):
    """Full-text search for relevant units.
    
    Args:
        dataset_id: Dataset ID
        request: Query request with query text, top_k, etc.
    
    Returns:
        List of relevant units with scores
    """
    try:
        # Get dataset info from cache or database
        collection_name, engine = get_dataset_info(dataset_id)
        
        # Initialize full-text retriever (index name mirrors collection_name convention)
        retriever = FullTextRetriever(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            top_k=request.top_k or 10
        )
        
        # Perform full-text search
        units = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Rerank units (best-effort)
        units = _rerank_units(request.query, units, top_k=request.top_k)
        
        # Build response
        unit_responses: List[UnitResponse] = []
        for unit in units:
            metadata = unit.metadata if isinstance(unit.metadata, dict) else unit.metadata.__dict__
            tree = unit.get_view(LODLevel.HIGH) if unit.is_lod else None
            unit_responses.append(UnitResponse(
                unit_id=unit.unit_id,
                unit_type=unit.unit_type,
                content=unit.content,
                metadata=metadata,
                doc_id=unit.doc_id,
                score=unit.score,
                tree=tree if isinstance(tree, dict) else None,
            ))
        
        return ApiResponse(
            success=True,
            code=200,
            data=unit_responses
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fulltext search failed: {str(e)}")


@router.post("/{dataset_id}/query/fusion", response_model=ApiResponse[List[UnitResponse]])
async def query_fusion(dataset_id: str, request: QueryRequest):
    """Fusion search combining vector and fulltext search.

    Uses vector (Qdrant) and full-text (Meilisearch) retrievers as two
    independent recall channels, unions their candidates, then applies the
    unified reranker on top to produce the final ranking.
    """
    try:
        # Get dataset info
        collection_name, engine = get_dataset_info(dataset_id)

        # Initialize embedder and vector store
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=embedder,
            timeout=60,
        )

        # Base retrievers
        vector_retriever = VectorRetriever(vector_store=vector_store)
        fulltext_retriever = FullTextRetriever(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            top_k=request.top_k or 10,
        )

        # Determine recall sizes
        api_top_k = request.top_k or 5
        recall_top_k = api_top_k * 2

        # Recall from vector and fulltext independently
        vector_units = vector_retriever.retrieve(
            query=request.query,
            top_k=recall_top_k,
            filters=request.filters,
        )
        
        # Rewrite natural language query to keywords for BM25 fulltext recall
        ft_query = request.fulltext_query or _rewrite_for_fulltext(request.query)

        fulltext_units = fulltext_retriever.retrieve(
            query=ft_query,
            top_k=recall_top_k,
            filters=request.filters,
        )

        # Union candidates by unit_id (vector has priority if duplicates)
        candidates: Dict[str, Any] = {}
        for unit in fulltext_units + vector_units:
            candidates[unit.unit_id] = unit

        units = list(candidates.values())

        # Rerank fused candidates (best-effort)
        units = _rerank_units(request.query, units, top_k=api_top_k)

        # Build response
        unit_responses: List[UnitResponse] = []
        for unit in units:
            metadata = unit.metadata if isinstance(unit.metadata, dict) else unit.metadata.__dict__
            tree = unit.get_view(LODLevel.HIGH) if getattr(unit, "is_lod", False) else None
            unit_responses.append(
                UnitResponse(
                    unit_id=unit.unit_id,
                    unit_type=unit.unit_type,
                    content=unit.content,
                    metadata=metadata,
                    doc_id=unit.doc_id,
                    score=unit.score,
                    tree=tree if isinstance(tree, dict) else None,
                )
            )

        return ApiResponse(success=True, code=200, data=unit_responses)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fusion search failed: {str(e)}")


async def resolve_lod_ids(
    request: "TreeQueryRequest",
    vector_store
) -> Dict[str, Optional[str]]:
    """Resolve LOD unit_id and doc_id from request parameters.

    Returns {"unit_id": ..., "doc_id": ...}.

    Four cases:
    1. unit_id only  -> fetch unit to get real doc_id
    2. doc_id only   -> fetch LOD unit via doc_id + mode filter
    3. both provided -> get by unit_id, validate doc_id matches
    4. neither       -> rejected by schema validator before reaching here
    """
    from zag.schemas import ProcessingMode

    if request.unit_id and not request.doc_id:
        # Case 1: unit_id only - fetch unit to get doc_id
        units = vector_store.get([request.unit_id])
        if not units:
            raise ValueError(f"Unit not found: {request.unit_id}")
        return {"unit_id": request.unit_id, "doc_id": units[0].doc_id}

    if request.doc_id and not request.unit_id:
        # Case 2: doc_id only - targeted fetch, O(1)
        units = await vector_store.afetch({
            "doc_id": request.doc_id,
            "metadata.custom.mode": ProcessingMode.LOD
        })
        if not units:
            raise ValueError(f"No LOD unit found for doc_id: {request.doc_id}")
        return {"unit_id": units[0].unit_id, "doc_id": request.doc_id}

    # Case 3: both provided - use unit_id but validate doc_id consistency
    units = vector_store.get([request.unit_id])
    if not units:
        raise ValueError(f"Unit not found: {request.unit_id}")
    if units[0].doc_id != request.doc_id:
        raise ValueError(
            f"unit_id '{request.unit_id}' belongs to doc_id '{units[0].doc_id}', "
            f"not '{request.doc_id}'"
        )
    return {"unit_id": request.unit_id, "doc_id": request.doc_id}


@router.post("/{dataset_id}/query/tree/simple", response_model=ApiResponse[Dict[str, Any]])
async def query_tree_simple(dataset_id: str, request: TreeQueryRequest):
    """Tree query using SimpleRetriever.
    
    Args:
        dataset_id: Dataset ID
        request: Tree query request with query, unit_id, max_depth
    
    Returns:
        Tree retrieval result with nodes and path
    """
    try:
        # Get dataset info
        collection_name, engine = get_dataset_info(dataset_id)
        
        # Initialize embedder
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        
        # Initialize vector store
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=embedder,
            timeout=60
        )
        
        # Initialize SimpleRetriever
        from zag.retrievers.tree import SimpleRetriever
        retriever = SimpleRetriever(
            vector_store=vector_store,
            llm_uri=f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            api_key=config.OPENAI_API_KEY,
            max_depth=request.max_depth
        )
        
        # Resolve lod_unit_id and doc_id from request (supports unit_id or doc_id)
        ids = await resolve_lod_ids(request, vector_store)
        lod_unit_id = ids["unit_id"]
        doc_id = ids["doc_id"]

        # Retrieve (internally fetches unit and parses tree)
        result = await retriever.retrieve(request.query, lod_unit_id)
        
        # Build response
        return ApiResponse(
            success=True,
            code=200,
            data={
                "nodes": [node.model_dump() for node in result.nodes],
                "path": result.path,
                "unit_id": lod_unit_id,
                "doc_id": doc_id
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree query failed: {str(e)}")


@router.post("/{dataset_id}/query/tree/mcts", response_model=ApiResponse[Dict[str, Any]])
async def query_tree_mcts(dataset_id: str, request: TreeQueryRequest):
    """Tree query using MCTSRetriever.
    
    Args:
        dataset_id: Dataset ID
        request: Tree query request with query, unit_id, preset
    
    Returns:
        Tree retrieval result with nodes and path
    """
    try:
        # Get dataset info
        collection_name, engine = get_dataset_info(dataset_id)
        
        # Initialize embedder
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        
        # Initialize vector store
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=embedder,
            timeout=60
        )
        
        # Initialize MCTSRetriever
        from zag.retrievers.tree import MCTSRetriever
        retriever = MCTSRetriever.from_preset(
            preset_name=request.preset,
            api_key=config.OPENAI_API_KEY,
            verbose=False,
            vector_store=vector_store
        )
        
        # Resolve lod_unit_id and doc_id from request (supports unit_id or doc_id)
        ids = await resolve_lod_ids(request, vector_store)
        lod_unit_id = ids["unit_id"]
        doc_id = ids["doc_id"]

        # Retrieve (internally fetches unit and parses tree)
        result = await retriever.retrieve(request.query, lod_unit_id)
        
        # Build response
        return ApiResponse(
            success=True,
            code=200,
            data={
                "nodes": [node.model_dump() for node in result.nodes],
                "path": result.path,
                "unit_id": lod_unit_id,
                "doc_id": doc_id
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree query failed: {str(e)}")


@router.post("/{dataset_id}/query/tree/skeleton", response_model=ApiResponse[Dict[str, Any]])
async def query_tree_skeleton(dataset_id: str, request: TreeQueryRequest):
    """Tree query using SkeletonRetriever.
    
    Args:
        dataset_id: Dataset ID
        request: Tree query request with query, unit_id
    
    Returns:
        Tree retrieval result with nodes and path
    """
    try:
        # Get dataset info
        collection_name, engine = get_dataset_info(dataset_id)
        
        # Initialize embedder
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        
        # Initialize vector store
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=embedder,
            timeout=60
        )
        
        # Initialize SkeletonRetriever
        from zag.retrievers.tree import SkeletonRetriever
        retriever = SkeletonRetriever(
            llm_uri=f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            api_key=config.OPENAI_API_KEY,
            verbose=False,
            vector_store=vector_store
        )

        # Resolve lod_unit_id and doc_id from request (supports unit_id or doc_id)
        ids = await resolve_lod_ids(request, vector_store)
        lod_unit_id = ids["unit_id"]
        doc_id = ids["doc_id"]

        # Retrieve (internally fetches unit and parses tree)
        # mode: "fast" -> use summary, "accurate" -> use full text
        use_full_text = request.mode == "accurate"
        result = await retriever.retrieve(
            request.query,
            lod_unit_id,
            use_full_text=use_full_text
        )
        
        # Build response
        return ApiResponse(
            success=True,
            code=200,
            data={
                "nodes": [node.model_dump() for node in result.nodes],
                "path": result.path,
                "unit_id": lod_unit_id,
                "doc_id": doc_id
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tree query failed: {str(e)}")


