"""Query API endpoints."""
import logging
from fastapi import APIRouter, HTTPException
from typing import List, Tuple, Dict, Any, Optional
from cachetools import cached, TTLCache
from cachetools.keys import hashkey
import threading
import traceback
import asyncio

logger = logging.getLogger("rag.query")

from ..schemas import QueryRequest, UnitResponse, ApiResponse, TreeQueryRequest, _extract_tags
from ..database import get_connection
from .. import config
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers.basic import VectorRetriever, FullTextRetriever
from zag.retrievers.composite import QueryFusionRetriever, FusionMode, QueryRewriteRetriever
from zag.postprocessors import Reranker
from zag.schemas import LODLevel

router = APIRouter(prefix="/datasets", tags=["query"])

# Global cache for dataset info
_dataset_cache = TTLCache(maxsize=1000, ttl=300)  # 5 minutes TTL
_cache_lock = threading.RLock()

# Singletons: embedder and reranker are config-driven and shared across all requests.
# vector_store and fulltext_retriever depend on collection_name so they are keyed by collection.
# No lock needed: worst case is two threads both init and one overwrites the other —
# all these objects are stateless connection clients, double-init is harmless.
_embedder: Optional[Any] = None
_reranker: Optional[Any] = None
_vector_store_cache: Dict[str, Any] = {}
_fulltext_retriever_cache: Dict[str, Any] = {}

# Limits concurrent Cohere rerank calls to prevent burst 429 / connection failures.
# Created lazily inside the event loop (asyncio.Semaphore must run in a loop context).
# Each uvicorn worker process has its own semaphore instance.
_rerank_semaphore: Optional[asyncio.Semaphore] = None


def _get_embedder() -> Any:
    """Return a cached Embedder singleton (created once, reused for all requests)."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
    return _embedder


def _get_vector_store(collection_name: str) -> Any:
    """Return a cached QdrantVectorStore for the given collection.

    gRPC channel setup to a remote host takes ~2-7s; caching the client
    eliminates that overhead on every query request.
    """
    if collection_name not in _vector_store_cache:
        _vector_store_cache[collection_name] = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            grpc_port=config.VECTOR_STORE_GRPC_PORT,
            prefer_grpc=True,
            collection_name=collection_name,
            embedder=_get_embedder(),
            timeout=60,
        )
    return _vector_store_cache[collection_name]


def _get_fulltext_retriever(collection_name: str, top_k: int = 10) -> Any:
    """Return a cached FullTextRetriever for the given collection.

    FullTextRetriever.__init__ calls meilisearch.Client.get_index() which is a
    synchronous HTTP request.  Caching avoids that blocking call on every query.
    """
    if collection_name not in _fulltext_retriever_cache:
        _fulltext_retriever_cache[collection_name] = FullTextRetriever(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            top_k=top_k,
        )
    return _fulltext_retriever_cache[collection_name]


def _get_reranker() -> Any:
    """Return a cached Reranker singleton."""
    global _reranker
    if _reranker is None:
        _reranker = Reranker(config.RERANKER_URI, api_key=config.COHERE_API_KEY)
    return _reranker


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



async def _rerank_units(
    query: str,
    units: list[Any],
    top_k: Optional[int] = None,
) -> list[Any]:
    """Apply reranker to units (async, does not block the event loop).

    Uses reranker.arerank() for providers that support async (e.g. Cohere AsyncClient).
    Falls back to asyncio.to_thread for providers without async support.
    A per-worker Semaphore caps concurrent Cohere calls at 15 to prevent burst
    rate-limit errors (429) and connection failures under high load.
    On any error, returns the original units to avoid breaking the query pipeline.
    """
    if not units:
        return []

    global _rerank_semaphore
    if _rerank_semaphore is None:
        # Lazy-init inside event loop: asyncio.Semaphore must be created here.
        _rerank_semaphore = asyncio.Semaphore(25)

    try:
        reranker = _get_reranker()
        async with _rerank_semaphore:
            if hasattr(reranker, 'arerank'):
                return await reranker.arerank(query, units, top_k)
            else:
                return await asyncio.to_thread(reranker.rerank, query, units, top_k)
    except Exception as e:
        # Best-effort reranking: log and fall back to original units
        logger.warning("Reranking failed, using original units: %s", e)
        return units


def _filter_by_score(units: list[Any], min_score: Optional[float]) -> list[Any]:
    """Filter out units whose score is below min_score.

    No-op when min_score is None. Should be called after reranking so that
    the threshold is applied against final reranker scores.
    """
    if min_score is None:
        return units
    return [u for u in units if (u.score or 0.0) >= min_score]


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
        # Cache-hit is an O(1) dict lookup — call directly without a thread.
        # Only use asyncio.to_thread on first call (gRPC channel setup blocks ~2-5s).
        if collection_name in _vector_store_cache:
            vector_store = _get_vector_store(collection_name)
        else:
            vector_store = await asyncio.to_thread(_get_vector_store, collection_name)
        retriever = VectorRetriever(vector_store=vector_store)
        
        # Perform search
        units = await retriever.aretrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        
        # Rerank then filter by score threshold
        units = await _rerank_units(request.query, units, top_k=request.top_k)
        units = _filter_by_score(units, request.min_score)

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
                score=unit.score,
                tags=_extract_tags(unit),
            ))
        
        return ApiResponse(
            success=True,
            code=200,
            data=unit_responses
        )
        
    except Exception as e:
        traceback.print_exc()
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
        
        # Use cached retriever to avoid meilisearch.get_index() on every request
        api_top_k = request.top_k or 10
        if collection_name in _fulltext_retriever_cache:
            retriever = _get_fulltext_retriever(collection_name, api_top_k)
        else:
            retriever = await asyncio.to_thread(_get_fulltext_retriever, collection_name, api_top_k)
        
        # Perform full-text search
        import time as _t
        _t0 = _t.perf_counter()
        units = await retriever.aretrieve(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )
        logger.debug("[FT] meilisearch: %.2fs, hits=%d", _t.perf_counter()-_t0, len(units))
        
        # Rerank then filter by score threshold
        _t1 = _t.perf_counter()
        units = await _rerank_units(request.query, units, top_k=request.top_k)
        logger.debug("[FT] rerank: %.2fs", _t.perf_counter()-_t1)
        units = _filter_by_score(units, request.min_score)

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
                tags=_extract_tags(unit),
            ))
        
        return ApiResponse(
            success=True,
            code=200,
            data=unit_responses
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Fulltext search failed: {str(e)}")


@router.post("/{dataset_id}/query/fusion", response_model=ApiResponse[List[UnitResponse]])
async def query_fusion(dataset_id: str, request: QueryRequest):
    """Fusion search combining vector and fulltext search.

    Uses vector (Qdrant) and full-text (Meilisearch) retrievers as two
    independent recall channels via QueryFusionRetriever, then applies
    the unified reranker on top to produce the final ranking.

    The fulltext retriever is wrapped with QueryRewriteRetriever to convert
    natural language queries into BM25-friendly keywords before retrieval.
    """
    try:
        import time as _time
        # Get dataset info
        collection_name, engine = get_dataset_info(dataset_id)

        # Determine recall sizes
        api_top_k = request.top_k or 5
        recall_top_k = api_top_k * 2

        # Initialize vector store and fulltext retriever.
        # Cache-hit is an O(1) dict lookup — no thread needed.
        # Only use asyncio.to_thread on first call (blocking IO on cache miss).
        _t0 = _time.perf_counter()
        need_init = (
            collection_name not in _vector_store_cache
            or collection_name not in _fulltext_retriever_cache
        )
        if need_init:
            vector_store, base_ft_retriever = await asyncio.gather(
                asyncio.to_thread(_get_vector_store, collection_name),
                asyncio.to_thread(_get_fulltext_retriever, collection_name, recall_top_k),
            )
        else:
            vector_store = _get_vector_store(collection_name)
            base_ft_retriever = _get_fulltext_retriever(collection_name, recall_top_k)
        logger.debug("[fusion] init: %.2fs", _time.perf_counter()-_t0)

        # Vector retriever uses the original natural language query
        vector_retriever = VectorRetriever(vector_store=vector_store)

        # Fulltext retriever: pass the query directly to BM25.
        # Meilisearch handles natural language queries well without LLM rewrite.
        # If the caller provides an explicit fulltext_query override, honour it.
        if request.fulltext_query:
            fulltext_retriever = QueryRewriteRetriever(
                retriever=base_ft_retriever,
                rewrite_fn=lambda q: request.fulltext_query,
            )
        else:
            fulltext_retriever = base_ft_retriever

        # Fuse both recall channels concurrently
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=FusionMode.RECIPROCAL_RANK,
            top_k=recall_top_k,
        )
        _t1 = _time.perf_counter()
        units = await fusion_retriever.aretrieve(
            query=request.query,
            top_k=recall_top_k,
            filters=request.filters,
        )
        logger.debug("[fusion] retrieve: %.2fs", _time.perf_counter()-_t1)

        # Rerank then filter by score threshold
        _t2 = _time.perf_counter()
        units = await _rerank_units(request.query, units, top_k=api_top_k)
        logger.debug("[fusion] rerank: %.2fs", _time.perf_counter()-_t2)
        units = _filter_by_score(units, request.min_score)

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
                    tags=_extract_tags(unit),
                )
            )

        return ApiResponse(success=True, code=200, data=unit_responses)

    except Exception as e:
        traceback.print_exc()
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
        # Case 2: doc_id only - targeted fetch, O(1); only need the first LOD unit
        units = await vector_store.afetch({
            "doc_id": request.doc_id,
            "metadata.custom.mode": ProcessingMode.LOD
        }, limit=1)
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
        
        # Use cached embedder and vector store
        vector_store = await asyncio.to_thread(_get_vector_store, collection_name)
        from zag.retrievers.tree import SimpleRetriever
        retriever = SimpleRetriever(
            vector_store=vector_store,
            llm_uri=f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            api_key=config.LLM_API_KEY,
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
        
        # Use cached embedder and vector store
        vector_store = await asyncio.to_thread(_get_vector_store, collection_name)
        from zag.retrievers.tree import MCTSRetriever
        retriever = MCTSRetriever.from_preset(
            preset_name=request.preset,
            api_key=config.LLM_API_KEY,
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
        
        # Use cached embedder and vector store
        vector_store = await asyncio.to_thread(_get_vector_store, collection_name)
        from zag.retrievers.tree import SkeletonRetriever
        retriever = SkeletonRetriever(
            llm_uri=f"{config.LLM_PROVIDER}/{config.LLM_MODEL}",
            api_key=config.LLM_API_KEY,
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


