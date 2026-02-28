"""Dataset API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import Dict, List
import json
import threading

from cachetools import TTLCache
from cachetools.keys import hashkey

from ..database import get_connection, now, generate_id
from ..schemas import (
    DatasetCreate,
    DatasetUpdate,
    DatasetResponse,
    ApiResponse,
    MessageResponse,
    TaskResponse
)
from ..constants import TaskStatus

router = APIRouter(prefix="/datasets", tags=["datasets"])

# ---------------------------------------------------------------------------
# Catalog cache - per dataset_id, refreshed every CATALOG_TTL seconds
# ---------------------------------------------------------------------------
CATALOG_TTL = 600  # 10 minutes
_catalog_cache: TTLCache = TTLCache(maxsize=100, ttl=CATALOG_TTL)
_catalog_lock = threading.RLock()


def _build_catalog(dataset_id: str, collection_name: str) -> Dict[str, Dict[str, str]]:
    """
    Scroll the entire Qdrant collection and build a lender → {doc_id: file_name} map.

    Only fetches doc_id + metadata payload (no content, no vectors) to minimise
    bandwidth. Deduplicates by doc_id so each document appears exactly once.
    """
    from qdrant_client import QdrantClient
    from .. import config

    client = QdrantClient(
        host=config.VECTOR_STORE_HOST,
        port=config.VECTOR_STORE_PORT,
        grpc_port=config.VECTOR_STORE_GRPC_PORT,
        prefer_grpc=True,
        timeout=30,  # 30s per individual scroll call
    )

    catalog: Dict[str, Dict[str, str]] = {}
    seen_doc_ids: set = set()
    offset = None

    while True:
        results, next_offset = client.scroll(
            collection_name=collection_name,
            offset=offset,
            limit=1000,
            # Fetch only what we need - avoids transferring content/embedding fields
            with_payload=["doc_id", "metadata"],
            with_vectors=False,
        )

        for point in results:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")

            # Skip if no doc_id or already processed this document
            if not doc_id or doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            metadata = payload.get("metadata") or {}
            custom = metadata.get("custom") or {}
            document = metadata.get("document") or {}

            lender = custom.get("lender")
            file_name = document.get("file_name") or doc_id

            # Skip units without a known lender
            if not lender:
                continue

            if lender not in catalog:
                catalog[lender] = {}
            catalog[lender][doc_id] = file_name

        if not next_offset:
            break
        offset = next_offset

    return catalog


def _get_catalog(dataset_id: str) -> Dict[str, Dict[str, str]]:
    """Return catalog from cache, rebuilding if stale."""
    import concurrent.futures
    from .. import config

    # Resolve collection_name first so cache key is stable regardless of dataset_id
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        if config.DEFAULT_COLLECTION_NAME:
            collection_name = config.DEFAULT_COLLECTION_NAME
        else:
            raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        collection_name = row["name"]

    # Cache keyed on collection_name so warmup and API requests share the same entry
    cache_key = hashkey(collection_name)
    with _catalog_lock:
        if cache_key in _catalog_cache:
            print(f"[Catalog] Cache HIT for collection {collection_name}")
            return _catalog_cache[cache_key]

    print(f"[Catalog] Cache MISS for collection {collection_name}, building...")

    # Run build in a thread with an overall timeout to avoid hanging forever
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_build_catalog, dataset_id, collection_name)
        try:
            catalog = future.result(timeout=120)  # 2 minutes max
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise HTTPException(
                status_code=504,
                detail="Catalog build timed out. The collection may be too large. Try again later.",
            )

    with _catalog_lock:
        _catalog_cache[cache_key] = catalog

    return catalog

# Import cache invalidation function (will be available after query module loads)
_invalidate_cache_fn = None

def _get_cache_invalidator():
    """Lazy load cache clear function to avoid circular import."""
    global _invalidate_cache_fn
    if _invalidate_cache_fn is None:
        try:
            from .query import clear_dataset_cache
            _invalidate_cache_fn = clear_dataset_cache
        except ImportError:
            pass
    return _invalidate_cache_fn


@router.post("", response_model=ApiResponse[DatasetResponse])
def create_dataset(dataset: DatasetCreate):
    """Create a new dataset and corresponding vector store collection.
    
    Safe creation logic:
    1. If dataset exists in DB: verify vector collection exists, create if missing
    2. If dataset not in DB: write to DB, then verify/create vector collection
    """
    from zag.storages.vector import QdrantVectorStore
    from zag.embedders import Embedder
    from .. import config
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset with same name already exists
    cursor.execute("SELECT * FROM datasets WHERE name = ?", (dataset.name,))
    existing_row = cursor.fetchone()
    
    if existing_row:
        # Dataset exists in DB - ensure vector collection exists
        conn.close()
        _ensure_vector_collection(dataset.name, dataset.engine)
        
        config_data = json.loads(existing_row["config"]) if existing_row["config"] else None
        data = DatasetResponse(
            dataset_id=str(existing_row["id"]),
            collection_name=existing_row["name"],
            name=existing_row["name"],
            description=existing_row["description"],
            engine=existing_row["engine"],
            config=config_data,
            created_at=existing_row["created_at"],
            updated_at=existing_row["updated_at"]
        )
        return ApiResponse(success=True, code=200, message="Dataset already exists", data=data)
    
    # Create new dataset in DB
    timestamp = now()
    config_json = json.dumps(dataset.config) if dataset.config else None
    dataset_id = generate_id()
    
    try:
        cursor.execute(
            """
            INSERT INTO datasets (id, name, description, engine, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, dataset.name, dataset.description, dataset.engine, config_json, timestamp, timestamp)
        )
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")
    
    conn.close()
    
    # Ensure vector collection exists (create if not exists)
    try:
        _ensure_vector_collection(dataset.name, dataset.engine)
    except Exception as e:
        # Rollback database if vector store creation fails
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        conn.commit()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to create vector collection: {str(e)}")
    
    data = DatasetResponse(
        dataset_id=str(dataset_id),
        collection_name=dataset.name,
        name=dataset.name,
        description=dataset.description,
        engine=dataset.engine,
        config=dataset.config,
        created_at=timestamp,
        updated_at=timestamp
    )
    
    return ApiResponse(success=True, code=200, message="Dataset created successfully", data=data)


def _ensure_vector_collection(collection_name: str, engine: str):
    """Ensure vector collection exists, create if not exists.
    
    Args:
        collection_name: Collection name
        engine: Vector store engine (qdrant, etc.)
        
    Raises:
        HTTPException: If engine not supported or creation fails
    """
    from zag.storages.vector import QdrantVectorStore
    from zag.embedders import Embedder
    from .. import config
    
    if engine != "qdrant":
        raise HTTPException(status_code=501, detail=f"Engine '{engine}' not supported")
    
    embedder = Embedder(
        config.EMBEDDING_URI,
        api_key=config.OPENAI_API_KEY
    )
    
    # QdrantVectorStore.server will create collection if not exists via _ensure_collection
    QdrantVectorStore.server(
        host=config.VECTOR_STORE_HOST,
        port=config.VECTOR_STORE_PORT,
        grpc_port=config.VECTOR_STORE_GRPC_PORT,
        prefer_grpc=True,
        collection_name=collection_name,
        embedder=embedder
    )


@router.get("", response_model=ApiResponse[List[DatasetResponse]])
def list_datasets():
    """List all datasets."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM datasets ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        config = json.loads(row["config"]) if row["config"] else None
        results.append(DatasetResponse(
            dataset_id=str(row["id"]),
            collection_name=row["name"],
            name=row["name"],
            description=row["description"],
            engine=row["engine"],
            config=config,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/{dataset_id}", response_model=ApiResponse[DatasetResponse])
def get_dataset(dataset_id: str):
    """Get dataset by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    config = json.loads(row["config"]) if row["config"] else None
    
    data = DatasetResponse(
        dataset_id=str(row["id"]),
        collection_name=row["name"],
        name=row["name"],
        description=row["description"],
        engine=row["engine"],
        config=config,
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
    
    return ApiResponse(success=True, code=200, data=data)


@router.get("/{dataset_id}/tasks", response_model=ApiResponse[List[TaskResponse]])
def list_dataset_tasks(dataset_id: str):
    """Get all tasks for a specific dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    cursor.execute(
        "SELECT * FROM tasks WHERE dataset_id = ? ORDER BY created_at DESC",
        (dataset_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        error_message = json.loads(row["error_message"]) if row["error_message"] else None
        results.append(TaskResponse(
            task_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            doc_id=str(row["doc_id"]),
            mode=row["mode"] if "mode" in row.keys() else "classic",
            status=row["status"],
            progress=row["progress"],
            error_message=error_message,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/{dataset_id}/catalog", response_model=ApiResponse[Dict[str, Dict[str, str]]])
def get_dataset_catalog(dataset_id: str, refresh: bool = False):
    """Return a lender → {doc_id: file_name} catalog for the dataset.

    The result is cached for 10 minutes (CATALOG_TTL). On cache miss the collection
    is scrolled once and the result stored; subsequent requests within the TTL window
    return the cached value immediately.

    Response shape:
        {
            "JMAC Lending": {
                "abc123": "DSCR_Prime_v2.pdf",
                ...
            },
            "Angel Oak": {
                "def456": "Non-QM_Jumbo.pdf"
            }
        }
    """
    try:
        if refresh:
            # Resolve collection_name to invalidate the right cache entry
            from .. import config as _cfg
            _conn = get_connection()
            _cursor = _conn.cursor()
            _cursor.execute("SELECT name FROM datasets WHERE id = ?", (dataset_id,))
            _row = _cursor.fetchone()
            _conn.close()
            _coll = (_row["name"] if _row else None) or _cfg.DEFAULT_COLLECTION_NAME
            if _coll:
                with _catalog_lock:
                    _catalog_cache.pop(hashkey(_coll), None)
        catalog = _get_catalog(dataset_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build catalog: {str(e)}")

    return ApiResponse(success=True, code=200, data=catalog)


@router.patch("/{dataset_id}", response_model=ApiResponse[DatasetResponse])
def update_dataset(dataset_id: str, dataset: DatasetUpdate):
    """Update dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Build update query
    updates = []
    params = []
    
    if dataset.name is not None:
        updates.append("name = ?")
        params.append(dataset.name)
    
    if dataset.description is not None:
        updates.append("description = ?")
        params.append(dataset.description)
    
    if dataset.engine is not None:
        updates.append("engine = ?")
        params.append(dataset.engine)
    
    if dataset.config is not None:
        updates.append("config = ?")
        params.append(json.dumps(dataset.config))
    
    if not updates:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")
    
    timestamp = now()
    updates.append("updated_at = ?")
    params.append(timestamp)
    params.append(dataset_id)
    
    cursor.execute(
        f"UPDATE datasets SET {', '.join(updates)} WHERE id = ?",
        params
    )
    conn.commit()
    
    # Invalidate cache
    invalidate_fn = _get_cache_invalidator()
    if invalidate_fn:
        invalidate_fn(dataset_id)
    
    # Fetch updated dataset
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()
    
    config = json.loads(row["config"]) if row["config"] else None
    
    data = DatasetResponse(
        dataset_id=str(row["id"]),
        collection_name=row["name"],
        name=row["name"],
        description=row["description"],
        engine=row["engine"],
        config=config,
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
    
    return ApiResponse(success=True, code=200, message="Dataset updated successfully", data=data)


@router.delete("/{dataset_id}", response_model=ApiResponse[MessageResponse])
def delete_dataset(dataset_id: str):
    """Delete dataset and corresponding vector store collection."""
    from zag.storages.vector import QdrantVectorStore
    from zag.embedders import Embedder
    from .. import config
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists and get collection name
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    collection_name = row["name"]
    
    # Delete from database
    cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    conn.commit()
    conn.close()
    
    # Delete vector store collection
    try:
        embedder = Embedder(
            config.EMBEDDING_URI,
            api_key=config.OPENAI_API_KEY
        )
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            grpc_port=config.VECTOR_STORE_GRPC_PORT,
            prefer_grpc=True,
            collection_name=collection_name,
            embedder=embedder
        )
        vector_store.delete_collection()
    except Exception as e:
        # Log but don't fail the API call (collection might not exist)
        print(f"Warning: Failed to delete vector collection {collection_name}: {e}")
    
    # Invalidate cache
    invalidate_fn = _get_cache_invalidator()
    if invalidate_fn:
        invalidate_fn(dataset_id)
    
    return ApiResponse(
        success=True,
        code=200,
        message="Dataset deleted successfully",
        data=MessageResponse(message="Dataset deleted successfully")
    )
