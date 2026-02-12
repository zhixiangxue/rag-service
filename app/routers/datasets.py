"""Dataset API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List
import json

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
