"""Dataset API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List
import json

from ..database import get_connection, now
from ..schemas import DatasetCreate, DatasetUpdate, DatasetResponse, ApiResponse, MessageResponse

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
    """Create a new dataset or return existing one if name already exists."""
    # TODO: Call zag to create vector store collection
    # Should use zag's abstraction layer instead of directly calling Qdrant
    # This allows zag to handle different vector database engines
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset with same name already exists
    cursor.execute("SELECT * FROM datasets WHERE name = ?", (dataset.name,))
    existing_row = cursor.fetchone()
    
    if existing_row:
        # Return existing dataset
        conn.close()
        config = json.loads(existing_row["config"]) if existing_row["config"] else None
        data = DatasetResponse(
            dataset_id=str(existing_row["id"]),
            collection_name=existing_row["name"],
            name=existing_row["name"],
            description=existing_row["description"],
            engine=existing_row["engine"],
            config=config,
            created_at=existing_row["created_at"],
            updated_at=existing_row["updated_at"]
        )
        return ApiResponse(success=True, code=200, message="Dataset already exists", data=data)
    
    # Create new dataset
    timestamp = now()
    config_json = json.dumps(dataset.config) if dataset.config else None
    
    try:
        cursor.execute(
            """
            INSERT INTO datasets (name, description, engine, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset.name, dataset.description, dataset.engine, config_json, timestamp, timestamp)
        )
        conn.commit()
        dataset_id = cursor.lastrowid
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to create dataset: {str(e)}")
    
    conn.close()
    
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
    """Delete dataset."""
    # TODO: Call zag to delete vector store collection
    # Should use zag's abstraction layer instead of directly calling Qdrant
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete dataset
    cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
    conn.commit()
    conn.close()
    
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
