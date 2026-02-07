"""Unit API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..schemas import UnitResponse, UnitRawResponse, UnitUpdate, ApiResponse, MessageResponse
from .. import config
from .query import get_dataset_info
from zag.storages.vector import QdrantVectorStore

router = APIRouter(prefix="/datasets", tags=["units"])


def _get_vector_store(dataset_id: str):
    """Get vector store for dataset.
    
    Args:
        dataset_id: Dataset ID
        
    Returns:
        Vector store instance (QdrantVectorStore or other)
        
    Raises:
        HTTPException: If engine type is not supported
    """
    collection_name, engine = get_dataset_info(dataset_id)
    
    # Initialize vector store based on engine type
    if engine == "qdrant":
        return QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=None,  # Not needed for get/fetch/update/remove
            timeout=60
        )
    else:
        raise HTTPException(
            status_code=501, 
            detail=f"Vector store engine '{engine}' not supported yet"
        )


@router.get("/{dataset_id}/units/{unit_id}", response_model=ApiResponse[UnitResponse])
async def get_unit(dataset_id: str, unit_id: str):
    """Get a single unit by ID.
    
    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID
        
    Returns:
        Unit information
    """
    vector_store = _get_vector_store(dataset_id)
    
    units = await vector_store.aget([unit_id])
    if not units:
        raise HTTPException(status_code=404, detail="Unit not found")
    
    unit = units[0]
    return ApiResponse(
        success=True,
        code=200,
        data=UnitResponse.from_unit(unit)
    )


@router.get("/{dataset_id}/units", response_model=ApiResponse[List[UnitResponse]])
async def list_units(
    dataset_id: str,
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    doc_id: Optional[str] = None
):
    """List units in a dataset.
    
    Args:
        dataset_id: Dataset ID
        page: Page number (starting from 1)
        size: Page size
        doc_id: Optional filter by document ID
        
    Returns:
        List of units
    """
    vector_store = _get_vector_store(dataset_id)
    
    # Build filters
    filters = None
    if doc_id:
        filters = {"doc_id": doc_id}
    
    # Fetch all matching units
    units = await vector_store.afetch(filters)
    
    # Manual pagination
    total = len(units)
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    paginated_units = units[start_idx:end_idx]
    
    return ApiResponse(
        success=True,
        code=200,
        data=[UnitResponse.from_unit(unit) for unit in paginated_units]
    )


@router.patch("/{dataset_id}/units/{unit_id}", response_model=ApiResponse[UnitResponse])
async def update_unit(dataset_id: str, unit_id: str, update: UnitUpdate):
    """Update unit metadata.
    
    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID
        update: Metadata to update
        
    Returns:
        Updated unit information
    """
    vector_store = _get_vector_store(dataset_id)
    
    # Get existing unit
    units = await vector_store.aget([unit_id])
    if not units:
        raise HTTPException(status_code=404, detail="Unit not found")
    
    unit = units[0]
    
    # Update metadata
    if unit.metadata:
        unit.metadata.custom.update(update.metadata)
    else:
        from zag.schemas import UnitMetadata
        unit.metadata = UnitMetadata(custom=update.metadata)
    
    # Save back
    await vector_store.aupdate(unit)
    
    return ApiResponse(
        success=True,
        code=200,
        data=UnitResponse.from_unit(unit)
    )


@router.get("/{dataset_id}/units/{unit_id}/raw", response_model=ApiResponse[UnitRawResponse])
async def get_unit_raw(dataset_id: str, unit_id: str):
    """Get a single unit by ID with full raw data (including views/content).
    
    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID
        
    Returns:
        Complete unit information with all fields
    """
    vector_store = _get_vector_store(dataset_id)
    
    units = await vector_store.aget([unit_id])
    if not units:
        raise HTTPException(status_code=404, detail="Unit not found")
    
    unit = units[0]
    return ApiResponse(
        success=True,
        code=200,
        data=UnitRawResponse.from_unit(unit)
    )


@router.delete("/{dataset_id}/units/{unit_id}", response_model=ApiResponse[MessageResponse])
async def delete_unit(dataset_id: str, unit_id: str):
    """Delete a unit.
    
    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID
        
    Returns:
        Success message
    """
    vector_store = _get_vector_store(dataset_id)
    
    # Delete by unit_id
    await vector_store.adelete([unit_id])
    
    return ApiResponse(
        success=True,
        code=200,
        data=MessageResponse(
            message=f"Unit {unit_id} deleted"
        )
    )
