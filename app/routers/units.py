"""Unit API endpoints."""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..schemas import UnitResponse, UnitUpdate, UnitBatchUpdate, ApiResponse, MessageResponse
from .query import get_dataset_info

router = APIRouter(prefix="/datasets", tags=["units"])


@router.get("/{dataset_id}/units/{unit_id}", response_model=ApiResponse[UnitResponse])
async def get_unit(dataset_id: str, unit_id: str):
    """Get a single unit by ID.
    
    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID
        
    Returns:
        Unit information
    """
    # TODO: Implement using zag's unit storage abstraction
    # Should use zag's unified interface to retrieve unit from vector store
    # This allows zag to handle different vector database engines (qdrant/chroma/milvus)
    raise HTTPException(status_code=501, detail="Get unit not implemented yet")


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
    # TODO: Implement using zag's unit storage abstraction
    # Should use zag's unified interface to list/filter units
    # Support pagination and filtering by doc_id
    raise HTTPException(status_code=501, detail="List units not implemented yet")


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
    # TODO: Implement using zag's unit storage abstraction
    # Should use zag's unified interface to update unit metadata
    # Only metadata should be updatable, not content/embedding
    raise HTTPException(status_code=501, detail="Update unit not implemented yet")


@router.patch("/{dataset_id}/units:batch_update", response_model=ApiResponse[MessageResponse])
async def batch_update_units(dataset_id: str, updates: List[UnitBatchUpdate]):
    """Batch update unit metadata.
    
    Args:
        dataset_id: Dataset ID
        updates: List of unit updates
        
    Returns:
        Success message
    """
    # TODO: Implement using zag's unit storage abstraction
    # Should use zag's unified interface to batch update units
    # This is a high-frequency operation, needs optimization
    raise HTTPException(status_code=501, detail="Batch update units not implemented yet")


@router.delete("/{dataset_id}/units/{unit_id}", response_model=ApiResponse[MessageResponse])
async def delete_unit(dataset_id: str, unit_id: str):
    """Delete a unit.
    
    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID
        
    Returns:
        Success message
    """
    # TODO: Implement using zag's unit storage abstraction
    # Should use zag's unified interface to delete unit from vector store
    # Physical deletion, no soft delete
    raise HTTPException(status_code=501, detail="Delete unit not implemented yet")
