"""Unit API endpoints."""
import uuid
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from ..schemas import (
    UnitResponse, UnitRawResponse, UnitCreateRequest,
    ApiResponse, MessageResponse
)
from .. import config
from .query import get_dataset_info
from zag.storages.vector import QdrantVectorStore

router = APIRouter(prefix="/datasets", tags=["units"])


async def _list_units_for_doc(dataset_id: str, doc_id: str) -> list[dict]:
    """Fetch all units for a doc as plain dicts (used by export endpoints)."""
    vector_store = _get_vector_store(dataset_id)
    units = await vector_store.afetch({"doc_id": doc_id})
    result = []
    for unit in units:
        u = UnitResponse.from_unit(unit)
        result.append(u.model_dump())
    return result


def _get_vector_store(dataset_id: str, with_embedder: bool = False):
    """Get vector store for dataset.

    Args:
        dataset_id: Dataset ID
        with_embedder: If True, initialise embedder (required for re-embedding).

    Returns:
        Vector store instance

    Raises:
        HTTPException: If engine type is not supported
    """
    collection_name, engine = get_dataset_info(dataset_id)

    if engine == "qdrant":
        embedder = None
        if with_embedder:
            from zag.embedders import Embedder
            embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        return QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            prefer_grpc=False,
            collection_name=collection_name,
            embedder=embedder,
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
    """Delete a unit from both vector store and full-text search index.

    Args:
        dataset_id: Dataset ID
        unit_id: Unit ID

    Returns:
        Success message
    """
    collection_name, _ = get_dataset_info(dataset_id)
    vector_store = _get_vector_store(dataset_id)

    # Delete from Qdrant
    await vector_store.adelete([unit_id])

    # Delete from Meilisearch (best-effort: don't fail if index is missing)
    try:
        from zag.indexers import FullTextIndexer
        fulltext_indexer = FullTextIndexer(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            auto_create_index=False,
        )
        await fulltext_indexer.adelete(unit_id)
    except Exception as e:
        print(f"Warning: failed to delete unit {unit_id} from Meilisearch: {e}")

    return ApiResponse(
        success=True,
        code=200,
        data=MessageResponse(message=f"Unit {unit_id} deleted")
    )


@router.post("/{dataset_id}/units", response_model=ApiResponse[UnitResponse])
async def create_unit(dataset_id: str, body: UnitCreateRequest):
    """Manually create a TextUnit and index it in both vector store and full-text search.

    Intended for data-ops use: after reviewing an exported document, operators
    can add a corrected or supplementary unit alongside deleting the problematic one.

    All fields are required to ensure the unit is complete and correctly embedded.

    Args:
        dataset_id: Dataset ID
        body: Unit creation payload

    Returns:
        Newly created unit
    """
    from zag.schemas import TextUnit, UnitMetadata
    from zag.indexers import FullTextIndexer

    # Build unit with a fresh UUID
    unit_id = str(uuid.uuid4())
    metadata = UnitMetadata(custom=body.metadata_custom)
    unit = TextUnit(
        unit_id=unit_id,
        doc_id=body.doc_id,
        content=body.content,
        embedding_content=body.embedding_content,
        metadata=metadata,
        prev_unit_id=body.prev_unit_id,
        next_unit_id=body.next_unit_id,
    )

    # Embed and write to Qdrant
    vector_store = _get_vector_store(dataset_id, with_embedder=True)
    await vector_store.aadd([unit])

    # Index in Meilisearch (best-effort)
    collection_name, _ = get_dataset_info(dataset_id)
    try:
        fulltext_indexer = FullTextIndexer(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            auto_create_index=False,
        )
        await fulltext_indexer.aadd([unit])
    except Exception as e:
        print(f"Warning: failed to index unit {unit_id} in Meilisearch: {e}")

    return ApiResponse(
        success=True,
        code=201,
        data=UnitResponse.from_unit(unit)
    )
