"""Pydantic schemas for API request/response."""
from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, Any, List, TypeVar, Generic
from datetime import datetime
from enum import Enum


# ============ Processing Mode ============

class ProcessingMode(str, Enum):
    """Document processing mode."""
    CLASSIC = "classic"  # Chunk-based RAG
    LOD = "lod"          # Level-of-Detail indexing


class ReaderType(str, Enum):
    """Document reader type for parsing."""
    DEFAULT = "mineru"
    MINERU = "mineru"    # MinerU reader (GPU-accelerated, default)
    CLAUDE = "claude"    # Claude Vision reader (API-based, high quality)


# ============ Common Response Wrapper ============

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    """Unified API response wrapper."""
    success: bool
    code: int = 200
    message: Optional[str] = None
    data: Optional[T] = None


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str


# ============ Dataset ============

class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    engine: str = "qdrant"  # Vector store engine: qdrant, chroma, milvus, etc.
    config: Optional[Dict[str, Any]] = None


class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    engine: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    dataset_id: str
    collection_name: str
    name: str
    description: Optional[str]
    engine: str
    config: Optional[Dict[str, Any]]
    created_at: str
    updated_at: str


# ============ Document ============

class DocumentResponse(BaseModel):
    """Document response."""
    doc_id: str
    dataset_id: str
    file_name: str
    file_path: str
    file_url: Optional[str] = None  # HTTP URL for distributed worker access
    workspace_dir: str
    file_size: int
    file_type: str
    file_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # Document metadata from upload
    status: str  # PROCESSING, COMPLETED, FAILED, DISABLED
    task_id: Optional[str] = None
    unit_count: Optional[int] = None
    created_at: str
    updated_at: str


# ============ Task ============

class TaskResponse(BaseModel):
    """Task response."""
    task_id: str
    dataset_id: str
    doc_id: str
    mode: str = "classic"
    reader: str = ReaderType.DEFAULT  # Reader used: mineru | claude
    status: str  # PENDING, PROCESSING, COMPLETED, FAILED
    progress: int  # 0-100
    metadata: Optional[Dict[str, Any]] = None  # Document metadata from upload
    error_message: Optional[Dict[str, Any]] = None
    worker: Optional[str] = None  # hostname (ip) of the worker that processed this task
    created_at: str
    updated_at: str


class TaskStatusUpdate(BaseModel):
    """Task status update request."""
    status: Optional[str] = None
    progress: Optional[int] = None
    error_message: Optional[Dict[str, Any]] = None
    unit_count: Optional[int] = None
    worker: Optional[str] = None  # set by worker when starting PROCESSING

# ============ Unit ============

def _extract_tags(unit: Any) -> List[str]:
    """Extract tags list from unit.metadata.custom.tags, return [] if absent."""
    try:
        custom = unit.metadata.custom if unit.metadata else {}
        if isinstance(custom, dict):
            tags = custom.get("tags", [])
        else:
            tags = getattr(custom, "tags", []) or []
        return tags if isinstance(tags, list) else []
    except Exception:
        return []

class UnitResponse(BaseModel):
    """Unit response model (lightweight) - excludes large fields like views."""
    unit_id: str
    unit_type: str = "text"
    content: Any
    embedding_content: Optional[str] = None  # Text used for embedding (may differ from content)
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    relations: Dict[str, List[str]] = Field(default_factory=dict)
    has_views: bool = False  # Indicates if raw endpoint has multi-resolution views
    tree: Optional[Dict[str, Any]] = None  # HIGH view DocTree, only populated for LOD units
    score: Optional[float] = None  # Relevance score from query results
    tags: List[str] = Field(default_factory=list)  # Document tags from metadata.custom.tags

    @classmethod
    def from_unit(cls, unit: "BaseUnit") -> "UnitResponse":
        """Create lightweight UnitResponse from zag BaseUnit."""
        return cls(
            unit_id=unit.unit_id,
            unit_type=str(unit.unit_type) if unit.unit_type else "text",
            content=unit.content,
            embedding_content=getattr(unit, 'embedding_content', None),
            metadata=unit.metadata.model_dump() if unit.metadata else {},
            doc_id=unit.doc_id,
            prev_unit_id=unit.prev_unit_id,
            next_unit_id=unit.next_unit_id,
            relations=unit.relations if unit.relations else {},
            has_views=bool(unit.views) if hasattr(unit, 'views') else False,
            tags=_extract_tags(unit),
        )


class UnitRawResponse(BaseModel):
    """Unit raw response model (full) - includes all fields including views/content."""
    unit_id: str
    unit_type: str = "text"
    content: Any
    embedding_content: Optional[str] = None  # Text used for embedding (may differ from content)
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    relations: Dict[str, List[str]] = Field(default_factory=dict)
    views: Optional[List[Dict[str, Any]]] = None
    tags: List[str] = Field(default_factory=list)  # Document tags from metadata.custom.tags
    
    @classmethod
    def from_unit(cls, unit: "BaseUnit") -> "UnitRawResponse":
        """Create full UnitRawResponse from zag BaseUnit."""
        return cls(
            unit_id=unit.unit_id,
            unit_type=str(unit.unit_type) if unit.unit_type else "text",
            content=unit.content,
            embedding_content=getattr(unit, 'embedding_content', None),
            metadata=unit.metadata.model_dump() if unit.metadata else {},
            doc_id=unit.doc_id,
            prev_unit_id=unit.prev_unit_id,
            next_unit_id=unit.next_unit_id,
            relations=unit.relations if unit.relations else {},
            views=[v.model_dump() for v in unit.views] if unit.views else None,
            tags=_extract_tags(unit),
        )


class UnitBatchUpdate(BaseModel):
    unit_id: str
    metadata: Dict[str, Any]


class UnitCreateRequest(BaseModel):
    """Request body for manually creating a TextUnit.

    All fields are required to ensure the unit is complete and correctly
    embedded from the start. content and embedding_content are always paired:
    content is what LLM sees, embedding_content is what drives retrieval.
    """
    doc_id: str
    content: str
    embedding_content: str
    metadata_custom: Dict[str, Any]  # At minimum tags and mode are expected
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None


# ============ Query ============

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=200)
    filters: Optional[Dict[str, Any]] = None
    fulltext_query: Optional[str] = None  # Caller-provided keyword query for BM25; auto-rewritten if absent
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)  # Filter out units below this reranker score


class QueryResponse(BaseModel):
    results: List[dict]


# ============ Tree Query ============

class TreeQueryRequest(BaseModel):
    """Tree query request for tree-based retrieval."""
    query: str
    unit_id: Optional[str] = None
    doc_id: Optional[str] = None
    # SimpleRetriever params
    max_depth: int = Field(default=5, ge=1, le=10)
    # MCTSRetriever params
    preset: str = Field(default="balanced", pattern="^(fast|balanced|accurate|explore)$")
    # SkeletonRetriever params: retrieval mode (fast=summary, accurate=fulltext)
    mode: str = Field(default="fast", pattern="^(fast|accurate)$")

    @model_validator(mode='after')
    def validate_identifiers(self) -> 'TreeQueryRequest':
        """Require at least one of unit_id or doc_id."""
        if not self.unit_id and not self.doc_id:
            raise ValueError("Either unit_id or doc_id must be provided")
        return self


# ============ Health ============

class HealthResponse(BaseModel):
    status: str
    dependencies: Dict[str, str]


class DatasetStats(BaseModel):
    dataset_id: str
    document_count: int
    unit_count: int
    last_ingest_time: Optional[str] = None


# ============ Cache Upload ============

class CacheUploadResponse(BaseModel):
    """Response for cache upload."""
    doc_id: str
    cache_path: str
    size_bytes: int
    message: str


# ============ Page Location ============

class LocatePageItem(BaseModel):
    """Single item in page location request."""
    request_id: str = Field(..., description="Caller-provided ID, returned as-is for correlation")
    doc_id: str = Field(..., description="Document ID")
    text_start: str = Field(..., description="Beginning of the text snippet")
    text_end: Optional[str] = Field(None, description="End of the text snippet (optional)")


class LocatePageRequest(BaseModel):
    """Batch page location request."""
    items: List[LocatePageItem] = Field(..., min_length=1)


class LocatePageResult(BaseModel):
    """Result for a single locate request."""
    request_id: str
    doc_id: str
    page_numbers: List[int] = []
    found: bool
    error: Optional[str] = None


class LocatePageResponse(BaseModel):
    """Batch page location response."""
    results: List[LocatePageResult]
