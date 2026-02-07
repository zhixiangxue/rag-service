"""Pydantic schemas for API request/response."""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, TypeVar, Generic
from datetime import datetime
from enum import Enum


# ============ Processing Mode ============

class ProcessingMode(str, Enum):
    """Document processing mode."""
    CLASSIC = "classic"  # Chunk-based RAG
    LOD = "lod"          # Level-of-Detail indexing


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
    workspace_dir: str
    file_size: int
    file_type: str
    file_hash: Optional[str] = None
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
    status: str  # PENDING, PROCESSING, COMPLETED, FAILED
    progress: int  # 0-100
    metadata: Optional[Dict[str, Any]] = None  # Document metadata from upload
    error_message: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class TaskStatusUpdate(BaseModel):
    """Task status update request."""
    status: Optional[str] = None
    progress: Optional[int] = None
    error_message: Optional[Dict[str, Any]] = None
    unit_count: Optional[int] = None

# ============ Unit ============

class UnitResponse(BaseModel):
    """Unit response model (lightweight) - excludes large fields like views."""
    unit_id: str
    unit_type: str = "text"
    content: Any
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    relations: Dict[str, List[str]] = Field(default_factory=dict)
    has_views: bool = False  # Indicates if raw endpoint has multi-resolution views
    score: Optional[float] = None  # Relevance score from query results

    @classmethod
    def from_unit(cls, unit: "BaseUnit") -> "UnitResponse":
        """Create lightweight UnitResponse from zag BaseUnit."""
        return cls(
            unit_id=unit.unit_id,
            unit_type=str(unit.unit_type) if unit.unit_type else "text",
            content=unit.content,
            metadata=unit.metadata.model_dump() if unit.metadata else {},
            doc_id=unit.doc_id,
            prev_unit_id=unit.prev_unit_id,
            next_unit_id=unit.next_unit_id,
            relations=unit.relations if unit.relations else {},
            has_views=bool(unit.views) if hasattr(unit, 'views') else False
        )


class UnitRawResponse(BaseModel):
    """Unit raw response model (full) - includes all fields including views/content."""
    unit_id: str
    unit_type: str = "text"
    content: Any
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None
    prev_unit_id: Optional[str] = None
    next_unit_id: Optional[str] = None
    relations: Dict[str, List[str]] = Field(default_factory=dict)
    views: Optional[List[Dict[str, Any]]] = None
    
    @classmethod
    def from_unit(cls, unit: "BaseUnit") -> "UnitRawResponse":
        """Create full UnitRawResponse from zag BaseUnit."""
        return cls(
            unit_id=unit.unit_id,
            unit_type=str(unit.unit_type) if unit.unit_type else "text",
            content=unit.content,
            metadata=unit.metadata.model_dump() if unit.metadata else {},
            doc_id=unit.doc_id,
            prev_unit_id=unit.prev_unit_id,
            next_unit_id=unit.next_unit_id,
            relations=unit.relations if unit.relations else {},
            views=[v.model_dump() for v in unit.views] if unit.views else None
        )


class UnitUpdate(BaseModel):
    metadata: Dict[str, Any]


class UnitBatchUpdate(BaseModel):
    unit_id: str
    metadata: Dict[str, Any]


# ============ Query ============

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    results: List[dict]


# ============ Tree Query ============

class TreeQueryRequest(BaseModel):
    """Tree query request for tree-based retrieval."""
    query: str
    unit_id: str
    # SimpleRetriever params
    max_depth: int = Field(default=5, ge=1, le=10)
    # MCTSRetriever params
    preset: str = Field(default="balanced", pattern="^(fast|balanced|accurate|explore)$")
    # SkeletonRetriever params: retrieval mode (fast=summary, accurate=fulltext)
    mode: str = Field(default="fast", pattern="^(fast|accurate)$")


# ============ Health ============

class HealthResponse(BaseModel):
    status: str
    dependencies: Dict[str, str]


class DatasetStats(BaseModel):
    dataset_id: str
    document_count: int
    unit_count: int
    last_ingest_time: Optional[str] = None
