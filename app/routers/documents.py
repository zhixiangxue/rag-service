"""Document API endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import json
import sqlite3
import tempfile
from pathlib import Path

from zag.utils.hash import calculate_file_hash
from ..utils.s3 import get_s3_object_info, download_file_from_s3_async
from ..database import get_connection, now, generate_id
from ..schemas import (
    DocumentResponse,
    ApiResponse,
    MessageResponse,
    TaskResponse,
    ProcessingMode,
    _extract_tags,
    LocatePageRequest,
    LocatePageResult,
    LocatePageResponse,
    CacheUploadResponse,
)
from ..storage import get_storage
from ..constants import TaskStatus, DocumentStatus
from .. import config

router = APIRouter(prefix="/datasets/{dataset_id}/documents", tags=["documents"])


def _validate_metadata(metadata: Optional[str]) -> dict:
    """
    Validate metadata JSON string and guideline field.
    
    Args:
        metadata: JSON string or None
        
    Returns:
        Parsed metadata dict
        
    Raises:
        ValueError: If metadata is invalid or guideline value is not allowed
    """
    if metadata is None:
        raise ValueError(
            "metadata is required. Please provide a JSON object. "
            "Recommended fields: "
            '{"lender": "xxx", "guideline": "FannieMae|FreddieMac|VA|USDA|FHA", '
            '"overlays": ["xxx"], "tags": ["xxx"]}'
        )
    try:
        metadata_dict = json.loads(metadata)
        if not isinstance(metadata_dict, dict):
            raise ValueError("Metadata must be a JSON object")
    except json.JSONDecodeError:
        raise ValueError("Metadata must be valid JSON")

    # Validate guideline and overlays fields
    VALID_GUIDELINES = {"FannieMae", "FreddieMac", "VA", "USDA", "FHA"}

    if "guideline" in metadata_dict:
        guideline_val = metadata_dict["guideline"]
        if guideline_val not in VALID_GUIDELINES:
            raise ValueError(
                f'Invalid guideline value: "{guideline_val}". '
                f"Must be one of: {sorted(VALID_GUIDELINES)}"
            )

    if "overlays" in metadata_dict:
        overlays = metadata_dict["overlays"]
        if not isinstance(overlays, list):
            raise ValueError('overlays must be an array')
        for item in overlays:
            if item not in VALID_GUIDELINES:
                raise ValueError(
                    f'Invalid overlays value: "{item}". '
                    f"Must be one of: {sorted(VALID_GUIDELINES)}"
                )

    return metadata_dict


def _create_document_record(
    dataset_id: str,
    file_path: str,
    filename: str,
    metadata: Optional[str] = None
) -> dict:
    """
    Create document record after a local file is saved.
    Handles dataset validation, hash calculation, duplicate check, and database insert.

    Args:
        dataset_id: Dataset ID
        file_path: Local path to the saved file
        filename: Original filename
        metadata: Optional JSON metadata string

    Returns:
        dict with doc_id, file_name, file_path, file_hash, and status info

    Raises:
        ValueError: If metadata is invalid
        HTTPException: If dataset not found (404)
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Check if dataset exists
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Calculate file hash using zag's utility on saved file
        file_hash = calculate_file_hash(file_path)

        # Check for duplicate: same dataset + same file_hash
        cursor.execute(
            "SELECT id, file_name FROM documents WHERE dataset_id = ? AND file_hash = ?",
            (dataset_id, file_hash)
        )
        existing_doc = cursor.fetchone()

        if existing_doc:
            # File already uploaded, check if old file still exists
            cursor.execute(
                "SELECT file_path FROM documents WHERE id = ?",
                (existing_doc['id'],)
            )
            old_doc = cursor.fetchone()

            if old_doc and os.path.exists(old_doc['file_path']):
                # Old file exists, delete duplicate and reuse existing doc_id
                os.remove(file_path)
                return {
                    "dataset_id": str(dataset_id),
                    "doc_id": str(existing_doc['id']),
                    "file_name": existing_doc['file_name'],
                    "file_hash": file_hash,
                    "is_duplicate": True
                }
            else:
                # Old file doesn't exist, delete old record and continue with new upload
                cursor.execute("DELETE FROM documents WHERE id = ?", (existing_doc['id'],))
                conn.commit()

        # Extract workspace directory (parent of file)
        workspace_dir = os.path.dirname(file_path)

        file_size = os.path.getsize(file_path)
        file_type = filename.split(".")[-1] if "." in filename else "unknown"

        # Parse and validate metadata
        metadata_dict = _validate_metadata(metadata)
        metadata_json = json.dumps(metadata_dict)

        timestamp = now()
        # Use file_hash as doc_id to ensure same content gets same ID
        doc_id = file_hash

        # Create Document record with file_hash and metadata
        # Use INSERT OR IGNORE to handle race conditions
        try:
            cursor.execute(
                """
                INSERT INTO documents (id, dataset_id, file_name, file_path, workspace_dir, file_size, file_type, file_hash, metadata, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, dataset_id, filename, file_path, workspace_dir, file_size, file_type, file_hash, metadata_json, DocumentStatus.PROCESSING, timestamp, timestamp)
            )
            conn.commit()

            return {
                "dataset_id": str(dataset_id),
                "doc_id": str(doc_id),
                "file_name": filename,
                "file_path": file_path,
                "file_hash": file_hash,
                "is_duplicate": False
            }
        except sqlite3.IntegrityError:
            # UNIQUE constraint violation: duplicate file_hash
            # This means another request inserted the same file first
            conn.rollback()

            # Delete the newly uploaded file (duplicate)
            os.remove(file_path)

            # Re-query to get the existing document
            cursor.execute(
                "SELECT id, file_name FROM documents WHERE dataset_id = ? AND file_hash = ?",
                (dataset_id, file_hash)
            )
            existing_doc = cursor.fetchone()

            return {
                "dataset_id": str(dataset_id),
                "doc_id": str(existing_doc['id']),
                "file_name": existing_doc['file_name'],
                "file_hash": file_hash,
                "is_duplicate": True
            }
    finally:
        conn.close()


@router.post("", response_model=ApiResponse[dict])
async def upload_file(
    dataset_id: str,
    file: UploadFile = File(...),
    metadata: str = Form(...)
):
    """Upload file to dataset."""
    # Save file using storage abstraction
    storage = get_storage()
    file_path = storage.save(file.file, file.filename, dataset_id)
    
    # Create document record (handles dataset validation, hash, duplicate check, insert)
    try:
        result = _create_document_record(dataset_id, file_path, file.filename, metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    message = "File already exists, reusing existing document" if result["is_duplicate"] else "File uploaded successfully"
    return ApiResponse(success=True, code=200, message=message, data=result)


@router.post("/from-s3", response_model=ApiResponse[dict])
async def upload_from_s3(
    dataset_id: str,
    s3_url: str = Body(..., embed=True),
    metadata: str = Body(..., embed=True),
):
    """
    Download a file from S3 and register it in the dataset.

    Identical flow to local upload: download -> compute content hash -> insert record.
    doc_id = content hash, same dedup logic applies.
    S3 existence is verified upfront via a HEAD request (400 if not found).
    """
    if not s3_url.startswith("s3://"):
        raise HTTPException(status_code=400, detail="s3_url must start with 's3://'")

    filename = Path(s3_url).name
    if not filename:
        raise HTTPException(status_code=400, detail="Cannot determine filename from S3 URL")

    # Verify the file exists on S3 before downloading (cheap HEAD request)
    try:
        get_s3_object_info(
            s3_url,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_KEY,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"File not found in S3: {s3_url}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify S3 file: {str(e)}")

    # Download to a temp location
    storage = get_storage()
    temp_dir = Path(tempfile.gettempdir()) / f"s3_{dataset_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / filename

    try:
        await download_file_from_s3_async(
            s3_url,
            temp_file,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_KEY,
        )
    except Exception as e:
        temp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to download file from S3: {str(e)}")

    # Move to permanent storage
    try:
        with open(temp_file, "rb") as f:
            file_path = storage.save(f, filename, dataset_id)
        temp_file.unlink(missing_ok=True)
    except Exception as e:
        temp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create document record — identical to local upload (doc_id = content hash)
    try:
        result = _create_document_record(dataset_id, file_path, filename, metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Document record creation failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create document record: {str(e)}")

    message = (
        "File already exists, reusing existing document"
        if result["is_duplicate"]
        else "File downloaded from S3 and registered successfully"
    )
    return ApiResponse(success=True, code=200, message=message, data=result)


@router.post("/{doc_id}/tasks", response_model=ApiResponse[TaskResponse])
async def create_task(
    dataset_id: str,
    doc_id: str,
    mode: ProcessingMode = ProcessingMode.CLASSIC
):
    """Create a processing task for an existing document."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if document exists
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    doc = cursor.fetchone()

    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    # Parse document metadata
    doc_metadata = json.loads(doc["metadata"]) if "metadata" in doc.keys() and doc["metadata"] else None
    
    timestamp = now()
    task_id = generate_id()
    
    # Create Task record
    cursor.execute(
        """
        INSERT INTO tasks (id, dataset_id, doc_id, mode, status, progress, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, dataset_id, doc_id, mode.value, TaskStatus.PENDING, 0, timestamp, timestamp)
    )
    conn.commit()
    
    # Update document status and task_id
    cursor.execute(
        "UPDATE documents SET status = ?, task_id = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.PROCESSING, task_id, timestamp, doc_id)
    )
    conn.commit()
    conn.close()
    
    return ApiResponse(
        success=True,
        code=200,
        message="Task created successfully",
        data=TaskResponse(
            task_id=str(task_id),
            dataset_id=dataset_id,
            doc_id=doc_id,
            mode=mode.value,
            status=TaskStatus.PENDING,
            progress=0,
            metadata=doc_metadata,
            created_at=timestamp,
            updated_at=timestamp
        )
    )


@router.get("/{doc_id}/views", response_model=ApiResponse[list])
def get_document_views(
    dataset_id: str,
    doc_id: str,
    level: Optional[str] = None,
):
    """Get LOD views for a document.

    Args:
        level: Filter by level - low | medium | high | full. Returns all if omitted.

    Response items:
        - LOW/MEDIUM/FULL: {level, content, token_count}
        - HIGH (DocTree):  {level, tree, token_count}
    """
    from zag.storages.vector import QdrantVectorStore
    from zag.embedders import Embedder
    from zag.schemas import LODLevel

    # Validate level param
    valid_levels = {lv.value for lv in LODLevel}
    if level and level not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{level}'. Must be one of: {sorted(valid_levels)}"
        )

    # Resolve collection_name
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

    # Fetch LOD unit via QdrantVectorStore.fetch
    try:
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            grpc_port=config.VECTOR_STORE_GRPC_PORT,
            prefer_grpc=True,
            collection_name=collection_name,
            embedder=embedder,
        )
        units = vector_store.fetch({
            "doc_id": doc_id,
            "metadata.custom.mode": "lod",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch views: {str(e)}")

    if not units:
        raise HTTPException(status_code=404, detail="No LOD views found for this document")

    # Use the first (and only) LOD unit
    lod_unit = units[0]
    if not lod_unit.views:
        raise HTTPException(status_code=404, detail="LOD unit has no views")

    # Build response items, optionally filtered by level
    result = []
    tags = _extract_tags(lod_unit)
    for view in lod_unit.views:
        if level and view.level != level:
            continue

        if view.level == LODLevel.HIGH:
            result.append({
                "level": view.level if isinstance(view.level, str) else view.level.value,
                "tree": view.content,
                "token_count": view.token_count,
                "tags": tags,
            })
        else:
            result.append({
                "level": view.level if isinstance(view.level, str) else view.level.value,
                "content": view.content,
                "token_count": view.token_count,
                "tags": tags,
            })

    return ApiResponse(success=True, code=200, data=result)


@router.get("/{doc_id}/tasks", response_model=ApiResponse[List[TaskResponse]])
def list_document_tasks(dataset_id: str, doc_id: str):
    """Get all tasks for a specific document."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if document exists
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Query tasks and JOIN documents to get metadata
    cursor.execute(
        """
        SELECT t.*, d.metadata as doc_metadata
        FROM tasks t
        LEFT JOIN documents d ON t.doc_id = d.id
        WHERE t.dataset_id = ? AND t.doc_id = ?
        ORDER BY t.created_at DESC
        """,
        (dataset_id, doc_id)
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        error_message = json.loads(row["error_message"]) if row["error_message"] else None
        doc_metadata = json.loads(row["doc_metadata"]) if "doc_metadata" in row.keys() and row["doc_metadata"] else None
        results.append(TaskResponse(
            task_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            doc_id=str(row["doc_id"]),
            mode=row["mode"] if "mode" in row.keys() else "classic",
            status=row["status"],
            progress=row["progress"],
            metadata=doc_metadata,
            error_message=error_message,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("", response_model=ApiResponse[List[DocumentResponse]])
def list_documents(dataset_id: str, status: Optional[str] = None):
    """List documents in a dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Query documents
    if status:
        cursor.execute(
            "SELECT * FROM documents WHERE dataset_id = ? AND status = ? ORDER BY created_at DESC",
            (dataset_id, status)
        )
    else:
        cursor.execute(
            "SELECT * FROM documents WHERE dataset_id = ? ORDER BY created_at DESC",
            (dataset_id,)
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        doc_metadata = json.loads(row["metadata"]) if "metadata" in row.keys() and row["metadata"] else None
        results.append(DocumentResponse(
            doc_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            file_name=row["file_name"],
            file_path=row["file_path"],
            workspace_dir=row["workspace_dir"],
            file_size=row["file_size"],
            file_type=row["file_type"],
            file_hash=row["file_hash"] if "file_hash" in row.keys() else None,
            metadata=doc_metadata,
            status=row["status"],
            task_id=str(row["task_id"]) if row["task_id"] else None,
            unit_count=row["unit_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/{doc_id}", response_model=ApiResponse[DocumentResponse])
def get_document(dataset_id: str, doc_id: str):
    """Get document by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Parse document metadata
    doc_metadata = json.loads(row["metadata"]) if "metadata" in row.keys() and row["metadata"] else None
    
    # Build file_url for the worker to fetch the file
    stored_path = row["file_path"]

    if stored_path.startswith("s3://"):
        # S3-hosted file: the worker downloads it directly from S3
        file_url = stored_path
    else:
        # Local file: expose through the API's /files/ endpoint
        # Normalize path for cross-platform compatibility
        raw_path = stored_path.replace("\\", "/")
        file_path = Path(raw_path)

        # Make path absolute if it's relative
        if not file_path.is_absolute():
            file_path = Path.cwd() / file_path

        file_path = file_path.resolve()

        # Convert to relative path from UPLOAD_DIR
        storage = get_storage()
        base_dir = Path(storage.base_dir).resolve()

        try:
            rel_path = file_path.relative_to(base_dir)
            rel_path_url = rel_path.as_posix()
            file_url = f"http://{config.API_PUBLIC_HOST}:{config.API_PORT}/files/{rel_path_url}"
        except ValueError:
            # file_path is not under base_dir
            file_url = None
    
    data = DocumentResponse(
        doc_id=str(row["id"]),
        dataset_id=str(row["dataset_id"]),
        file_name=row["file_name"],
        file_path=row["file_path"],
        file_url=file_url,
        workspace_dir=row["workspace_dir"],
        file_size=row["file_size"],
        file_type=row["file_type"],
        file_hash=row["file_hash"] if "file_hash" in row.keys() else None,
        metadata=doc_metadata,
        status=row["status"],
        task_id=str(row["task_id"]) if row["task_id"] else None,
        unit_count=row["unit_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
    
    return ApiResponse(success=True, code=200, data=data)


@router.patch("/{doc_id}", response_model=ApiResponse[DocumentResponse])
async def update_document(
    dataset_id: str,
    doc_id: str,
    metadata: str = Body(..., embed=True)
):
    """Update document metadata."""
    # Validate metadata first
    try:
        metadata_dict = _validate_metadata(metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    conn = get_connection()
    cursor = conn.cursor()

    # Check if document exists
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    # Update metadata
    timestamp = now()
    metadata_json = json.dumps(metadata_dict)

    cursor.execute(
        "UPDATE documents SET metadata = ?, updated_at = ? WHERE id = ?",
        (metadata_json, timestamp, doc_id)
    )
    conn.commit()

    # Fetch updated record
    cursor.execute(
        "SELECT * FROM documents WHERE id = ?",
        (doc_id,)
    )
    updated = cursor.fetchone()
    conn.close()

    doc_metadata = json.loads(updated["metadata"]) if updated["metadata"] else None

    return ApiResponse(
        success=True,
        code=200,
        message="Document updated successfully",
        data=DocumentResponse(
            doc_id=str(updated["id"]),
            dataset_id=str(updated["dataset_id"]),
            file_name=updated["file_name"],
            file_path=updated["file_path"],
            workspace_dir=updated["workspace_dir"],
            file_size=updated["file_size"],
            file_type=updated["file_type"],
            file_hash=updated["file_hash"],
            metadata=doc_metadata,
            status=updated["status"],
            task_id=str(updated["task_id"]) if updated["task_id"] else None,
            unit_count=updated["unit_count"],
            created_at=updated["created_at"],
            updated_at=updated["updated_at"]
        )
    )


@router.delete("/{doc_id}", response_model=ApiResponse[MessageResponse])
def delete_document(dataset_id: str, doc_id: str):
    """Delete document and cleanup vector store."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if document exists and get dataset info
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Get dataset name as collection_name
    cursor.execute("SELECT name FROM datasets WHERE id = ?", (dataset_id,))
    dataset = cursor.fetchone()
    if not dataset:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    collection_name = dataset["name"]
    
    # Delete from database
    cursor.execute("DELETE FROM tasks WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    
    # Cleanup vector store
    try:
        from zag.storages.vector import QdrantVectorStore
        from zag.embedders import Embedder
        
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
        vector_store.delete_by_doc_id(doc_id)
    except Exception as e:
        # Log but don't fail the API call
        print(f"Warning: Failed to cleanup vector store for doc_id {doc_id}: {e}")

    # Cleanup fulltext store (Meilisearch)
    try:
        from zag.indexers import FullTextIndexer

        fulltext_indexer = FullTextIndexer(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            auto_create_index=False,
        )
        deleted = fulltext_indexer.delete_by_doc_id(doc_id)
        if deleted:
            print(f"Cleaned up {deleted} units from Meilisearch for doc_id {doc_id}")
    except Exception as e:
        print(f"Warning: Failed to cleanup Meilisearch for doc_id {doc_id}: {e}")
    
    return ApiResponse(
        success=True,
        code=200,
        message="Document deleted successfully",
        data=MessageResponse(message="Document deleted successfully")
    )


@router.post("/{doc_id}/cache", response_model=ApiResponse[CacheUploadResponse])
async def upload_document_cache(
    dataset_id: str,
    doc_id: str,
    file: UploadFile = File(...),
):
    """
    Upload PDF parsing cache from worker.
    
    Worker calls this after processing a PDF to upload the cache directory.
    The cache is extracted to PDF_CACHE_DIR/{doc_id}/ for later use by /locate API.
    
    Args:
        file: tar.gz archive of the cache directory
        
    Returns:
        Cache upload confirmation with path and size
    """
    import tarfile
    import shutil
    
    # Validate file type
    if not file.filename.endswith('.tar.gz'):
        raise HTTPException(status_code=400, detail="File must be a .tar.gz archive")
    
    # Prepare cache directory
    cache_dir = Path(config.PDF_CACHE_DIR) / doc_id
    
    # Remove existing cache if present
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    # Save uploaded file to temp location
    temp_file = Path(tempfile.gettempdir()) / f"cache_{doc_id}.tar.gz"
    try:
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract archive
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(temp_file, "r:gz") as tar:
            tar.extractall(path=cache_dir)
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        
        return ApiResponse(
            success=True,
            code=200,
            message="Cache uploaded successfully",
            data=CacheUploadResponse(
                doc_id=doc_id,
                cache_path=str(cache_dir),
                size_bytes=total_size,
                message=f"Cache extracted to {cache_dir}"
            )
        )
    except Exception as e:
        # Cleanup on error
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        raise HTTPException(status_code=500, detail=f"Failed to process cache: {str(e)}")
    finally:
        # Cleanup temp file
        temp_file.unlink(missing_ok=True)


@router.post("/locate", response_model=ApiResponse[LocatePageResponse])
async def locate_pages(
    dataset_id: str,
    request: LocatePageRequest = Body(...),
):
    """
    Locate page numbers for text snippets.
    
    Batch API that accepts multiple items, each with doc_id + text_start (+ optional text_end).
    Returns page numbers for each item.
    
    Performance:
    - Concurrent PDF loading for different docs
    - Concurrent text search within each PDF
    """
    import asyncio
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    
    from zag.schemas.pdf import PDF
    from zag.utils.page_inference import fuzzy_find_start
    
    # Cache directory for PDF parsing results
    cache_dir = Path(config.PDF_CACHE_DIR)
    
    # Group items by doc_id for efficient processing
    doc_groups: dict[str, list] = defaultdict(list)
    for item in request.items:
        doc_groups[item.doc_id].append(item)
    
    def find_pages_for_text(pdf: PDF, text_start: str, text_end: str = None) -> tuple[list[int] | None, bool]:
        """
        Find page numbers for a text snippet.
        
        Returns:
            (page_numbers, found)
        """
        # Find start position
        # Search entire document (no range limit for correctness)
        start_pos = fuzzy_find_start(text_start, pdf.content, start_from=0, threshold=0.80)
        
        if start_pos is None:
            return None, False
        
        # Determine end position
        if text_end:
            end_pos = fuzzy_find_start(text_end, pdf.content, start_from=start_pos + len(text_start), threshold=0.80, max_search_range=100000)
            if end_pos is None:
                # Fallback: estimate length from text_start
                end_pos = start_pos + len(text_start)
            else:
                end_pos = end_pos + len(text_end)
        else:
            end_pos = start_pos + len(text_start)
        
        # Build page positions
        page_positions = []
        current_pos = 0
        
        for page in pdf.pages:
            page_content = page.content or ""
            if not page_content.strip():
                page_positions.append((current_pos, current_pos, page.page_number))
                continue
            
            # Find page in full content
            page_sig = page_content[:50] if len(page_content) > 50 else page_content
            pos = pdf.content.find(page_sig, current_pos)
            
            if pos != -1:
                page_positions.append((pos, pos + len(page_content), page.page_number))
                current_pos = pos + 1
            else:
                page_positions.append((current_pos, current_pos + len(page_content), page.page_number))
                current_pos += len(page_content) + 1
        
        # Find overlapping pages
        overlapping = []
        for page_start, page_end, page_num in page_positions:
            if not (end_pos <= page_start or start_pos >= page_end):
                overlapping.append(page_num)
        
        return sorted(overlapping) if overlapping else None, True
    
    def process_single_doc(doc_id: str, items: list) -> list[LocatePageResult]:
        """Process all items for a single document."""
        results = []
        
        # Load PDF
        doc_cache_dir = cache_dir / doc_id
        if not doc_cache_dir.exists():
            for item in items:
                results.append(LocatePageResult(
                    request_id=item.request_id,
                    doc_id=doc_id,
                    page_numbers=None,
                    found=False,
                    error=f"Document cache not found: {doc_id}"
                ))
            return results
        
        try:
            pdf = PDF.load(doc_cache_dir)
        except Exception as e:
            for item in items:
                results.append(LocatePageResult(
                    request_id=item.request_id,
                    doc_id=doc_id,
                    page_numbers=None,
                    found=False,
                    error=f"Failed to load document: {str(e)}"
                ))
            return results
        
        # Process all items for this doc
        for item in items:
            try:
                page_numbers, found = find_pages_for_text(pdf, item.text_start, item.text_end)
                results.append(LocatePageResult(
                    request_id=item.request_id,
                    doc_id=doc_id,
                    page_numbers=page_numbers,
                    found=found,
                    error=None if found else "Text not found in document"
                ))
            except Exception as e:
                results.append(LocatePageResult(
                    request_id=item.request_id,
                    doc_id=doc_id,
                    page_numbers=None,
                    found=False,
                    error=f"Search error: {str(e)}"
                ))
        
        return results
    
    # Process all docs concurrently
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, process_single_doc, doc_id, items)
            for doc_id, items in doc_groups.items()
        ]
        all_results = await asyncio.gather(*futures)
    
    # Flatten results
    results = []
    for doc_results in all_results:
        results.extend(doc_results)
    
    return ApiResponse(
        success=True,
        code=200,
        data=LocatePageResponse(results=results)
    )
