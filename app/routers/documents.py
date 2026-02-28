"""Document API endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import json
import sqlite3
from pathlib import Path

from zag.utils.hash import calculate_file_hash
from ..utils.s3 import download_file_from_s3, download_file_from_s3_async
from ..database import get_connection, now, generate_id
from ..schemas import (
    DocumentResponse,
    ApiResponse,
    MessageResponse,
    TaskResponse,
    ProcessingMode
)
from ..storage import get_storage
from ..constants import TaskStatus, DocumentStatus
from .. import config

router = APIRouter(prefix="/datasets/{dataset_id}/documents", tags=["documents"])


def _create_document_record(
    dataset_id: str,
    file_path: str,
    filename: str,
    metadata: Optional[str] = None
) -> dict:
    """
    Create document record after file is saved.
    Handles dataset validation, hash calculation, duplicate check, and database insert.
    
    Args:
        dataset_id: Dataset ID
        file_path: Path to saved file
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
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
                if not isinstance(metadata_dict, dict):
                    raise ValueError("Metadata must be a JSON object")
            except json.JSONDecodeError:
                raise ValueError("Metadata must be valid JSON")
        else:
            metadata_dict = {}
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
    metadata: Optional[str] = Form(None)
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
    metadata: Optional[str] = Body(None, embed=True)
):
    """Download file from S3 URL and add to dataset."""
    filename = Path(s3_url).name
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid S3 URL")
    
    # Download file from S3 to temp location
    storage = get_storage()
    temp_dir = Path(storage.base_dir) / dataset_id / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / filename
    
    try:
        await download_file_from_s3_async(
            s3_url, 
            temp_file,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_KEY
        )
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] S3 download failed for {filename}: {error_msg}")
        
        # S3 404 errors should return 400 (Bad Request), not 500
        if "404" in error_msg or "Not Found" in error_msg:
            raise HTTPException(status_code=400, detail=f"File not found in S3: {s3_url}")
        else:
            raise HTTPException(status_code=500, detail=f"Failed to download file: {error_msg}")
    
    # Move to final location using storage abstraction
    try:
        with open(temp_file, 'rb') as f:
            file_path = storage.save(f, filename, dataset_id)
        temp_file.unlink()
    except Exception as e:
        print(f"[ERROR] File save failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Create document record (handles dataset validation, hash, duplicate check, insert)
    try:
        result = _create_document_record(dataset_id, file_path, filename, metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Document record creation failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create document record: {str(e)}")
    
    message = "File already exists, reusing existing document" if result["is_duplicate"] else "File uploaded successfully"
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
    
    # Generate file_url for distributed worker access
    # Normalize path for cross-platform compatibility
    raw_path = row["file_path"].replace("\\", "/")
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
        # Convert to POSIX format for URL
        rel_path_url = rel_path.as_posix()
        
        # Use API_PUBLIC_HOST for distributed workers
        file_url = f"http://{config.API_PUBLIC_HOST}:{config.API_PORT}/files/{rel_path_url}"
    except ValueError:
        # If file_path is not under base_dir, file_url is None
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
