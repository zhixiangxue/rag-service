"""Document API endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import os
import json
import xxhash

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


@router.post("", response_model=ApiResponse[dict])
async def upload_file(
    dataset_id: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Upload file to dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read file content
    file_content = await file.read()
    
    # Calculate file hash using xxhash
    file_hash = xxhash.xxh64(file_content).hexdigest()
    
    # Check for duplicate: same dataset + same file_hash
    cursor.execute(
        "SELECT id, file_name FROM documents WHERE dataset_id = ? AND file_hash = ?",
        (dataset_id, file_hash)
    )
    existing_doc = cursor.fetchone()
    
    if existing_doc:
        # File already uploaded, return existing doc_id
        conn.close()
        return ApiResponse(
            success=True,
            code=200,
            message="File already exists, reusing existing document",
            data={
                "dataset_id": str(dataset_id),
                "doc_id": str(existing_doc['id']),
                "file_name": existing_doc['file_name'],
                "file_hash": file_hash,
            }
        )
    
    # Save file using storage abstraction
    storage = get_storage()
    
    # Reset file pointer for storage.save
    await file.seek(0)
    file_path = storage.save(file.file, file.filename, dataset_id)
    
    # Extract workspace directory (parent of file)
    workspace_dir = os.path.dirname(file_path)
    
    file_size = len(file_content)
    file_type = file.filename.split(".")[-1] if "." in file.filename else "unknown"
    
    # Parse and validate metadata
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            if not isinstance(metadata_dict, dict):
                raise HTTPException(status_code=400, detail="Metadata must be a JSON object")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Metadata must be valid JSON")
    else:
        metadata_dict = {}
    metadata_json = json.dumps(metadata_dict)
    
    timestamp = now()
    doc_id = generate_id()
    
    # Create Document record with file_hash and metadata
    cursor.execute(
        """
        INSERT INTO documents (id, dataset_id, file_name, file_path, workspace_dir, file_size, file_type, file_hash, metadata, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, dataset_id, file.filename, file_path, workspace_dir, file_size, file_type, file_hash, metadata_json, DocumentStatus.PROCESSING, timestamp, timestamp)
    )
    conn.commit()
    conn.close()
    
    return ApiResponse(
        success=True,
        code=200,
        message="File uploaded successfully",
        data={
            "dataset_id": str(dataset_id),
            "doc_id": str(doc_id),
            "file_name": file.filename,
            "file_path": file_path,
            "file_hash": file_hash,
        }
    )


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
            created_at=timestamp,
            updated_at=timestamp
        )
    )


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
        results.append(DocumentResponse(
            doc_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            file_name=row["file_name"],
            file_path=row["file_path"],
            workspace_dir=row["workspace_dir"],
            file_size=row["file_size"],
            file_type=row["file_type"],
            file_hash=row["file_hash"] if "file_hash" in row.keys() else None,
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
    
    data = DocumentResponse(
        doc_id=str(row["id"]),
        dataset_id=str(row["dataset_id"]),
        file_name=row["file_name"],
        file_path=row["file_path"],
        workspace_dir=row["workspace_dir"],
        file_size=row["file_size"],
        file_type=row["file_type"],
        file_hash=row["file_hash"] if "file_hash" in row.keys() else None,
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
    
    return ApiResponse(
        success=True,
        code=200,
        message="Document deleted successfully",
        data=MessageResponse(message="Document deleted successfully")
    )
