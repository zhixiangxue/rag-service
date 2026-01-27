"""Document API endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import os
import json
import xxhash

from ..database import get_connection, now
from ..schemas import (
    DocumentResponse,
    ApiResponse,
    MessageResponse,
    TaskResponse
)
from ..storage import get_storage
from ..constants import TaskStatus, DocumentStatus

router = APIRouter(prefix="/datasets/{dataset_id}/documents", tags=["documents"])


@router.post("", response_model=ApiResponse[dict])
async def ingest_file(
    dataset_id: str,
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None)
):
    """Upload file and create ingest task."""
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
        conn.close()
        raise HTTPException(
            status_code=409,
            detail=f"File with same content already exists in this dataset (doc_id: {existing_doc['id']}, file: {existing_doc['file_name']})"
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
    
    # Parse metadata
    metadata_dict = json.loads(metadata) if metadata else None
    
    timestamp = now()
    
    # Create Document record with file_hash
    cursor.execute(
        """
        INSERT INTO documents (dataset_id, file_name, file_path, workspace_dir, file_size, file_type, file_hash, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, file.filename, file_path, workspace_dir, file_size, file_type, file_hash, DocumentStatus.PROCESSING, timestamp, timestamp)
    )
    conn.commit()
    doc_id = cursor.lastrowid
    
    # Create Task record
    cursor.execute(
        """
        INSERT INTO tasks (dataset_id, doc_id, status, progress, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (dataset_id, doc_id, TaskStatus.PENDING, 0, timestamp, timestamp)
    )
    conn.commit()
    task_id = cursor.lastrowid
    
    # Update document with task_id
    cursor.execute(
        "UPDATE documents SET task_id = ? WHERE id = ?",
        (task_id, doc_id)
    )
    conn.commit()
    conn.close()
    
    # Store file path and metadata in a way worker can access
    # (For now, worker will need to query task and find document)
    
    return ApiResponse(
        success=True,
        code=200,
        message="File uploaded and task created",
        data={
            "task_id": str(task_id),
            "doc_id": str(doc_id),
            "file_name": file.filename,
            "file_path": file_path,
            "file_hash": file_hash
        }
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
    """Delete (disable) document."""
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
    
    # Soft delete: update status to DISABLED
    timestamp = now()
    cursor.execute(
        "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.DISABLED, timestamp, doc_id)
    )
    conn.commit()
    conn.close()
    
    # TODO: Worker should delete associated units from vector store
    
    return ApiResponse(
        success=True,
        code=200,
        message="Document disabled successfully",
        data=MessageResponse(message="Document disabled successfully")
    )
