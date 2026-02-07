"""Task API endpoints."""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import json

from ..database import get_connection, now
from ..schemas import (
    TaskResponse,
    TaskStatusUpdate,
    ApiResponse,
    MessageResponse
)
from ..constants import TaskStatus, DocumentStatus

router = APIRouter(tags=["tasks"])


@router.get("/datasets/{dataset_id}/tasks", response_model=ApiResponse[List[TaskResponse]])
def list_tasks_by_dataset(dataset_id: str):
    """Get all tasks for a specific dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    cursor.execute(
        "SELECT * FROM tasks WHERE dataset_id = ? ORDER BY created_at DESC",
        (dataset_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        error_message = json.loads(row["error_message"]) if row["error_message"] else None
        results.append(TaskResponse(
            task_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            doc_id=str(row["doc_id"]),
            mode=row["mode"] if "mode" in row.keys() else "classic",
            status=row["status"],
            progress=row["progress"],
            error_message=error_message,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/tasks", response_model=ApiResponse[List[TaskResponse]])
def get_all_tasks(limit: int = 10, status: Optional[str] = None):
    """Get all tasks (optionally filtered by status).
    
    Args:
        limit: Maximum number of tasks to return (default: 10)
        status: Filter by task status (PENDING, PROCESSING, COMPLETED, FAILED)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    if status:
        cursor.execute(
            "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
            (status, limit)
        )
    else:
        cursor.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
    
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        error_message = json.loads(row["error_message"]) if row["error_message"] else None
        results.append(TaskResponse(
            task_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            doc_id=str(row["doc_id"]),
            status=row["status"],
            progress=row["progress"],
            error_message=error_message,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/tasks/pending", response_model=ApiResponse[List[TaskResponse]])
def get_pending_tasks(limit: int = 10):
    """Get pending tasks for worker to process."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # JOIN with documents to get metadata
    cursor.execute(
        """
        SELECT t.*, d.metadata as doc_metadata 
        FROM tasks t
        JOIN documents d ON t.doc_id = d.id
        WHERE t.status = ? 
        ORDER BY t.created_at ASC 
        LIMIT ?
        """,
        (TaskStatus.PENDING, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        error_message = json.loads(row["error_message"]) if row["error_message"] else None
        metadata = json.loads(row["doc_metadata"]) if row["doc_metadata"] else None
        results.append(TaskResponse(
            task_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            doc_id=str(row["doc_id"]),
            mode=row["mode"] if "mode" in row.keys() else "classic",
            status=row["status"],
            progress=row["progress"],
            metadata=metadata,
            error_message=error_message,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/tasks/{task_id}", response_model=ApiResponse[TaskResponse])
def get_task(task_id: str):
    """Get task status by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    
    error_message = json.loads(row["error_message"]) if row["error_message"] else None
    
    data = TaskResponse(
        task_id=str(row["id"]),
        dataset_id=str(row["dataset_id"]),
        doc_id=str(row["doc_id"]),
        mode=row["mode"] if "mode" in row.keys() else "classic",
        status=row["status"],
        progress=row["progress"],
        error_message=error_message,
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
    
    return ApiResponse(success=True, code=200, data=data)


@router.patch("/tasks/{task_id}", response_model=ApiResponse[TaskResponse])
def update_task(task_id: str, update: TaskStatusUpdate):
    """Update task (status, progress, error, etc)."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if task exists and get current status
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    
    current_status = row["status"]
    doc_id = row["doc_id"]
    
    # Validate status transition
    if update.status is not None and not TaskStatus.is_valid_transition(current_status, update.status):
        conn.close()
        raise HTTPException(
            status_code=409,
            detail=f"Invalid status transition: {current_status} -> {update.status}"
        )
    
    # Build update query
    updates = []
    params = []
    
    if update.status is not None:
        updates.append("status = ?")
        params.append(update.status)
        
        # Auto-set progress to 100 for terminal states
        if update.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            updates.append("progress = ?")
            params.append(100)
        elif update.status == TaskStatus.PROCESSING and update.progress is None:
            # Set initial progress for PROCESSING if not provided
            updates.append("progress = ?")
            params.append(10)
    
    if update.progress is not None:
        # Only allow progress update for PROCESSING status
        if current_status != TaskStatus.PROCESSING and update.status != TaskStatus.PROCESSING:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="Progress can only be updated for tasks in PROCESSING status"
            )
        
        # Validate progress range
        if not 0 <= update.progress <= 100:
            conn.close()
            raise HTTPException(status_code=400, detail="Progress must be between 0 and 100")
        
        # Don't add duplicate progress update if already added for terminal state
        if update.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            updates.append("progress = ?")
            params.append(update.progress)
    
    if update.error_message is not None:
        updates.append("error_message = ?")
        params.append(json.dumps(update.error_message))
    
    if update.unit_count is not None:
        # Update document's unit_count
        cursor.execute(
            "UPDATE documents SET unit_count = ? WHERE id = ?",
            (update.unit_count, doc_id)
        )
    
    if not updates:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")
    
    timestamp = now()
    updates.append("updated_at = ?")
    params.append(timestamp)
    params.append(task_id)
    
    cursor.execute(
        f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
        params
    )
    conn.commit()
    
    # Update document status based on task status
    if update.status == TaskStatus.COMPLETED:
        cursor.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            (DocumentStatus.COMPLETED, timestamp, doc_id)
        )
        conn.commit()
    elif update.status == TaskStatus.FAILED:
        cursor.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            (DocumentStatus.FAILED, timestamp, doc_id)
        )
        conn.commit()
    elif update.status == TaskStatus.PROCESSING:
        cursor.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            (DocumentStatus.PROCESSING, timestamp, doc_id)
        )
        conn.commit()
    
    # Fetch updated task
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    conn.close()
    
    error_message = json.loads(row["error_message"]) if row["error_message"] else None
    
    data = TaskResponse(
        task_id=str(row["id"]),
        dataset_id=str(row["dataset_id"]),
        doc_id=str(row["doc_id"]),
        mode=row["mode"] if "mode" in row.keys() else "classic",
        status=row["status"],
        progress=row["progress"],
        error_message=error_message,
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
    
    return ApiResponse(success=True, code=200, message="Task updated successfully", data=data)
