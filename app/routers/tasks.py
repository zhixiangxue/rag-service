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


@router.get("/tasks", response_model=ApiResponse[List[TaskResponse]])
def get_all_tasks(limit: int = 10, status: Optional[str] = None):
    """Get all tasks (optionally filtered by status).
    
    Args:
        limit: Maximum number of tasks to return (default: 10)
        status: Filter by task status (PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)
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
    """Get pending tasks for worker to process.
    
    Note: Worker must call PATCH /tasks/{task_id} to claim task (update status to PROCESSING)
    to prevent multiple workers from processing the same task.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
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


@router.get("/tasks/stats", response_model=ApiResponse[dict])
def get_task_stats():
    """Get task statistics grouped by status.
    
    Returns:
        - Total count
        - Count by status (PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)
        - Progress percentage (completed / total)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM tasks")
    total = cursor.fetchone()[0]
    
    # Get count by status
    cursor.execute(
        "SELECT status, COUNT(*) as count FROM tasks GROUP BY status"
    )
    status_counts = {row["status"]: row["count"] for row in cursor.fetchall()}
    
    conn.close()
    
    # Calculate progress
    completed = status_counts.get(TaskStatus.COMPLETED, 0)
    progress = (completed / total * 100) if total > 0 else 0
    
    return ApiResponse(
        success=True,
        code=200,
        data={
            "total": total,
            "by_status": {
                "pending": status_counts.get(TaskStatus.PENDING, 0),
                "processing": status_counts.get(TaskStatus.PROCESSING, 0),
                "completed": status_counts.get(TaskStatus.COMPLETED, 0),
                "failed": status_counts.get(TaskStatus.FAILED, 0),
                "cancelled": status_counts.get(TaskStatus.CANCELLED, 0)
            },
            "progress": {
                "completed": completed,
                "total": total,
                "percentage": round(progress, 2)
            }
        }
    )


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
    elif update.status == TaskStatus.CANCELLED:
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


@router.post("/tasks/{task_id}/retry", response_model=ApiResponse[MessageResponse])
def retry_task(task_id: str):
    """Retry a task (reset terminal state task to PENDING).
    
    This endpoint allows retrying tasks in any terminal state (COMPLETED, FAILED, or CANCELLED).
    The task will be reset to PENDING status and picked up by workers again.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if task exists
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    
    current_status = row["status"]
    doc_id = row["doc_id"]
    
    # Only allow retry for terminal states
    if current_status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot retry task in {current_status} status. Only terminal states (COMPLETED/FAILED/CANCELLED) can be retried."
        )
    
    timestamp = now()
    
    # Reset task to PENDING
    cursor.execute(
        "UPDATE tasks SET status = ?, progress = ?, error_message = NULL, updated_at = ? WHERE id = ?",
        (TaskStatus.PENDING, 0, timestamp, task_id)
    )
    conn.commit()
    
    # Reset document status to PROCESSING
    cursor.execute(
        "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.PROCESSING, timestamp, doc_id)
    )
    conn.commit()
    conn.close()
    
    return ApiResponse(
        success=True,
        code=200,
        message=f"Task {task_id} has been reset to PENDING and will be retried",
        data=MessageResponse(message=f"Task reset from {current_status} to PENDING")
    )


@router.post("/tasks/{task_id}/cancel", response_model=ApiResponse[MessageResponse])
def cancel_task(task_id: str):
    """Cancel a running task.
    
    This endpoint allows cancelling tasks in non-terminal states (PENDING or PROCESSING).
    The task will be marked as CANCELLED and workers will stop processing it.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if task exists
    cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")
    
    current_status = row["status"]
    doc_id = row["doc_id"]
    
    # Only allow cancelling non-terminal states
    if current_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel task in {current_status} status. Task is already in terminal state."
        )
    
    timestamp = now()
    
    # Update task status to CANCELLED
    cursor.execute(
        "UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?",
        (TaskStatus.CANCELLED, timestamp, task_id)
    )
    conn.commit()
    
    # Update document status to FAILED (cancelled task treated as failed)
    cursor.execute(
        "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.FAILED, timestamp, doc_id)
    )
    conn.commit()
    conn.close()
    
    return ApiResponse(
        success=True,
        code=200,
        message=f"Task {task_id} has been cancelled",
        data=MessageResponse(message=f"Task cancelled from {current_status} to CANCELLED")
    )

