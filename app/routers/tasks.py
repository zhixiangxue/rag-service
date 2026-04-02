"""Task API endpoints."""
import logging
import threading
from fastapi import APIRouter, HTTPException
from typing import List, Optional
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from ..database import get_connection, now
from ..repositories import TaskRepository, DocumentRepository
from ..schemas import (
    TaskResponse,
    TaskStatusUpdate,
    ApiResponse,
    MessageResponse,
    ReaderType
)
from ..constants import TaskStatus, DocumentStatus
from ..worker import later

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tasks"])

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _post_callback(url: str, payload: dict) -> None:
    """POST callback payload to the given URL with tenacity retry (up to 5 attempts)."""
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()


def _dispatch_callback(url: str, payload: dict) -> None:
    """Fire callback in a background thread; log and discard after 5 failed attempts."""
    def _run():
        try:
            _post_callback(url, payload)
            logger.info("Callback delivered: url=%s task_id=%s", url, payload.get("task_id"))
        except Exception as exc:
            logger.error(
                "Callback permanently failed after 5 attempts: url=%s task_id=%s error=%s",
                url, payload.get("task_id"), exc,
            )
    threading.Thread(target=_run, daemon=True).start()

def _row_to_task_response(row: dict) -> TaskResponse:
    """Convert a task dict row to TaskResponse."""
    error_message = json.loads(row["error_message"]) if row["error_message"] else None
    return TaskResponse(
        task_id=str(row["id"]),
        dataset_id=str(row["dataset_id"]),
        doc_id=str(row["doc_id"]),
        mode=row.get("mode", "classic"),
        reader=row.get("reader", ReaderType.DEFAULT),
        status=row["status"],
        progress=row["progress"],
        callback=row.get("callback"),
        error_message=error_message,
        worker=row.get("worker"),
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )


@router.get("/tasks", response_model=ApiResponse[List[TaskResponse]])
def get_all_tasks(limit: int = 10, status: Optional[str] = None):
    """Get all tasks (optionally filtered by status).
    
    Args:
        limit: Maximum number of tasks to return (default: 10)
        status: Filter by task status (PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)
    """
    conn = get_connection()
    rows = TaskRepository(conn).list_all(status=status, limit=limit)
    conn.close()

    return ApiResponse(
        success=True, code=200,
        data=[_row_to_task_response(row) for row in rows]
    )


@router.get("/tasks/stats", response_model=ApiResponse[dict])
def get_task_stats():
    """Get task statistics grouped by status.
    
    Returns:
        - Total count
        - Count by status (PENDING, PROCESSING, COMPLETED, FAILED, CANCELLED)
        - Progress percentage (completed / total)
    """
    conn = get_connection()
    repo = TaskRepository(conn)
    total = repo.count_total()
    status_counts = repo.count_by_status()
    conn.close()

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
    row = TaskRepository(conn).get(task_id)
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Task not found")

    return ApiResponse(success=True, code=200, data=_row_to_task_response(row))


@router.patch("/tasks/{task_id}", response_model=ApiResponse[TaskResponse])
def update_task(task_id: str, update: TaskStatusUpdate):
    """Update task (status, progress, error, etc)."""
    conn = get_connection()
    task_repo = TaskRepository(conn)
    doc_repo = DocumentRepository(conn)

    row = task_repo.get(task_id)
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

    # Build fields dict for task update
    fields = {}

    if update.status is not None:
        fields["status"] = update.status

        # Auto-set progress for terminal states
        if update.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            fields["progress"] = 100
        elif update.status == TaskStatus.PROCESSING and update.progress is None:
            fields["progress"] = 10

    if update.progress is not None:
        # Only allow progress update for PROCESSING status
        if current_status != TaskStatus.PROCESSING and update.status != TaskStatus.PROCESSING:
            conn.close()
            raise HTTPException(
                status_code=400,
                detail="Progress can only be updated for tasks in PROCESSING status"
            )

        if not 0 <= update.progress <= 100:
            conn.close()
            raise HTTPException(status_code=400, detail="Progress must be between 0 and 100")

        # Don't overwrite terminal-state progress already set above
        if update.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            fields["progress"] = update.progress

    if update.error_message is not None:
        fields["error_message"] = json.dumps(update.error_message)

    if update.worker is not None:
        fields["worker"] = update.worker

    if update.unit_count is not None:
        doc_repo.update_unit_count(doc_id, update.unit_count)

    if not fields:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")

    timestamp = now()
    task_repo.update(task_id, fields, timestamp)

    # Sync document status with task status
    if update.status == TaskStatus.COMPLETED:
        doc_repo.update_status(doc_id, DocumentStatus.COMPLETED, timestamp)
    elif update.status == TaskStatus.FAILED:
        doc_repo.update_status(doc_id, DocumentStatus.FAILED, timestamp)
    elif update.status == TaskStatus.CANCELLED:
        doc_repo.update_status(doc_id, DocumentStatus.FAILED, timestamp)
    elif update.status == TaskStatus.PROCESSING:
        doc_repo.update_status(doc_id, DocumentStatus.PROCESSING, timestamp)

    row = task_repo.get(task_id)
    conn.close()

    # Fire callback if task reached a terminal state
    if update.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}:
        callback_url = row.get("callback") if row else None
        if callback_url:
            payload = {
                "task_id": task_id,
                "dataset_id": row["dataset_id"],
                "doc_id": row["doc_id"],
                "status": row["status"],
                "progress": row["progress"],
                "error_message": json.loads(row["error_message"]) if row.get("error_message") else None,
                "timestamp": row["updated_at"],
            }
            _dispatch_callback(callback_url, payload)

    return ApiResponse(success=True, code=200, message="Task updated successfully",
                       data=_row_to_task_response(row))


@router.post("/tasks/{task_id}/retry", response_model=ApiResponse[MessageResponse])
def retry_task(task_id: str):
    """Retry a task (reset terminal state task to PENDING).
    
    This endpoint allows retrying tasks in any terminal state (COMPLETED, FAILED, or CANCELLED).
    The task will be reset to PENDING status and picked up by workers again.
    """
    conn = get_connection()
    task_repo = TaskRepository(conn)
    doc_repo = DocumentRepository(conn)

    row = task_repo.get(task_id)
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
    task_repo.reset(task_id, timestamp)
    doc_repo.update_status(doc_id, DocumentStatus.PROCESSING, timestamp)
    conn.close()

    # Re-enqueue to Dramatiq so the worker picks it up immediately
    later.process_document(task_id)

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
    task_repo = TaskRepository(conn)
    doc_repo = DocumentRepository(conn)

    row = task_repo.get(task_id)
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
    task_repo.cancel(task_id, timestamp)
    doc_repo.update_status(doc_id, DocumentStatus.FAILED, timestamp)
    conn.close()

    return ApiResponse(
        success=True,
        code=200,
        message=f"Task {task_id} has been cancelled",
        data=MessageResponse(message=f"Task cancelled from {current_status} to CANCELLED")
    )

