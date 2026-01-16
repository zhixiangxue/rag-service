"""Worker process for polling and processing tasks."""
import time
import requests
import schedule
import asyncio
from pathlib import Path

from . import config
from .processor import process_document
from ..app.constants import TaskStatus


def fetch_pending_tasks(limit: int = 1):
    """Fetch pending tasks from API.
    
    Args:
        limit: Maximum 1 task per fetch (enforced)
    """
    # IMPORTANT: Only fetch 1 task at a time to avoid unnecessary locking
    # Task processing is time-consuming, no need to fetch multiple
    limit = 1
    
    try:
        response = requests.get(
            f"{config.API_BASE_URL}/tasks/pending",
            params={"limit": limit},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])
    except Exception as e:
        print(f"Error fetching tasks: {e}")
        return []


def update_task(task_id: str, status: str, progress: int = 0, error_message: str = None, unit_count: int = None):
    """Update task via API.
    
    Returns:
        bool: True if update successful, False if failed (e.g., task already locked)
    """
    try:
        payload = {
            "status": status,
            "progress": progress
        }
        if error_message:
            payload["error_message"] = error_message
        if unit_count is not None:
            payload["unit_count"] = unit_count
        
        response = requests.patch(
            f"{config.API_BASE_URL}/tasks/{task_id}",
            json=payload,
            timeout=10
        )
        
        # Handle 409 Conflict (task already locked by another worker)
        if response.status_code == 409:
            print(f"  Task {task_id} already locked by another worker")
            return False
        
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error updating task status: {e}")
        return False


def get_document_info(dataset_id: str, doc_id: str):
    """Get document information from API."""
    try:
        response = requests.get(
            f"{config.API_BASE_URL}/datasets/{dataset_id}/documents/{doc_id}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("data")
    except Exception as e:
        print(f"Error fetching document info: {e}")
        return None


def process_task(task):
    """Process a single task."""
    task_id = task["task_id"]
    dataset_id = task["dataset_id"]
    doc_id = task["doc_id"]
    
    print(f"\n{'='*60}")
    print(f"Processing task: {task_id}")
    print(f"  Dataset: {dataset_id}")
    print(f"  Document: {doc_id}")
    print(f"{'='*60}")
    
    # IMPORTANT: Immediately update to PROCESSING to lock the task
    # This prevents other workers from picking up the same task
    if not update_task(task_id, TaskStatus.PROCESSING, progress=10):
        print(f"  ✗ Failed to lock task, skipping")
        return
    
    try:
        # Get document info
        doc_info = get_document_info(dataset_id, doc_id)
        if not doc_info:
            raise Exception("Document not found")
        
        file_path = doc_info["file_path"]
        workspace_dir = doc_info["workspace_dir"]
        
        print(f"  File: {file_path}")
        print(f"  Workspace: {workspace_dir}")
        
        # Validate file exists
        if not Path(file_path).exists():
            raise Exception(f"File not found: {file_path}")
        
        # Process document
        print(f"\n  Starting document processing...")
        result = asyncio.run(process_document(
            file_path=Path(file_path),
            workspace_dir=Path(workspace_dir),
            custom_metadata={
                "dataset_id": dataset_id,
                "doc_id": doc_id
            },
            on_progress=lambda progress: update_task(task_id, TaskStatus.PROCESSING, progress=progress)
        ))
        unit_count = result["unit_count"]
        
        print(f"\n  ✓ Processing completed")
        print(f"  Created {unit_count} units")
        
        # Update to COMPLETED
        update_task(task_id, TaskStatus.COMPLETED, progress=100, unit_count=unit_count)
        print(f"\n{'='*60}")
        print(f"Task {task_id} completed successfully")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n  ✗ Processing failed: {e}")
        update_task(task_id, TaskStatus.FAILED, error_message=str(e))
        print(f"\n{'='*60}")
        print(f"Task {task_id} failed")
        print(f"{'='*60}\n")


def poll_and_process_tasks():
    """Poll for pending tasks and process them."""
    try:
        # Fetch pending tasks
        tasks = fetch_pending_tasks(limit=1)
        
        if tasks:
            for task in tasks:
                process_task(task)
                
    except Exception as e:
        print(f"Error in poll_and_process_tasks: {e}")


def run_worker():
    """Main worker loop using schedule."""
    print("\n" + "="*60)
    print("Worker Started")
    print("="*60)
    print(f"API Base URL: {config.API_BASE_URL}")
    print(f"Poll Interval: {config.WORKER_POLL_INTERVAL}s")
    print(f"Vector Store: {config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT}")
    print(f"Embedding: {config.EMBEDDING_URI}")
    print("="*60 + "\n")
    
    # Schedule the polling task
    schedule.every(config.WORKER_POLL_INTERVAL).seconds.do(poll_and_process_tasks)
    
    # Run immediately on startup
    poll_and_process_tasks()
    
    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nWorker stopped by user")


if __name__ == "__main__":
    run_worker()
