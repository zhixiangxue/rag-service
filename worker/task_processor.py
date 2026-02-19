"""Single task processor - runs in isolated subprocess.

This module processes a single task and exits, ensuring complete resource cleanup.
Each task runs in its own process to prevent GPU memory leaks.
"""
import sys
import os
import asyncio
import httpx
import tempfile
import gc
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .constants import TaskStatus, ProcessingMode
from .exceptions import ProcessingError, TaskCancelledException
from . import config
from .indexers.classic import index_classic
from .indexers.lod import index_lod

console = Console()


async def update_task_status(
    api_base_url: str,
    task_id: str,
    status: str,
    progress: Optional[int] = None,
    error_message: Optional[Dict[str, Any]] = None,
    unit_count: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """Update task status via API and return task data."""
    payload = {"status": status}
    if progress is not None:
        payload["progress"] = progress
    if error_message is not None:
        payload["error_message"] = error_message
    if unit_count is not None:
        payload["unit_count"] = unit_count
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{api_base_url}/tasks/{task_id}",
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            # Return task data from response
            return result.get("data") if result.get("success") else None
    except Exception as e:
        console.print(f"[red]Failed to update task status: {e}[/red]")
        return None


async def get_document_info(api_base_url: str, dataset_id: str, doc_id: str) -> Dict[str, Any]:
    """Get document information from API."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{api_base_url}/datasets/{dataset_id}/documents/{doc_id}"
        )
        response.raise_for_status()
        return response.json()["data"]


async def download_file(file_url: str, dest_path: Path) -> None:
    """Download file from HTTP URL to local path."""
    console.print(f"[dim]Downloading file from: {file_url}[/dim]")
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.get(file_url)
        response.raise_for_status()
        
        # Create parent directory if not exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(dest_path, 'wb') as f:
            f.write(response.content)
    
    console.print(f"[green]File downloaded to: {dest_path}[/green]")


async def process_single_task(task: Dict[str, Any]) -> int:
    """Process a single task and return exit code.
    
    Returns:
        0 for success, 1 for failure
    """
    task_id = task["task_id"]
    dataset_id = task["dataset_id"]
    doc_id = task["doc_id"]
    mode = task.get("mode", "classic")
    custom_metadata = task.get("metadata")
    api_base_url = config.API_BASE_URL
    
    console.print("\n" + "=" * 80)
    console.print(f"[bold cyan]Processing Task {task_id} (PID: {os.getpid()})[/bold cyan]")
    
    temp_dir = None
    doc_info = None
    
    try:
        # Update to PROCESSING (claim task)
        console.print("[dim]Claiming task...[/dim]")
        task_data = await update_task_status(api_base_url, task_id, TaskStatus.PROCESSING, progress=0)
        
        if not task_data:
            console.print(f"[yellow]Failed to claim task {task_id}[/yellow]")
            return 1
        
        console.print("[green]Task claimed successfully[/green]")
        
        # Get document info
        doc_info = await get_document_info(api_base_url, dataset_id, doc_id)
        
        # Download file
        if "file_url" not in doc_info or not doc_info["file_url"]:
            raise ValueError(f"No file_url available for document {doc_id}")
        
        file_url = doc_info["file_url"]
        temp_dir = Path(tempfile.gettempdir()) / "rag-worker" / task_id
        file_path = temp_dir / doc_info["file_name"]
        
        try:
            await download_file(file_url, file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download file: {e}")
        
        workspace_dir = temp_dir
        
        # Get dataset collection_name
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{api_base_url}/datasets/{dataset_id}")
            response.raise_for_status()
            dataset_info = response.json()["data"]
        
        collection_name = dataset_info["collection_name"]
        
        # Display mapping
        mapping_table = Table(show_header=False, box=None, padding=(0, 2))
        mapping_table.add_column("Item", style="cyan")
        mapping_table.add_column("Arrow", style="white")
        mapping_table.add_column("Target", style="green bold")
        
        mapping_table.add_row("File", "-->", file_path.name)
        mapping_table.add_row("Mode", "-->", mode.upper())
        mapping_table.add_row("Collection", "-->", collection_name)
        mapping_table.add_row("Vector Store", "-->", f"Qdrant @ {config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT}")
        
        console.print("\n")
        console.print(Panel(mapping_table, title="[bold]Data Mapping[/bold]", border_style="magenta"))
        console.print("")
        
        # Process document
        console.print("\n[bold]Starting document processing pipeline...[/bold]")
        
        # Progress callback
        async def update_progress(progress: int):
            """Update task progress, detect cancellation, ignore 409 (task already in terminal state)."""
            try:
                response = await update_task_status(api_base_url, task_id, TaskStatus.PROCESSING, progress=progress)
                
                # Check if task was cancelled
                if response and response.get("status") == TaskStatus.CANCELLED:
                    raise TaskCancelledException(f"Task {task_id} was cancelled by user")
                    
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 409:
                    # Task already completed or failed, ignore progress update
                    console.print(f"[dim]Task in terminal state, skipping progress update[/dim]")
                else:
                    console.print(f"[yellow]Progress update failed ({e.response.status_code}): {e}[/yellow]")
            except TaskCancelledException:
                # Re-raise cancellation exception
                raise
            except Exception as e:
                console.print(f"[yellow]Progress update failed: {e}[/yellow]")
        
        # Route to indexer
        if mode == ProcessingMode.LOD:
            result = await index_lod(
                file_path=file_path,
                collection_name=collection_name,
                custom_metadata=custom_metadata,
                on_progress=update_progress,
                vector_store_grpc_port=config.VECTOR_STORE_GRPC_PORT
            )
        else:  # classic mode
            result = await index_classic(
                file_path=file_path,
                workspace_dir=workspace_dir,
                collection_name=collection_name,
                meilisearch_index_name=collection_name,
                custom_metadata=custom_metadata,
                on_progress=update_progress,
                vector_store_grpc_port=config.VECTOR_STORE_GRPC_PORT
            )
        
        # Complete task
        console.print("[dim]Updating task status to COMPLETED...[/dim]")
        await update_task_status(
            api_base_url,
            task_id,
            TaskStatus.COMPLETED,
            progress=100,
            unit_count=result["unit_count"]
        )
        
        console.print(f"\n[bold green]✓ Task {task_id} completed[/bold green]")
        return 0
    
    except TaskCancelledException as e:
        # Task was cancelled by user
        console.print(f"\n[bold yellow]⚠ Task {task_id} cancelled: {e}[/bold yellow]")
        
        # Task status already updated by API, no need to update again
        # Just clean up and exit
        return 2  # Exit code 2 indicates cancellation
        
    except ProcessingError as e:
        # Structured processing error with error code
        console.print(f"\n[bold red]✗ Task {task_id} failed: {e}[/bold red]")
        
        # Log full error details for debugging
        console.print(f"[dim]Error Code: {e.error_code}[/dim]")
        console.print(f"[dim]Error Type: {e.error_type}[/dim]")
        if e.suggestion:
            console.print(f"[yellow]Suggestion: {e.suggestion}[/yellow]")
        
        await update_task_status(
            api_base_url,
            task_id,
            TaskStatus.FAILED,
            error_message=e.to_dict()  # Structured error
        )
        return 1
        
    except Exception as e:
        # Unstructured error (unexpected)
        console.print(f"\n[bold red]✗ Task {task_id} failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        
        await update_task_status(
            api_base_url,
            task_id,
            TaskStatus.FAILED,
            error_message={
                "error_code": "UNKNOWN_ERROR",
                "error_type": "unknown",
                "message": str(e),
                "exception_type": type(e).__name__
            }
        )
        return 1
    
    finally:
        # Cleanup temporary files
        if doc_info and "file_url" in doc_info and doc_info["file_url"] and temp_dir:
            try:
                if temp_dir.exists():
                    import shutil
                    shutil.rmtree(temp_dir)
                    console.print("[dim]Temporary files cleaned up[/dim]")
            except Exception as e:
                console.print(f"[yellow]Failed to cleanup temp files: {e}[/yellow]")
        
        # Force cleanup (process will exit anyway)
        gc.collect()


async def main():
    """Entry point for subprocess task processor."""
    if len(sys.argv) < 2:
        console.print("[red]Error: task JSON required as argument[/red]")
        sys.exit(1)
    
    import json
    task_json = sys.argv[1]
    task = json.loads(task_json)
    
    exit_code = await process_single_task(task)
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
