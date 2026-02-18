"""RAG Worker - Process document ingestion tasks.

This worker polls the RAG service for PENDING tasks and processes them using
the document processing pipeline (prepare_manifest + import).
"""
import time
import asyncio
import httpx
import json
import gc
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential

from .constants import TaskStatus, ProcessingMode
from . import config
from .indexers.classic import index_classic
from .indexers.lod import index_lod

console = Console()


def cleanup_gpu_memory():
    """Clean up GPU memory to prevent memory leaks."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory stats before cleanup
            allocated_before = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Get memory stats after cleanup
            allocated_after = torch.cuda.memory_allocated() / 1024**2  # MB
            freed = allocated_before - allocated_after
            
            if freed > 0:
                console.print(f"[dim]GPU memory freed: {freed:.1f} MB[/dim]")
    except ImportError:
        pass  # torch not available
    except Exception as e:
        console.print(f"[yellow]GPU cleanup warning: {e}[/yellow]")


class RagWorker:
    """RAG document processing worker."""
    
    def __init__(self):
        self.api_base_url = config.API_BASE_URL
        self.poll_interval = config.WORKER_POLL_INTERVAL
        self.running = False
    
    async def get_pending_tasks(self) -> list:
        """Fetch pending tasks from API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base_url}/tasks/pending",
                params={"limit": 1}  # Process one at a time
            )
            response.raise_for_status()
            data = response.json()
            return data["data"]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
        reraise=True
    )
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[int] = None,
        error_message: Optional[Dict[str, Any]] = None,
        unit_count: Optional[int] = None
    ) -> bool:
        """Update task status via API with retry.
        
        Returns:
            True if update succeeded, False otherwise
        """
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
                    f"{self.api_base_url}/tasks/{task_id}",
                    json=payload
                )
                response.raise_for_status()
                return True
        except Exception as e:
            console.print(f"[red]Failed to update task status: {e}[/red]")
            return False
    
    async def get_document_info(self, dataset_id: str, doc_id: str) -> Dict[str, Any]:
        """Get document information from API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base_url}/datasets/{dataset_id}/documents/{doc_id}"
            )
            response.raise_for_status()
            return response.json()["data"]
    
    async def download_file(self, file_url: str, dest_path: Path) -> None:
        """Download file from HTTP URL to local path.
        
        Args:
            file_url: HTTP URL to download from
            dest_path: Local destination path
        """
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
    
    async def process_task(self, task: Dict[str, Any]):
        """Process a single task."""
        task_id = task["task_id"]
        dataset_id = task["dataset_id"]
        doc_id = task["doc_id"]
        mode = task.get("mode", "classic")  # Default to classic if not specified
        custom_metadata = task.get("metadata")  # Document metadata from upload
        
        console.print("\n" + "=" * 80)
        console.print(f"[bold cyan]Processing Task {task_id}[/bold cyan]")
        
        temp_dir = None
        doc_info = None
        
        try:
            # Update to PROCESSING (claim task)
            console.print("[dim]Claiming task...[/dim]")
            claimed = await self.update_task_status(task_id, TaskStatus.PROCESSING, progress=0)
            
            if not claimed:
                console.print(f"[yellow]Failed to claim task {task_id}, skipping...[/yellow]")
                return
            
            console.print("[green]Task claimed successfully[/green]")
            
            # Get document info (file_url for download)
            doc_info = await self.get_document_info(dataset_id, doc_id)
            
            # Download file to temporary directory
            if "file_url" not in doc_info or not doc_info["file_url"]:
                raise ValueError(f"No file_url available for document {doc_id}")
            
            file_url = doc_info["file_url"]
            # Use system temporary directory (cross-platform)
            temp_dir = Path(tempfile.gettempdir()) / "rag-worker" / task_id
            file_path = temp_dir / doc_info["file_name"]
            
            try:
                await self.download_file(file_url, file_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download file: {e}")
            
            workspace_dir = temp_dir
            
            # Get dataset info to retrieve collection_name
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_base_url}/datasets/{dataset_id}"
                )
                response.raise_for_status()
                dataset_info = response.json()["data"]
            
            collection_name = dataset_info["collection_name"]
            
            # 使用 Table 显示映射关系
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
            
            # Process document using integrated pipeline
            console.print("\n[bold]Starting document processing pipeline...[/bold]")
            
            # 定义进度回调函数（async 版本，串行等待）
            async def update_progress(progress: int):
                """更新任务进度到 API（async 版本，串行执行）."""
                try:
                    await self.update_task_status(task_id, TaskStatus.PROCESSING, progress=progress)
                except Exception as e:
                    console.print(f"[yellow]Progress update failed: {e}[/yellow]")
            
            # Route to different indexer based on mode
            if mode == ProcessingMode.LOD:
                result = await index_lod(
                    file_path=file_path,
                    collection_name=collection_name,
                    custom_metadata=custom_metadata,
                    on_progress=update_progress,
                    vector_store_grpc_port=config.VECTOR_STORE_GRPC_PORT
                )
            else:  # classic mode (default)
                result = await index_classic(
                    file_path=file_path,
                    workspace_dir=workspace_dir,
                    collection_name=collection_name,  # Use dataset's collection_name
                    meilisearch_index_name=collection_name,  # Use same name for index
                    custom_metadata=custom_metadata,
                    on_progress=update_progress,  # 传递 async 回调
                    vector_store_grpc_port=config.VECTOR_STORE_GRPC_PORT
                )
            
            # Complete task with actual unit count
            console.print("[dim]Updating task status to COMPLETED...[/dim]")
            await self.update_task_status(
                task_id,
                TaskStatus.COMPLETED,
                progress=100,
                unit_count=result["unit_count"]
            )
            
            console.print(f"\n[bold green]✓ Task {task_id} completed[/bold green]")
            
        except Exception as e:
            console.print(f"\n[bold red]✗ Task {task_id} failed: {e}[/bold red]")
            import traceback
            traceback.print_exc()
            
            await self.update_task_status(
                task_id,
                TaskStatus.FAILED,
                error_message={"error": str(e), "type": type(e).__name__}
            )
        
        finally:
            # Cleanup GPU memory (always executed)
            console.print("[dim]Cleaning up GPU memory...[/dim]")
            cleanup_gpu_memory()
            
            # Cleanup temporary files if downloaded
            if doc_info and "file_url" in doc_info and doc_info["file_url"] and temp_dir:
                try:
                    if temp_dir.exists():
                        import shutil
                        shutil.rmtree(temp_dir)
                        console.print("[dim]Temporary files cleaned up[/dim]")
                except Exception as e:
                    console.print(f"[yellow]Failed to cleanup temp files: {e}[/yellow]")
    
    async def run(self):
        """Main worker loop."""
        self.running = True
        
        # Check GPU availability and memory at startup
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                console.print(f"\n[green]GPU detected: {gpu_name}[/green]")
                console.print(f"[green]Total GPU memory: {total_memory:.1f} GB[/green]")
            else:
                console.print("\n[yellow]No GPU detected, using CPU mode[/yellow]")
        except ImportError:
            console.print("\n[yellow]PyTorch not available, GPU detection skipped[/yellow]")
        
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            "[bold cyan]RAG Worker Started[/bold cyan]\n"
            f"API: {self.api_base_url}\n"
            f"Poll interval: {self.poll_interval}s",
            border_style="cyan"
        ))
        console.print("=" * 70 + "\n")
        
        while self.running:
            try:
                console.print(f"[dim]Polling tasks... ({datetime.now().strftime('%H:%M:%S')})[/dim]")
                
                # Fetch pending tasks
                tasks = await self.get_pending_tasks()
                
                if tasks:
                    console.print(f"[cyan]Found {len(tasks)} pending task(s)[/cyan]")
                    for task in tasks:
                        await self.process_task(task)
                else:
                    console.print("[dim]No pending tasks, waiting...[/dim]")
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Worker stopped by user[/yellow]")
                self.running = False
                break
            except Exception as e:
                console.print(f"\n[red]Worker error: {e}[/red]")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.poll_interval)
        
        console.print("\n[bold]Worker shutdown[/bold]")


async def main():
    """Worker entry point."""
    worker = RagWorker()
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted[/yellow]")
