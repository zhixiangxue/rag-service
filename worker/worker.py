"""RAG Worker - Process document ingestion tasks.

This worker polls the RAG service for PENDING tasks and processes them using
the document processing pipeline (prepare_manifest + import).
"""
import time
import asyncio
import httpx
import json
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..app.constants import TaskStatus
from . import config
from .processor import process_document

console = Console()


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
    
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[int] = None,
        error_message: Optional[Dict[str, Any]] = None,
        unit_count: Optional[int] = None
    ):
        """Update task status via API."""
        payload = {"status": status}
        if progress is not None:
            payload["progress"] = progress
        if error_message is not None:
            payload["error_message"] = error_message
        if unit_count is not None:
            payload["unit_count"] = unit_count
        
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.api_base_url}/tasks/{task_id}",  # 修改：去掉 /status
                json=payload
            )
            response.raise_for_status()
    
    async def get_document_info(self, dataset_id: str, doc_id: str) -> Dict[str, Any]:
        """Get document information from API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.api_base_url}/datasets/{dataset_id}/documents/{doc_id}"
            )
            response.raise_for_status()
            return response.json()["data"]
    
    async def process_task(self, task: Dict[str, Any]):
        """Process a single task."""
        task_id = task["task_id"]
        dataset_id = task["dataset_id"]
        doc_id = task["doc_id"]
        
        console.print("\n" + "=" * 80)
        console.print(f"[bold cyan]Processing Task {task_id}[/bold cyan]")
        
        try:
            # Update to PROCESSING
            await self.update_task_status(task_id, TaskStatus.PROCESSING, progress=0)
            
            # Get document info (file path, workspace dir)
            doc_info = await self.get_document_info(dataset_id, doc_id)
            file_path = Path(doc_info["file_path"])
            workspace_dir = Path(doc_info["workspace_dir"])
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
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
            
            result = await process_document(
                file_path=file_path,
                workspace_dir=workspace_dir,
                collection_name=collection_name,  # Use dataset's collection_name
                meilisearch_index_name=collection_name,  # Use same name for index
                custom_metadata=None,  # TODO: Load from doc_info if available
                on_progress=update_progress,  # 传递 async 回调
                vector_store_grpc_port=config.VECTOR_STORE_GRPC_PORT
            )
            
            # Complete task with actual unit count
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
    
    async def run(self):
        """Main worker loop."""
        self.running = True
        
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
