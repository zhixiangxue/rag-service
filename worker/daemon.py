"""RAG Worker - Process document ingestion tasks.

This worker polls the RAG service for PENDING tasks and processes them in
isolated subprocesses to ensure complete GPU memory cleanup after each task.

Supports graceful shutdown: Press Ctrl+C once to finish current task before exit,
press twice to force immediate shutdown.
"""
import time
import asyncio
import httpx
import json
import sys
import subprocess
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from .constants import TaskStatus
from . import config

console = Console()


def check_gpu_memory_usage() -> Optional[float]:
    """Check current GPU memory usage percentage.
    
    Returns:
        Memory usage ratio (0.0-1.0), or None if GPU not available
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        # Get memory stats
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        
        return allocated / total if total > 0 else 0.0
    except ImportError:
        return None
    except Exception as e:
        console.print(f"[yellow]Failed to check GPU memory: {e}[/yellow]")
        return None


def cleanup_gpu_memory():
    """Clean up GPU memory aggressively."""
    try:
        import torch
        import gc
        
        if torch.cuda.is_available():
            # Synchronize CUDA operations
            torch.cuda.synchronize()
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Force garbage collection (multiple times)
            for _ in range(3):
                gc.collect()
            
            # Clear cache again
            torch.cuda.empty_cache()
            
            console.print("[dim]GPU memory cleaned[/dim]")
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[yellow]GPU cleanup warning: {e}[/yellow]")


class RagWorker:
    """RAG document processing worker - task scheduler."""
    
    def __init__(self):
        self.api_base_url = config.API_BASE_URL
        self.poll_interval = config.WORKER_POLL_INTERVAL
        self.gpu_memory_threshold = 0.5  # 50% threshold
        self.running = False
        self.shutdown_requested = False  # Graceful shutdown flag
        self.force_shutdown = False  # Force shutdown flag
        self.current_subprocess: Optional[subprocess.Popen] = None  # Track running subprocess
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
    
    def _handle_shutdown_signal(self, signum: int, frame) -> None:
        """Handle shutdown signals (Ctrl+C, kill).
        
        First signal: Request graceful shutdown (wait for current task)
        Second signal: Force immediate shutdown
        """
        if self.force_shutdown:
            # Already in force shutdown, ignore
            return
        
        if self.shutdown_requested:
            # Second signal - force shutdown
            console.print("\n[bold red]Force shutdown requested![/bold red]")
            self.force_shutdown = True
            
            # Terminate current subprocess if exists
            if self.current_subprocess and self.current_subprocess.poll() is None:
                console.print("[yellow]Terminating current task...[/yellow]")
                self.current_subprocess.terminate()
                try:
                    self.current_subprocess.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    console.print("[red]Task not responding, killing...[/red]")
                    self.current_subprocess.kill()
            
            console.print("[red]Worker terminated[/red]")
            sys.exit(1)
        else:
            # First signal - graceful shutdown
            console.print("\n[bold yellow]Graceful shutdown requested...[/bold yellow]")
            console.print("[dim]Will exit after current task completes (press Ctrl+C again to force)[/dim]")
            self.shutdown_requested = True
    
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
    
    def process_task_in_subprocess(self, task: Dict[str, Any]) -> int:
        """Process a task in an isolated subprocess.
        
        Args:
            task: Task data dictionary
            
        Returns:
            Exit code from subprocess (0 = success, non-zero = failure)
        """
        task_id = task["task_id"]
        console.print(f"\n[cyan]Spawning subprocess for task {task_id}...[/cyan]")
        
        # Serialize task to JSON
        task_json = json.dumps(task)
        
        # Get Python interpreter path (use same venv)
        python_executable = sys.executable
        
        # Build command to run task_processor
        cmd = [
            python_executable,
            "-m",
            "worker.task_processor",
            task_json
        ]
        
        try:
            # Start subprocess with Popen (non-blocking, allows tracking)
            self.current_subprocess = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr
            )
            
            # Wait for subprocess to complete
            exit_code = self.current_subprocess.wait()
            self.current_subprocess = None  # Clear reference
            
            if exit_code == 0:
                console.print(f"[green]Subprocess completed successfully[/green]")
            else:
                console.print(f"[red]Subprocess failed with exit code {exit_code}[/red]")
            
            return exit_code
            
        except Exception as e:
            console.print(f"[red]Failed to spawn subprocess: {e}[/red]")
            self.current_subprocess = None
            return 1
    
    async def run(self):
        """Main worker loop."""
        self.running = True
        
        # Check GPU availability at startup
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                console.print(f"\n[green]GPU detected: {gpu_name}[/green]")
                console.print(f"[green]Total GPU memory: {total_memory:.1f} GB[/green]")
                console.print(f"[yellow]GPU memory threshold: {self.gpu_memory_threshold * 100:.0f}%[/yellow]")
            else:
                console.print("\n[yellow]No GPU detected, using CPU mode[/yellow]")
        except ImportError:
            console.print("\n[yellow]PyTorch not available, GPU detection skipped[/yellow]")
        
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            "[bold cyan]RAG Worker Started (Process Isolation Mode)[/bold cyan]\n"
            f"API: {self.api_base_url}\n"
            f"Poll interval: {self.poll_interval}s\n"
            f"Mode: Subprocess isolation\n"
            f"Shutdown: Ctrl+C once = graceful, twice = force",
            border_style="cyan"
        ))
        console.print("=" * 70 + "\n")
        
        while self.running:
            try:
                # Check if graceful shutdown requested
                if self.shutdown_requested:
                    if self.current_subprocess and self.current_subprocess.poll() is None:
                        # Task still running, wait
                        console.print("[dim]Waiting for current task to finish...[/dim]")
                        await asyncio.sleep(1)
                        continue
                    else:
                        # No running task, safe to exit
                        console.print("[green]No active tasks, shutting down gracefully[/green]")
                        break
                
                console.print(f"[dim]Polling tasks... ({datetime.now().strftime('%H:%M:%S')})[/dim]")
                
                # Fetch pending tasks
                tasks = await self.get_pending_tasks()
                
                if tasks:
                    console.print(f"[cyan]Found {len(tasks)} pending task(s)[/cyan]")
                    
                    # Check GPU memory before accepting task
                    gpu_usage = check_gpu_memory_usage()
                    if gpu_usage is not None:
                        console.print(f"[dim]GPU memory usage: {gpu_usage * 100:.1f}%[/dim]")
                        
                        if gpu_usage > self.gpu_memory_threshold:
                            console.print(
                                f"[yellow]GPU memory usage ({gpu_usage * 100:.1f}%) > "
                                f"threshold ({self.gpu_memory_threshold * 100:.0f}%), "
                                f"skipping task and cleaning...[/yellow]"
                            )
                            cleanup_gpu_memory()
                            await asyncio.sleep(self.poll_interval)
                            continue
                    
                    # Process tasks in subprocess
                    for task in tasks:
                        # Check shutdown flag before starting new task
                        if self.shutdown_requested:
                            console.print("[yellow]Shutdown requested, skipping new tasks[/yellow]")
                            break
                        
                        exit_code = self.process_task_in_subprocess(task)
                        
                        # Cleanup after subprocess (defensive)
                        cleanup_gpu_memory()
                else:
                    console.print("[dim]No pending tasks, waiting...[/dim]")
                
                # Wait before next poll
                await asyncio.sleep(self.poll_interval)
                
            except KeyboardInterrupt:
                # Handled by signal handler, but keep for safety
                if not self.shutdown_requested:
                    self.shutdown_requested = True
                    console.print("\n[yellow]Graceful shutdown requested...[/yellow]")
            except Exception as e:
                console.print(f"\n[red]Worker error: {e}[/red]")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(self.poll_interval)
        
        console.print("\n[bold]Worker shutdown complete[/bold]")


async def main():
    """Worker entry point."""
    worker = RagWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
