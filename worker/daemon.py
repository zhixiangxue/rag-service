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
        self.max_retries = 3  # Maximum retry attempts per task
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
    
    async def claim_task(self) -> Optional[Dict[str, Any]]:
        """Claim a pending task from API (atomic operation).
        
        Returns:
            Task data if available, None if no tasks or claim failed
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(f"{self.api_base_url}/tasks/claim")
                response.raise_for_status()
                data = response.json()
                return data["data"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    # No pending tasks (normal, not an error)
                    return None
                else:
                    # Real error (network issue, server error, etc.)
                    console.print(f"[red]⚠️  API error {e.response.status_code}: {e.response.text[:100]}[/red]")
                    return None
            except httpx.RequestError as e:
                # Network error (connection refused, timeout, etc.)
                console.print(f"[red]⚠️  Network error: {type(e).__name__} - {str(e)[:100]}[/red]")
                return None
            except Exception as e:
                console.print(f"[red]⚠️  Unexpected error: {type(e).__name__} - {str(e)[:100]}[/red]")
                return None
    
    def process_task_in_subprocess(self, task: Dict[str, Any]) -> int:
        """Process a task in an isolated subprocess with automatic retry.
        
        If the subprocess fails, automatically retries on the same machine
        to leverage checkpoint recovery. This ensures that partial progress
        (e.g., completed parts in large file processing) is preserved.
        
        Args:
            task: Task data dictionary
            
        Returns:
            Exit code from subprocess (0 = success, non-zero = failure after all retries)
        """
        task_id = task["task_id"]
        
        for attempt in range(1, self.max_retries + 1):
            if attempt > 1:
                console.print(f"\n[yellow]🔄 Retry attempt {attempt}/{self.max_retries} for task {task_id}[/yellow]")
                console.print(f"[dim]Checkpoint recovery will restore previous progress...[/dim]")
            else:
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
                    return 0
                elif exit_code == 2:
                    # Task was cancelled by user, don't retry
                    console.print(f"[yellow]Task cancelled by user, skipping retry[/yellow]")
                    return 2
                else:
                    # Task failed
                    console.print(f"[red]Subprocess failed with exit code {exit_code}[/red]")
                    
                    if attempt < self.max_retries:
                        console.print(f"[yellow]Will retry in 5 seconds (attempt {attempt}/{self.max_retries})...[/yellow]")
                        import time
                        time.sleep(5)  # Wait before retry
                        # Continue to next attempt
                    else:
                        console.print(f"[red]All {self.max_retries} retry attempts exhausted, task failed[/red]")
                        return exit_code
                
            except Exception as e:
                console.print(f"[red]Failed to spawn subprocess: {e}[/red]")
                self.current_subprocess = None
                
                if attempt < self.max_retries:
                    console.print(f"[yellow]Will retry in 5 seconds (attempt {attempt}/{self.max_retries})...[/yellow]")
                    import time
                    time.sleep(5)
                else:
                    console.print(f"[red]All {self.max_retries} retry attempts exhausted[/red]")
                    return 1
        
        # Should never reach here
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
            f"Max retries: {self.max_retries}\n"
            f"Mode: Subprocess isolation with checkpoint recovery\n"
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
                
                # Claim a pending task (atomic operation)
                task = await self.claim_task()
                
                if task:
                    console.print(f"[cyan]Claimed task {task['task_id']}[/cyan]")
                    
                    # Check GPU memory before processing
                    gpu_usage = check_gpu_memory_usage()
                    if gpu_usage is not None:
                        console.print(f"[dim]GPU memory usage: {gpu_usage * 100:.1f}%[/dim]")
                        
                        if gpu_usage > self.gpu_memory_threshold:
                            console.print(
                                f"[yellow]GPU memory usage ({gpu_usage * 100:.1f}%) > "
                                f"threshold ({self.gpu_memory_threshold * 100:.0f}%), "
                                f"cleaning before processing...[/yellow]"
                            )
                            cleanup_gpu_memory()
                    
                    # Check shutdown flag before starting
                    if self.shutdown_requested:
                        console.print("[yellow]Shutdown requested, skipping task[/yellow]")
                    else:
                        # Process task in subprocess
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
    def print_config_summary():
        """Print configuration summary before startup."""
        print("\n" + "=" * 60)
        print("  RAG Worker Configuration Summary")
        print("=" * 60)
        print(f"\n[API Server]")
        print(f"  API_HOST: {config.API_HOST}")
        print(f"  API_PORT: {config.API_PORT}")
        print(f"  API_BASE_URL: {config.API_BASE_URL}")
        print(f"\n[Worker Settings]")
        print(f"  WORKER_POLL_INTERVAL: {config.WORKER_POLL_INTERVAL}s")
        print(f"  WORKER_MAX_RETRIES: {config.WORKER_MAX_RETRIES}")
        print(f"\n[Vector Store]")
        print(f"  VECTOR_STORE_TYPE: {config.VECTOR_STORE_TYPE}")
        print(f"  VECTOR_STORE_HOST: {config.VECTOR_STORE_HOST}")
        print(f"  VECTOR_STORE_PORT: {config.VECTOR_STORE_PORT}")
        print(f"  VECTOR_STORE_GRPC_PORT: {config.VECTOR_STORE_GRPC_PORT}")
        print(f"\n[Meilisearch]")
        print(f"  MEILISEARCH_HOST: {config.MEILISEARCH_HOST}")
        print(f"  MEILISEARCH_API_KEY: {'✅ set' if config.MEILISEARCH_API_KEY else '❌ not set'}")
        print(f"\n[Embedding]")
        print(f"  EMBEDDING_URI: {config.EMBEDDING_URI}")
        print(f"  OPENAI_API_KEY: {'✅ set' if config.OPENAI_API_KEY else '❌ not set'}")
        print(f"\n[LLM]")
        print(f"  LLM_PROVIDER: {config.LLM_PROVIDER}")
        print(f"  LLM_MODEL: {config.LLM_MODEL}")
        print(f"\n[Document Processing]")
        print(f"  DOCUMENT_READER: {config.DOCUMENT_READER}")
        print(f"  MAX_PAGES_PER_PART: {config.MAX_PAGES_PER_PART}")
        print(f"  USE_GPU: {config.USE_GPU}")
        print(f"  NUM_THREADS: {config.NUM_THREADS}")
        print(f"  MAX_CHUNK_TOKENS: {config.MAX_CHUNK_TOKENS}")
        print(f"  TARGET_TOKEN_SIZE: {config.TARGET_TOKEN_SIZE}")
        print("=" * 60)

    def confirm_config() -> bool:
        """Interactive confirmation of configuration before startup."""
        print_config_summary()

        if not sys.stdin.isatty():
            print("\n[Worker] Running in non-interactive mode, skipping confirmation.")
            return True

        try:
            response = input("\n[Worker] Do you want to proceed with this configuration? [Y/n]: ").strip().lower()
            if response in ('', 'y', 'yes'):
                print("[Worker] Configuration confirmed. Starting worker...\n")
                return True
            else:
                print("[Worker] Configuration rejected. Exiting.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n[Worker] Interrupted. Exiting.")
            return False

    def validate_services():
        """Validate critical services are reachable before starting."""
        import requests
        from openai import OpenAI

        errors = []

        # Validate OpenAI API Key
        if config.OPENAI_API_KEY:
            try:
                client = OpenAI(api_key=config.OPENAI_API_KEY)
                client.models.list()
            except Exception as e:
                errors.append(f"OpenAI API Key invalid: {str(e)}")
        else:
            errors.append("OPENAI_API_KEY is not set")

        # Validate Vector Store (Qdrant)
        try:
            response = requests.get(
                f"http://{config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT}/collections",
                timeout=2
            )
            if response.status_code != 200:
                errors.append(f"Vector Store unreachable: HTTP {response.status_code}")
        except Exception as e:
            errors.append(f"Vector Store unreachable: {str(e)}")

        # Validate Meilisearch
        try:
            response = requests.get(f"{config.MEILISEARCH_HOST}/health", timeout=2)
            if response.status_code != 200:
                errors.append(f"Meilisearch unreachable: HTTP {response.status_code}")
        except Exception as e:
            errors.append(f"Meilisearch unreachable: {str(e)}")

        if errors:
            console.print("\n[bold red]Service Validation Failed:[/bold red]")
            for error in errors:
                console.print(f"  ✗ {error}")
            console.print("\n[yellow]Please check your .env file and ensure all services are running.[/yellow]\n")
            raise RuntimeError(f"Service validation failed: {len(errors)} error(s)")

    if not confirm_config():
        sys.exit(1)

    validate_services()

    worker = RagWorker()
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
