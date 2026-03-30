"""RAG Worker - Dramatiq actor for document ingestion tasks.

Each task runs in an isolated subprocess to ensure complete GPU memory cleanup.

Start worker:
    python -m dramatiq worker.main        (from rag-service/)
"""
import csv
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import dramatiq
import httpx
from dramatiq.brokers.redis import RedisBroker
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .constants import TaskStatus
from . import config

console = Console(force_terminal=True)

# ── Broker ─────────────────────────────────────────────────────────────────────

broker = RedisBroker(host=config.REDIS_HOST, port=config.REDIS_PORT)
dramatiq.set_broker(broker)


# ── Startup: config summary + connectivity checks ───────────────────────────

def _print_worker_config() -> None:
    """Print worker configuration summary on startup."""
    print("\n" + "=" * 60)
    print("  RAG Worker Configuration Summary")
    print("=" * 60)
    print(f"\n[Redis / Broker]")
    print(f"  REDIS_HOST: {config.REDIS_HOST}")
    print(f"  REDIS_PORT: {config.REDIS_PORT}")
    print(f"\n[Vector Store]")
    print(f"  VECTOR_STORE_HOST: {config.VECTOR_STORE_HOST}")
    print(f"  VECTOR_STORE_PORT: {config.VECTOR_STORE_PORT}")
    print(f"\n[Meilisearch]")
    print(f"  MEILISEARCH_HOST: {config.MEILISEARCH_HOST}")
    print(f"  MEILISEARCH_API_KEY: {'set' if config.MEILISEARCH_API_KEY else 'not set'}")
    print(f"\n[Embedding / LLM]")
    print(f"  EMBEDDING_URI: {config.EMBEDDING_URI}")
    print(f"  LLM_PROVIDER: {config.LLM_PROVIDER}")
    print(f"  LLM_MODEL: {config.LLM_MODEL}")
    print(f"\n[Processing]")
    print(f"  USE_GPU: {config.USE_GPU}")
    print(f"  NUM_THREADS: {config.NUM_THREADS}")
    print(f"  MAX_CHUNK_TOKENS: {config.MAX_CHUNK_TOKENS}")
    print("=" * 60)


def _check_worker_dependencies() -> None:
    """Check Redis, Qdrant, Meilisearch, embedding, and GPU. Fail-fast: stops on first failure."""
    import requests as _requests

    def _check_redis():
        try:
            import redis as _redis
            r = _redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            r.ping()
            return True, f"[Redis] Reachable: {config.REDIS_HOST}:{config.REDIS_PORT}"
        except Exception as exc:
            return False, f"[Redis] Cannot connect to {config.REDIS_HOST}:{config.REDIS_PORT}: {exc}"

    def _check_qdrant():
        qdrant_url = f"http://{config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT}"
        try:
            resp = _requests.get(f"{qdrant_url}/healthz", timeout=5)
            resp.raise_for_status()
            return True, f"[Qdrant] Reachable: {qdrant_url}"
        except Exception as exc:
            return False, f"[Qdrant] Cannot connect to {qdrant_url}: {exc}"

    def _check_meilisearch():
        if not config.MEILISEARCH_HOST:
            return False, "[Meilisearch] MEILISEARCH_HOST is not set"
        try:
            resp = _requests.get(f"{config.MEILISEARCH_HOST.rstrip('/')}/health", timeout=5)
            resp.raise_for_status()
            return True, f"[Meilisearch] Reachable: {config.MEILISEARCH_HOST}"
        except Exception as exc:
            return False, f"[Meilisearch] Cannot connect to {config.MEILISEARCH_HOST}: {exc}"

    def _check_embedding():
        embedding_uri = config.EMBEDDING_URI
        if embedding_uri != "openai/text-embedding-3-small":
            return False, f"[Embedding] EMBEDDING_URI must be 'openai/text-embedding-3-small', got '{embedding_uri}'"
        if config.OPENAI_API_KEY:
            return True, f"[Embedding] OPENAI_API_KEY set, model: {embedding_uri}"
        return False, "[Embedding] OPENAI_API_KEY is not set"

    def _check_gpu():
        if not config.USE_GPU:
            return True, "[GPU] USE_GPU=false, skipping"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                return True, f"[GPU] CUDA available: {gpu_name} ({total_mem:.1f} GB)"
            return False, "[GPU] USE_GPU=true but CUDA is not available"
        except ImportError:
            return False, "[GPU] USE_GPU=true but torch is not installed"

    checks = [
        ("[Redis]",       _check_redis),
        ("[Qdrant]",      _check_qdrant),
        ("[Meilisearch]", _check_meilisearch),
        ("[Embedding]",   _check_embedding),
        ("[GPU]",         _check_gpu),
    ]

    print("[Worker] Checking dependencies...")
    for i, (label, fn) in enumerate(checks):
        ok, msg = fn()
        if ok:
            print(f"  ✅ {msg}")
        else:
            print(f"  ❌ {msg}")
            for skipped_label, _ in checks[i + 1:]:
                print(f"  ⌛ {skipped_label} (not checked)")
            print("[Worker] Dependency check failed. Exiting.")
            sys.exit(1)

    print("[Worker] All dependencies OK. Starting worker...\n")


_print_worker_config()
_check_worker_dependencies()


# ── GPU helpers (preserved from daemon.py) ─────────────────────────────────────

def check_gpu_memory_usage() -> Optional[float]:
    """Return GPU memory usage ratio (0.0-1.0), or None if unavailable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return allocated / total if total > 0 else 0.0
    except ImportError:
        return None
    except Exception as e:
        console.print(f"[yellow]Failed to check GPU memory: {e}[/yellow]")
        return None


def cleanup_gpu_memory():
    """Aggressively free GPU memory after subprocess exits."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            for _ in range(3):
                gc.collect()
            torch.cuda.empty_cache()
            console.print("[dim]GPU memory cleaned[/dim]")
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[yellow]GPU cleanup warning: {e}[/yellow]")


# ── CSV logging (preserved from daemon.py) ─────────────────────────────────────

def _log_task_result(
    task_id: str,
    dataset_id: str,
    doc_id: str,
    file_name: str,
    mode: str,
    status: str,
    start_time: datetime,
    end_time: datetime,
    exit_code: int,
    error_message: str = "",
):
    """Append task result to ~/.zag/worker/tasks.csv."""
    log_dir = Path.home() / ".zag" / "worker"
    log_file = log_dir / "tasks.csv"
    fields = [
        "task_id", "dataset_id", "doc_id", "file_name", "mode",
        "status", "start_time", "end_time", "duration_seconds", "exit_code", "error_message",
    ]
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        duration = (end_time - start_time).total_seconds()
        file_exists = log_file.exists()
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "task_id": task_id,
                "dataset_id": dataset_id,
                "doc_id": doc_id,
                "file_name": file_name or "",
                "mode": mode,
                "status": status,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": f"{duration:.2f}",
                "exit_code": exit_code,
                "error_message": error_message or "",
            })
        console.print(f"[dim]Task result logged to {log_file}[/dim]")
    except Exception as e:
        console.print(f"[yellow]Failed to log task result: {e}[/yellow]")


# ── API helpers ────────────────────────────────────────────────────────────────

def _fetch_task(task_id: str) -> dict:
    """GET /tasks/{task_id} — returns parsed data dict. Raises on error."""
    url = f"{config.API_BASE_URL}/tasks/{task_id}"
    resp = httpx.get(url, headers=config.API_HEADERS, timeout=30.0)
    resp.raise_for_status()
    return resp.json()["data"]


def _fetch_doc_metadata(dataset_id: str, doc_id: str) -> Optional[dict]:
    """GET /datasets/{dataset_id}/documents/{doc_id} — returns metadata dict or None."""
    try:
        url = f"{config.API_BASE_URL}/datasets/{dataset_id}/documents/{doc_id}"
        resp = httpx.get(url, headers=config.API_HEADERS, timeout=30.0)
        resp.raise_for_status()
        return resp.json()["data"].get("metadata")
    except Exception as e:
        console.print(f"[yellow]Could not fetch doc metadata: {e}[/yellow]")
        return None


def _patch_task_status(task_id: str, status: str, worker: str = None):
    """PATCH /tasks/{task_id} to set status (and optionally worker). Swallows errors."""
    try:
        url = f"{config.API_BASE_URL}/tasks/{task_id}"
        body = {"status": status}
        if worker is not None:
            body["worker"] = worker
        httpx.patch(url, json=body, headers=config.API_HEADERS, timeout=30.0)
    except Exception as e:
        console.print(f"[yellow]Failed to patch task status: {e}[/yellow]")


def _get_worker() -> str:
    """Return 'hostname (ip)' string. Never raises — returns 'unknown' on any failure."""
    try:
        import socket
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        return f"{hostname} ({ip})"
    except Exception:
        return "unknown"


# ── Dramatiq actor ─────────────────────────────────────────────────────────────

@dramatiq.actor(max_retries=3, min_backoff=5000, time_limit=float("inf"))
def process_document(task_id: str):
    """
    Receive a task_id, fetch task details from the API, then run task_processor
    in an isolated subprocess.

    Exit code contract (same as daemon.py):
        0  → success  (task_processor already updated status to COMPLETED)
        2  → cancelled by user, skip silently
        3  → permanent failure (e.g. page limit), no retry
        other → transient failure, raise → Dramatiq retries up to max_retries
    """
    console.print(f"\n[cyan]process_document received task_id={task_id}[/cyan]")

    # ── 1. Fetch task details ──────────────────────────────────────────────────
    try:
        task_data = _fetch_task(task_id)
    except Exception as e:
        console.print(f"[red]Failed to fetch task {task_id}: {e}[/red]")
        raise RuntimeError(f"Could not fetch task details: {e}") from e

    dataset_id = task_data["dataset_id"]
    doc_id = task_data["doc_id"]
    mode = task_data.get("mode", "classic")
    reader = task_data.get("reader", "mineru")
    status = task_data.get("status", "")

    # ── 2. Skip if already cancelled ──────────────────────────────────────────
    if status == TaskStatus.CANCELLED:
        console.print(f"[yellow]Task {task_id} is CANCELLED, skipping[/yellow]")
        return

    # ── 3. Fetch document metadata (needed by indexing pipeline) ──────────────
    metadata = _fetch_doc_metadata(dataset_id, doc_id)

    # ── 4. Mark task as PROCESSING and record worker identity ─────────────────
    worker = _get_worker()
    _patch_task_status(task_id, TaskStatus.PROCESSING, worker=worker)
    console.print(f"[dim]Worker identity: {worker}[/dim]")

    # ── 5. Check GPU memory ───────────────────────────────────────────────────
    gpu_usage = check_gpu_memory_usage()
    if gpu_usage is not None:
        console.print(f"[dim]GPU memory before subprocess: {gpu_usage * 100:.1f}%[/dim]")

    # ── 6. Build task JSON (same schema task_processor expects) ───────────────
    task_json = json.dumps({
        "task_id": task_id,
        "dataset_id": dataset_id,
        "doc_id": doc_id,
        "mode": mode,
        "reader": reader,
        "metadata": metadata,
    })

    # ── 7. Spawn isolated subprocess ──────────────────────────────────────────
    cmd = [sys.executable, "-m", "worker.task_processor", task_json]
    console.print(f"[cyan]Spawning subprocess for task {task_id}...[/cyan]")

    start_time = datetime.now()
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, stdout=None, stderr=None, env=env)

    # ── 8. Wait for subprocess, polling for cancel every 5 seconds ───────────
    CANCEL_POLL_INTERVAL = 5  # seconds
    while proc.poll() is None:
        time.sleep(CANCEL_POLL_INTERVAL)
        try:
            task_status = _fetch_task(task_id).get("status")
            if task_status == TaskStatus.CANCELLED:
                console.print(f"[yellow]Cancel detected — killing subprocess for task {task_id}[/yellow]")
                proc.kill()
                proc.wait()  # reap zombie
                break
        except Exception as e:
            console.print(f"[dim]Cancel check failed (ignored): {e}[/dim]")

    exit_code = proc.returncode if proc.returncode is not None else proc.wait()
    end_time = datetime.now()

    console.print(f"[dim]Subprocess exited with code {exit_code}[/dim]")

    # ── 8. Map exit code to status ────────────────────────────────────────────
    status_map = {0: "COMPLETED", 1: "FAILED", 2: "CANCELLED", 3: "FAILED", -9: "CANCELLED"}
    final_status = status_map.get(exit_code, "FAILED")

    # ── 9. GPU cleanup ────────────────────────────────────────────────────────
    cleanup_gpu_memory()

    # ── 10. CSV log ───────────────────────────────────────────────────────────
    _log_task_result(
        task_id=task_id,
        dataset_id=dataset_id,
        doc_id=doc_id,
        file_name="",          # not critical; task_processor already logged to API
        mode=mode,
        status=final_status,
        start_time=start_time,
        end_time=end_time,
        exit_code=exit_code,
    )

    # ── 11. Handle exit codes ─────────────────────────────────────────────────
    duration = (end_time - start_time).total_seconds()

    status_labels = {
        0: ("COMPLETED", "green"),
        2: ("CANCELLED", "yellow"),
        -9: ("CANCELLED", "yellow"),
        3: ("FAILED (permanent)", "yellow"),
    }
    label, color = status_labels.get(exit_code, ("FAILED", "red"))

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column()
    t.add_row("Status", f"[bold {color}]{label}[/bold {color}]")
    t.add_row("Task", task_id)
    t.add_row("Mode", mode.upper())
    t.add_row("Duration", f"{duration:.1f}s")
    console.print(Panel(t, border_style=color, expand=False))

    if exit_code not in status_labels:
        # Transient failure — raise so Dramatiq retries
        raise RuntimeError(
            f"Task {task_id} subprocess exited with code {exit_code} (transient failure)"
        )
