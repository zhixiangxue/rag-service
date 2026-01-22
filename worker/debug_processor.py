"""
Local debug script for document processing pipeline.

This script allows testing the document processing pipeline without API/DB dependencies.
Simply run it and input the PDF file path when prompted.

Usage:
    cd /Users/zhixiang.xue/zeitro/zag-ai/
    python -m rag-service.worker.debug_processor
    
    # Then input file path (drag & drop supported):
    > /path/to/file.pdf
"""

import asyncio
import os
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from . import config

# Force localhost BEFORE importing processor (processor imports config values at module level)
def force_localhost_config():
    """Force all service endpoints to localhost for local debugging."""
    if config.VECTOR_STORE_HOST != "localhost":
        config.VECTOR_STORE_HOST = "localhost"
    if not config.MEILISEARCH_HOST.startswith("http://localhost"):
        config.MEILISEARCH_HOST = "http://localhost:7700"

force_localhost_config()

# NOW import processor (after config modification)
from .processor import process_document

# ========== Configuration (loaded from .env) ==========
# All configs are read from worker/.env via config module
# Override here if needed for specific debugging:

# Debug-specific config
DEBUG_WORKSPACE_ROOT = Path("./.workspace/debug")  # Fixed workspace for debug runs

# Derived config (not in config.py, only for debug script)
LLM_URI = f"{config.LLM_PROVIDER}/{config.LLM_MODEL}"
VECTOR_STORE_GRPC_PORT = int(os.getenv("VECTOR_STORE_GRPC_PORT", "16334"))
COLLECTION_NAME = "debug_collection"  # Debug-only collection, avoid production data
MEILISEARCH_URL = config.MEILISEARCH_HOST
MEILISEARCH_INDEX_NAME = "debug_index"  # Debug-only index, avoid production data

# =======================================================

console = Console()


def create_temp_workspace(pdf_name: str) -> Path:
    """Create temporary workspace for debugging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = DEBUG_WORKSPACE_ROOT / f"{pdf_name}_{timestamp}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def print_progress_callback(progress: int):
    """Simple progress callback for console output."""
    console.print(f"[dim]Progress: {progress}%[/dim]")


def normalize_path(path_str: str) -> Path:
    """
    Normalize file path by removing quotes and handling drag & drop paths.
    
    Handles:
    - Single quotes: '/path/to/file.pdf'
    - Double quotes: "/path/to/file.pdf"
    - Escaped spaces: /path/to/my\\ file.pdf
    - Plain paths: /path/to/file.pdf
    """
    # Remove leading/trailing whitespace
    path_str = path_str.strip()
    
    # Remove surrounding quotes (single or double)
    if (path_str.startswith("'") and path_str.endswith("'")) or \
       (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1]
    
    # Handle escaped spaces (unescape them)
    path_str = path_str.replace("\\ ", " ")
    
    return Path(path_str)


async def debug_process(pdf_path: Path):
    """Process document without API/DB dependencies."""
    
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold cyan]Document Processor Debug Mode[/bold cyan]\n\n"
        f"PDF: {pdf_path.name}\n"
        f"Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB",
        border_style="cyan"
    ))
    console.print("=" * 70 + "\n")
    
    # Auto-create timestamped workspace
    workspace_dir = create_temp_workspace(pdf_path.stem)
    console.print(f"[cyan]📁 Workspace: {workspace_dir}[/cyan]\n")
    
    # Process document
    start_time = datetime.now()
    
    try:
        result = await process_document(
            file_path=pdf_path,
            workspace_dir=workspace_dir,
            collection_name=COLLECTION_NAME,
            meilisearch_index_name=MEILISEARCH_INDEX_NAME,
            custom_metadata=None,  # No custom metadata in debug mode
            on_progress=print_progress_callback,  # Real-time progress
            vector_store_grpc_port=VECTOR_STORE_GRPC_PORT
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Print summary
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            f"[bold green]✓ Processing Complete[/bold green]\n\n"
            f"Units created: {result['unit_count']}\n"
            f"Parts processed: {result['parts_processed']}\n"
            f"Source hash: {result['source_hash']}\n"
            f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.1f} min)\n\n"
            f"Workspace: {workspace_dir}",
            border_style="green"
        ))
        console.print("=" * 70 + "\n")
        
        return result
        
    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        console.print(f"\n[bold red]✗ Processing failed after {elapsed:.2f}s: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Main entry point - interactive file input."""
    
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold cyan]RAG Document Processor - Debug Mode[/bold cyan]\n\n"
        "Process a PDF file without API/DB dependencies.\n"
        "Please verify configuration before proceeding.",
        border_style="cyan"
    ))
    console.print("=" * 70 + "\n")
    
    # Show config first for user confirmation
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  LLM: {config.LLM_PROVIDER}/{config.LLM_MODEL}")
    console.print(f"  Embedding: {config.EMBEDDING_URI}")
    console.print(f"  Vector Store: {config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT} (gRPC: {VECTOR_STORE_GRPC_PORT})")
    console.print(f"  Meilisearch: {config.MEILISEARCH_HOST}")
    console.print(f"  Collection: {COLLECTION_NAME}")
    console.print(f"  Index: {MEILISEARCH_INDEX_NAME}\n")
    
    # Ask for confirmation
    console.print("[bold yellow]Continue with this configuration? [Y/n]:[/bold yellow] ", end="")
    confirm = input().strip().lower()
    if confirm == 'n':
        console.print("[yellow]Cancelled[/yellow]")
        return 0
    
    # Interactive file input
    try:
        console.print("\n[bold]Enter PDF file path (drag & drop supported):[/bold] ", end="")
        path_input = input().strip()
        
        if not path_input:
            console.print("[red]No file path provided[/red]")
            return 1
        
        # Normalize path (handle quotes and escaped spaces)
        pdf_path = normalize_path(path_input)
        
        # Validate file
        if not pdf_path.exists():
            console.print(f"[red]Error: File not found: {pdf_path}[/red]")
            return 1
        
        if not pdf_path.is_file():
            console.print(f"[red]Error: Not a file: {pdf_path}[/red]")
            return 1
        
        if pdf_path.suffix.lower() != '.pdf':
            console.print(f"[yellow]Warning: File extension is not .pdf: {pdf_path.suffix}[/yellow]")
            console.print("Continue anyway? [y/N]: ", end="")
            confirm = input().strip().lower()
            if confirm != 'y':
                console.print("[yellow]Cancelled[/yellow]")
                return 0
        
        # Process document
        await debug_process(pdf_path)
        return 0
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        return 130
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {e}[/bold red]")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))
