"""
Local debug script for LOD indexing.

This script allows testing the LOD indexing pipeline locally.

Usage:
    python -m rag-service.worker.debug.debug_lod
"""

import asyncio
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel

from .. import config

# Force localhost
def force_localhost_config():
    """Force all service endpoints to localhost for local debugging."""
    if config.VECTOR_STORE_HOST != "localhost":
        config.VECTOR_STORE_HOST = "localhost"

force_localhost_config()

from ..indexers.lod import index_lod

# ========== Configuration ==========
COLLECTION_NAME = "debug_lod_collection"

console = Console()


def normalize_path(path_str: str) -> Path:
    """Normalize file path."""
    path_str = path_str.strip()
    
    if path_str.startswith('& '):
        path_str = path_str[2:].strip()
    
    if (path_str.startswith("'") and path_str.endswith("'")) or \
       (path_str.startswith('"') and path_str.endswith('"')):
        path_str = path_str[1:-1]
    
    path_str = path_str.replace("\\ ", " ")
    
    return Path(path_str)


async def debug_process(pdf_path: Path):
    """Process document using LOD indexing."""
    
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold cyan]LOD Indexing - Debug Mode[/bold cyan]\n\n"
        f"PDF: {pdf_path.name}\n"
        f"Size: {pdf_path.stat().st_size / 1024 / 1024:.2f} MB",
        border_style="cyan"
    ))
    console.print("=" * 70 + "\n")
    
    start_time = datetime.now()
    
    try:
        result = await index_lod(
            file_path=pdf_path,
            collection_name=COLLECTION_NAME,
            custom_metadata=None,
            vector_store_grpc_port=config.VECTOR_STORE_GRPC_PORT
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        console.print("\n" + "=" * 70)
        console.print(Panel.fit(
            f"[bold green]✓ LOD Indexing Complete[/bold green]\n\n"
            f"LOD Unit ID: {result['lod_unit_id']}\n"
            f"Views: {result['views_count']}\n"
            f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.1f} min)",
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
    """Main entry point."""
    
    console.print("\n" + "=" * 70)
    console.print(Panel.fit(
        "[bold cyan]LOD Indexing - Debug Mode[/bold cyan]\n\n"
        "Index entire PDF with multi-resolution views (LOD).\n"
        "Please verify configuration before proceeding.",
        border_style="cyan"
    ))
    console.print("=" * 70 + "\n")
    
    # Show config
    console.print("[bold]Configuration:[/bold]")
    console.print(f"  Vector Store: {config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT}")
    console.print(f"  Collection: {COLLECTION_NAME}\n")
    
    # Ask for confirmation
    console.print("[bold yellow]Continue? [Y/n]:[/bold yellow] ", end="")
    confirm = input().strip().lower()
    if confirm == 'n':
        console.print("[yellow]Cancelled[/yellow]")
        return 0
    
    try:
        # Input file path
        console.print("\n[bold]Enter PDF file path (drag & drop supported):[/bold] ", end="")
        path_input = input().strip()
        if not path_input:
            console.print("[red]No file path provided[/red]")
            return 1
        
        pdf_path = normalize_path(path_input)
        
        if not pdf_path.exists():
            console.print(f"[red]Error: File not found: {pdf_path}[/red]")
            return 1
        
        if not pdf_path.is_file():
            console.print(f"[red]Error: Not a file: {pdf_path}[/red]")
            return 1
        
        # Process
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
