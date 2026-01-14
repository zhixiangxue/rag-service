"""
Qdrant Snapshot Dump Script

Export a snapshot of the collection from local Qdrant server.
Run this script on your local machine where Qdrant is running.

Usage:
    python dump.py

Output:
    - Creates a snapshot on Qdrant server
    - Downloads it to ./snapshots/ directory
"""

import os
import requests
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "16333"))
COLLECTION_NAME = "mortgage_guidelines"
SNAPSHOT_DIR = Path(__file__).parent / "output/snapshots"

def create_snapshot():
    """Create a snapshot on Qdrant server"""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/snapshots"
    
    console.print(f"\n[cyan]Creating snapshot for collection: {COLLECTION_NAME}[/cyan]")
    console.print(f"[dim]Qdrant URL: {url}[/dim]\n")
    
    try:
        response = requests.post(url, timeout=300)  # 5 min timeout
        response.raise_for_status()
        
        result = response.json()
        snapshot_name = result["result"]["name"]
        
        console.print(f"[green]✓ Snapshot created: {snapshot_name}[/green]")
        return snapshot_name
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]❌ Failed to create snapshot: {e}[/red]")
        raise


def download_snapshot(snapshot_name: str):
    """Download the snapshot file"""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/snapshots/{snapshot_name}"
    
    # Ensure snapshot directory exists
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    output_file = SNAPSHOT_DIR / snapshot_name
    
    console.print(f"\n[cyan]Downloading snapshot...[/cyan]")
    console.print(f"[dim]Saving to: {output_file}[/dim]\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading...", total=None)
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            if total_size > 0:
                progress.update(task, total=total_size)
            
            # Download with progress
            downloaded = 0
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress.update(task, completed=downloaded)
        
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        console.print(f"[green]✓ Download complete: {file_size_mb:.2f} MB[/green]")
        return output_file
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]❌ Failed to download snapshot: {e}[/red]")
        raise


def get_collection_info():
    """Get collection information"""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        info = response.json()["result"]
        vectors_count = info.get("vectors_count", 0)
        points_count = info.get("points_count", 0)
        
        return vectors_count, points_count
        
    except Exception as e:
        console.print(f"[yellow]⚠ Could not get collection info: {e}[/yellow]")
        return None, None


def main():
    """Main entry point"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]           Qdrant Snapshot Dump Script            [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    
    # Check Qdrant connection
    console.print(f"\n[dim]Checking Qdrant connection at {QDRANT_HOST}:{QDRANT_PORT}...[/dim]")
    try:
        health_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz"
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        console.print("[green]✓ Qdrant is running[/green]")
    except Exception as e:
        console.print(f"[red]❌ Cannot connect to Qdrant: {e}[/red]")
        console.print(f"[yellow]Make sure Qdrant is running at {QDRANT_HOST}:{QDRANT_PORT}[/yellow]")
        return
    
    # Get collection info
    vectors_count, points_count = get_collection_info()
    if vectors_count is not None:
        console.print(f"[cyan]Collection info: {points_count} points, {vectors_count} vectors[/cyan]")
    
    try:
        # Step 1: Create snapshot
        snapshot_name = create_snapshot()
        
        # Step 2: Download snapshot
        output_file = download_snapshot(snapshot_name)
        
        # Summary
        console.print("\n[bold green]═══════════════════════════════════════════════════════════[/bold green]")
        console.print("[bold green]                   Dump Complete!                   [/bold green]")
        console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
        console.print(f"\n[green]Snapshot file: {output_file}[/green]")
        console.print(f"\n[yellow]Next steps:[/yellow]")
        console.print(f"  1. Copy {output_file} to your remote server")
        console.print(f"  2. Run recover.py on the remote server")
        console.print()
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Dump failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
