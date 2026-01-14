"""
Qdrant Snapshot Recover Script

Upload and recover a collection snapshot to remote Qdrant server.
Run this script on your remote server where Qdrant is running.

Usage:
    python recover.py snapshots/mortgage_guidelines-xxx.snapshot

Or if snapshot is in current directory:
    python recover.py mortgage_guidelines-xxx.snapshot
"""

import os
import sys
import requests
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "16333"))
COLLECTION_NAME = "mortgage_guidelines"


def check_collection_exists():
    """Check if collection already exists"""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}"
    
    try:
        response = requests.get(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def create_collection():
    """Create an empty collection for snapshot recovery"""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}"
    
    console.print(f"\n[cyan]Creating collection '{COLLECTION_NAME}'...[/cyan]")
    
    # Create a minimal collection configuration
    # The actual schema will be overwritten by the snapshot
    payload = {
        "vectors": {
            "size": 1536,  # Placeholder, will be replaced by snapshot
            "distance": "Cosine"
        }
    }
    
    try:
        response = requests.put(url, json=payload, timeout=30)
        response.raise_for_status()
        console.print(f"[green]✓ Collection created[/green]")
        return True
    except requests.exceptions.RequestException as e:
        console.print(f"[red]❌ Failed to create collection: {e}[/red]")
        raise


def recover_from_snapshot_direct(snapshot_file: Path):
    """Recover collection directly by uploading snapshot (official API)"""
    # Official Qdrant API: POST /collections/{name}/snapshots/upload?priority=snapshot
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/snapshots/upload?priority=snapshot"
    
    console.print(f"\n[cyan]Uploading snapshot to collection endpoint...[/cyan]")
    console.print(f"[dim]File: {snapshot_file}[/dim]")
    console.print(f"[dim]Size: {snapshot_file.stat().st_size / (1024*1024):.2f} MB[/dim]\n")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Uploading...", total=snapshot_file.stat().st_size)
            
            with open(snapshot_file, 'rb') as f:
                # Progress tracking wrapper
                class ProgressFile:
                    def __init__(self, file, task_id, progress):
                        self.file = file
                        self.task_id = task_id
                        self.progress = progress
                    
                    def read(self, size=-1):
                        chunk = self.file.read(size)
                        if chunk:
                            self.progress.update(self.task_id, advance=len(chunk))
                        return chunk
                    
                    def __getattr__(self, name):
                        return getattr(self.file, name)
                
                progress_file = ProgressFile(f, task, progress)
                
                # Upload snapshot using official API
                response = requests.post(
                    url,
                    files={'snapshot': (snapshot_file.name, progress_file, 'application/octet-stream')},
                    timeout=600
                )
                response.raise_for_status()
                
                result = response.json()
                console.print(f"[green]\u2713 Snapshot uploaded and restored successfully[/green]")
                console.print(f"[dim]Response: {result}[/dim]")
        
        return True
        
    except requests.exceptions.RequestException as e:
        console.print(f"[red]\u274c Failed: {e}[/red]")
        if hasattr(e, 'response') and e.response:
            console.print(f"[dim]Response: {e.response.text}[/dim]")
        raise


def verify_collection():
    """Verify the recovered collection"""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        info = response.json()["result"]
        vectors_count = info.get("vectors_count", 0)
        points_count = info.get("points_count", 0)
        
        console.print(f"\n[green]✓ Collection verified:[/green]")
        console.print(f"  [cyan]Points: {points_count}[/cyan]")
        console.print(f"  [cyan]Vectors: {vectors_count}[/cyan]")
        
        return True
        
    except Exception as e:
        console.print(f"[yellow]⚠ Could not verify collection: {e}[/yellow]")
        return False


def main():
    """Main entry point"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]         Qdrant Snapshot Recover Script            [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    
    # Check arguments
    if len(sys.argv) < 2:
        console.print("\n[red]❌ Error: Snapshot file path required[/red]")
        console.print("\n[yellow]Usage:[/yellow]")
        console.print("  python recover.py snapshots/mortgage_guidelines-xxx.snapshot")
        console.print("  python recover.py mortgage_guidelines-xxx.snapshot")
        return
    
    snapshot_path = Path(sys.argv[1])
    
    # Check if file exists
    if not snapshot_path.exists():
        console.print(f"\n[red]❌ Error: Snapshot file not found: {snapshot_path}[/red]")
        
        # Try to find in snapshots directory
        alt_path = Path("snapshots") / snapshot_path.name
        if alt_path.exists():
            console.print(f"[yellow]Found in snapshots directory, using: {alt_path}[/yellow]")
            snapshot_path = alt_path
        else:
            return
    
    snapshot_name = snapshot_path.name
    
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
    
    # Check if collection exists
    collection_exists = check_collection_exists()
    
    if collection_exists:
        console.print(f"\n[yellow]⚠ Warning: Collection '{COLLECTION_NAME}' already exists[/yellow]")
        console.print(f"[yellow]Recovery will overwrite the existing collection[/yellow]")
        
        confirm = console.input("\n[bold]Continue? (yes/no): [/bold]").strip().lower()
        if confirm not in ['yes', 'y']:
            console.print("\n[yellow]Recovery cancelled[/yellow]")
            return
    else:
        # Create collection if it doesn't exist
        console.print(f"\n[cyan]Collection '{COLLECTION_NAME}' does not exist[/cyan]")
        try:
            create_collection()
        except Exception as e:
            console.print(f"[red]❌ Failed to create collection: {e}[/red]")
            return
    
    try:
        # Step 1: Upload and recover in one go (using global endpoint)
        recover_from_snapshot_direct(snapshot_path)
        
        # Step 3: Verify collection
        verify_collection()
        
        # Summary
        console.print("\n[bold green]═══════════════════════════════════════════════════════════[/bold green]")
        console.print("[bold green]                 Recovery Complete!                 [/bold green]")
        console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
        console.print(f"\n[green]Collection '{COLLECTION_NAME}' is now available on this server[/green]")
        console.print(f"\n[yellow]Next steps:[/yellow]")
        console.print(f"  1. Update your application's QDRANT_HOST to point to this server")
        console.print(f"  2. Test your queries with query.py")
        console.print()
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Recovery failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
