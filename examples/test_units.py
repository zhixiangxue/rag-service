"""Test units API."""
import requests
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Load environment variables
load_dotenv()

console = Console()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def test_get_units(dataset_id: str, limit: int = 10, offset: int = 0):
    """Test: GET /datasets/{id}/units - Get units from dataset"""
    console.print(f"\n[bold cyan]Test: Get Units[/bold cyan]")
    console.print(f"Dataset ID: {dataset_id}")
    console.print(f"Limit: {limit}, Offset: {offset}")
    
    response = requests.get(
        f"{BASE_URL}/datasets/{dataset_id}/units",
        params={"limit": limit, "offset": offset}
    )
    
    console.print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()["data"]
        units = data["units"]
        total = data["total"]
        
        console.print(f"\n[green]✓ Retrieved {len(units)} units (Total: {total})[/green]")
        
        table = Table(title="Units")
        table.add_column("Unit ID", style="cyan", no_wrap=True)
        table.add_column("Type", style="yellow")
        table.add_column("Doc ID", style="magenta")
        table.add_column("Content Preview", style="white")
        
        for unit in units:
            content_preview = unit["content"][:80] + ("..." if len(unit["content"]) > 80 else "")
            table.add_row(
                unit["unit_id"],
                unit.get("unit_type", "text"),
                unit.get("doc_id", "N/A"),
                content_preview
            )
        
        console.print(table)
        
        # Show detailed view of first unit
        if units:
            console.print(f"\n[bold yellow]First Unit Details:[/bold yellow]")
            first_unit = units[0]
            console.print(Panel(
                first_unit["content"][:500] + ("..." if len(first_unit["content"]) > 500 else ""),
                title=f"Unit ID: {first_unit['unit_id']}",
                border_style="cyan"
            ))
            
            if "metadata" in first_unit and first_unit["metadata"]:
                console.print(f"\n[dim]Metadata: {first_unit['metadata']}[/dim]")
    else:
        console.print(f"[red]✗ Failed: {response.json()}[/red]")


if __name__ == "__main__":
    console.print("\n[bold]========================================[/bold]")
    console.print("[bold]  Units API Tests[/bold]")
    console.print("[bold]========================================[/bold]")
    
    # Get inputs
    dataset_id = input("\nEnter dataset_id: ").strip()
    
    if not dataset_id:
        console.print("[red]dataset_id is required[/red]")
        exit(1)
    
    limit = input("Enter limit [10]: ").strip() or "10"
    offset = input("Enter offset [0]: ").strip() or "0"
    
    test_get_units(dataset_id, int(limit), int(offset))
    
    console.print("\n[bold green]Test completed![/bold green]\n")
