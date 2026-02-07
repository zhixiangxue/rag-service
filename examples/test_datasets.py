"""Test dataset CRUD operations."""
import requests
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Load environment variables
load_dotenv()

console = Console()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def test_create_dataset():
    """Test: POST /datasets - Create dataset"""
    console.print("\n[bold cyan]Test 1: Create Dataset[/bold cyan]")
    
    response = requests.post(
        f"{BASE_URL}/datasets",
        json={
            "name": "test_dataset",
            "description": "Test dataset for API testing"
        }
    )
    
    console.print(f"Status: {response.status_code}")
    console.print(f"Response text: {response.text}")
    
    if response.status_code != 200:
        console.print(f"[red]✗ Failed: {response.text}[/red]")
        return None
    
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        dataset_id = data["data"]["dataset_id"]
        console.print(f"[green]✓ Dataset created: {dataset_id}[/green]")
        return dataset_id
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")
        return None


def test_list_datasets():
    """Test: GET /datasets - List all datasets"""
    console.print("\n[bold cyan]Test 2: List Datasets[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/datasets")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    
    if response.status_code == 200:
        datasets = data["data"]
        
        table = Table(title="Datasets")
        table.add_column("ID", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Collection", style="yellow")
        table.add_column("Description")
        
        for ds in datasets:
            table.add_row(
                ds["dataset_id"],
                ds["name"],
                ds["collection_name"],
                ds.get("description", "")
            )
        
        console.print(table)
        console.print(f"[green]✓ Found {len(datasets)} dataset(s)[/green]")
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")


def test_get_dataset(dataset_id: str):
    """Test: GET /datasets/{id} - Get dataset details"""
    console.print(f"\n[bold cyan]Test 3: Get Dataset {dataset_id}[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        console.print("[green]✓ Dataset details retrieved[/green]")
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")


def test_delete_dataset(dataset_id: str):
    """Test: DELETE /datasets/{id} - Delete dataset"""
    console.print(f"\n[bold cyan]Test 4: Delete Dataset {dataset_id}[/bold cyan]")
    
    response = requests.delete(f"{BASE_URL}/datasets/{dataset_id}")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        console.print("[green]✓ Dataset deleted[/green]")
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")


if __name__ == "__main__":
    console.print("\n[bold]========================================[/bold]")
    console.print("[bold]  Dataset API Tests[/bold]")
    console.print("[bold]========================================[/bold]")
    
    # Test sequence
    dataset_id = test_create_dataset()
    
    if dataset_id:
        test_list_datasets()
        test_get_dataset(dataset_id)
        
        # Ask before deleting
        console.print("\n[yellow]Press Enter to delete the test dataset (or Ctrl+C to keep it)...[/yellow]")
        input()
        test_delete_dataset(dataset_id)
    
    console.print("\n[bold green]All tests completed![/bold green]\n")
