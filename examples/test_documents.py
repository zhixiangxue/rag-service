"""Test document upload and task creation."""
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.table import Table
import time

# Load environment variables
load_dotenv()

console = Console()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def test_upload_file(dataset_id: str, file_path: str):
    """Test: POST /datasets/{id}/documents - Upload file"""
    console.print(f"\n[bold cyan]Test 1: Upload File[/bold cyan]")
    console.print(f"File: {file_path}")
    
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/documents",
            files={"file": (Path(file_path).name, f, "application/pdf")}
        )
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        doc_id = data["data"]["doc_id"]
        console.print(f"[green]✓ File uploaded: {doc_id}[/green]")
        return doc_id
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")
        return None


def test_create_task(dataset_id: str, doc_id: str, mode: str = "classic"):
    """Test: POST /datasets/{id}/documents/{doc_id}/tasks - Create processing task"""
    console.print(f"\n[bold cyan]Test 2: Create Task (mode={mode})[/bold cyan]")
    
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}/tasks",
        params={"mode": mode}
    )
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        task_id = data["data"]["task_id"]
        console.print(f"[green]✓ Task created: {task_id}[/green]")
        return task_id
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")
        return None


def test_list_documents(dataset_id: str):
    """Test: GET /datasets/{id}/documents - List documents"""
    console.print(f"\n[bold cyan]Test 3: List Documents[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/documents")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    
    if response.status_code == 200:
        docs = data["data"]
        
        table = Table(title="Documents")
        table.add_column("Doc ID", style="cyan")
        table.add_column("File Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Task ID")
        
        for doc in docs:
            table.add_row(
                doc["doc_id"],
                doc["file_name"],
                doc["status"],
                str(doc.get("task_id", ""))
            )
        
        console.print(table)
        console.print(f"[green]✓ Found {len(docs)} document(s)[/green]")
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")


def test_get_document(dataset_id: str, doc_id: str):
    """Test: GET /datasets/{id}/documents/{doc_id} - Get document details"""
    console.print(f"\n[bold cyan]Test 4: Get Document {doc_id}[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        console.print("[green]✓ Document details retrieved[/green]")
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")


def test_delete_document(dataset_id: str, doc_id: str):
    """Test: DELETE /datasets/{id}/documents/{doc_id} - Delete document"""
    console.print(f"\n[bold cyan]Test 5: Delete Document {doc_id}[/bold cyan]")
    
    response = requests.delete(f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        console.print("[green]✓ Document deleted (including vector store cleanup)[/green]")
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")


if __name__ == "__main__":
    console.print("\n[bold]========================================[/bold]")
    console.print("[bold]  Document API Tests[/bold]")
    console.print("[bold]========================================[/bold]")
    
    # Get inputs
    dataset_id = input("\nEnter dataset_id (or press Enter to create new): ").strip()
    
    if not dataset_id:
        console.print("[yellow]Creating test dataset...[/yellow]")
        response = requests.post(
            f"{BASE_URL}/datasets",
            json={"name": "test_docs", "description": "Test dataset for document tests"}
        )
        if response.status_code == 200:
            dataset_id = response.json()["data"]["dataset_id"]
            console.print(f"[green]Created dataset: {dataset_id}[/green]")
        else:
            console.print("[red]Failed to create dataset[/red]")
            exit(1)
    
    file_path = input("Enter PDF file path: ").strip().strip('"').strip("'")
    
    if not Path(file_path).exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        exit(1)
    
    mode = input("Enter processing mode (classic/lod) [classic]: ").strip() or "classic"
    
    # Test sequence
    doc_id = test_upload_file(dataset_id, file_path)
    
    if doc_id:
        task_id = test_create_task(dataset_id, doc_id, mode)
        
        if task_id:
            console.print(f"\n[yellow]Task {task_id} created. Worker will process it...[/yellow]")
            console.print("[dim]Check task status with test_tasks.py[/dim]")
        
        test_list_documents(dataset_id)
        test_get_document(dataset_id, doc_id)
        
        # Ask before deleting
        console.print("\n[yellow]Press Enter to delete the document (or Ctrl+C to keep it)...[/yellow]")
        input()
        test_delete_document(dataset_id, doc_id)
    
    console.print("\n[bold green]All tests completed![/bold green]\n")
