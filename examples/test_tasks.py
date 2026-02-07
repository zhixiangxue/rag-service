"""Test task status queries."""
import requests
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import time

# Load environment variables
load_dotenv()

console = Console()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def test_list_tasks(dataset_id: str):
    """Test: GET /datasets/{id}/tasks - List tasks for dataset"""
    console.print(f"\n[bold cyan]Test 1: List Tasks for Dataset {dataset_id}[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/tasks")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    
    if response.status_code == 200:
        tasks = data["data"]
        
        table = Table(title="Tasks")
        table.add_column("Task ID", style="cyan")
        table.add_column("Doc ID", style="yellow")
        table.add_column("Mode", style="magenta")
        table.add_column("Status", style="green")
        table.add_column("Progress", style="blue")
        
        for task in tasks:
            table.add_row(
                str(task["task_id"]),
                task["doc_id"],
                task.get("mode", "classic"),
                task["status"],
                f"{task.get('progress', 0)}%"
            )
        
        console.print(table)
        console.print(f"[green]✓ Found {len(tasks)} task(s)[/green]")
        return tasks
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")
        return []


def test_get_task(task_id: str):
    """Test: GET /tasks/{task_id} - Get task details"""
    console.print(f"\n[bold cyan]Test 2: Get Task {task_id}[/bold cyan]")
    
    response = requests.get(f"{BASE_URL}/tasks/{task_id}")
    
    console.print(f"Status: {response.status_code}")
    data = response.json()
    console.print(data)
    
    if response.status_code == 200:
        task = data["data"]
        console.print(f"[green]✓ Task {task_id}: {task['status']} ({task.get('progress', 0)}%)[/green]")
        return task
    else:
        console.print(f"[red]✗ Failed: {data}[/red]")
        return None


def monitor_task(task_id: str, interval: int = 2):
    """Monitor task progress until completion"""
    console.print(f"\n[bold cyan]Monitoring Task {task_id}[/bold cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        task_progress = progress.add_task(f"Task {task_id}", total=100)
        
        while True:
            response = requests.get(f"{BASE_URL}/tasks/{task_id}")
            
            if response.status_code == 200:
                task_data = response.json()["data"]
                status = task_data["status"]
                prog = task_data.get("progress", 0)
                
                progress.update(task_progress, completed=prog, description=f"Task {task_id} - {status}")
                
                if status in ["completed", "failed"]:
                    if status == "completed":
                        console.print(f"[green]✓ Task completed successfully[/green]")
                        if "unit_count" in task_data:
                            console.print(f"  Units indexed: {task_data['unit_count']}")
                    else:
                        console.print(f"[red]✗ Task failed[/red]")
                        if "error_message" in task_data:
                            console.print(f"  Error: {task_data['error_message']}")
                    break
                
                time.sleep(interval)
            else:
                console.print(f"[red]✗ Failed to get task status[/red]")
                break


if __name__ == "__main__":
    console.print("\n[bold]========================================[/bold]")
    console.print("[bold]  Task API Tests[/bold]")
    console.print("[bold]========================================[/bold]")
    
    # Get inputs
    dataset_id = input("\nEnter dataset_id: ").strip()
    
    if not dataset_id:
        console.print("[red]dataset_id is required[/red]")
        exit(1)
    
    # List all tasks
    tasks = test_list_tasks(dataset_id)
    
    if tasks:
        task_id = input("\nEnter task_id to monitor (or press Enter to skip): ").strip()
        
        if task_id:
            test_get_task(task_id)
            
            should_monitor = input("Monitor task progress? (y/n) [y]: ").strip().lower()
            if should_monitor != 'n':
                monitor_task(task_id)
    
    console.print("\n[bold green]All tests completed![/bold green]\n")
