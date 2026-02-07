"""End-to-end workflow: create dataset, upload doc, create tasks."""
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
import time

load_dotenv()

console = Console()

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = os.getenv("API_PORT", "8000")
BASE_URL = f"http://{API_HOST}:{API_PORT}"


def create_or_select_dataset():
    """Step 1: Create or select dataset."""
    console.print("\n[bold cyan]Step 1: Dataset Setup[/bold cyan]")
    
    # List existing datasets
    response = requests.get(f"{BASE_URL}/datasets")
    if response.status_code == 200:
        datasets = response.json()["data"]
        if datasets:
            console.print(f"\n[yellow]Found {len(datasets)} existing dataset(s):[/yellow]")
            for idx, ds in enumerate(datasets, 1):
                console.print(f"  {idx}. {ds['name']} (ID: {ds['dataset_id']})")
    
    # Ask user choice
    choice = Prompt.ask(
        "\nCreate new dataset or use existing?",
        choices=["new", "existing"],
        default="new"
    )
    
    if choice == "existing" and datasets:
        idx = int(Prompt.ask("Select dataset number", default="1")) - 1
        dataset_id = datasets[idx]["dataset_id"]
        dataset_name = datasets[idx]["name"]
        console.print(f"[green]✓ Using dataset: {dataset_name} ({dataset_id})[/green]")
        return dataset_id, dataset_name
    
    # Create new dataset
    dataset_name = Prompt.ask("Enter dataset name", default="test_dataset")
    description = Prompt.ask("Enter description (optional)", default="E2E test dataset")
    
    response = requests.post(
        f"{BASE_URL}/datasets",
        json={"name": dataset_name, "description": description}
    )
    
    if response.status_code == 200:
        data = response.json()["data"]
        dataset_id = data["dataset_id"]
        console.print(f"[green]✓ Dataset created: {dataset_name} ({dataset_id})[/green]")
        return dataset_id, dataset_name
    else:
        console.print(f"[red]✗ Failed to create dataset: {response.text}[/red]")
        exit(1)


def upload_document(dataset_id):
    """Step 2: Upload document."""
    console.print("\n[bold cyan]Step 2: Upload Document[/bold cyan]")
    
    file_path = Prompt.ask("Enter PDF file path").strip().strip('"').strip("'")
    
    if not Path(file_path).exists():
        console.print(f"[red]✗ File not found: {file_path}[/red]")
        exit(1)
    
    console.print(f"Uploading: {file_path}")
    
    with open(file_path, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/documents",
            files={"file": (Path(file_path).name, f, "application/pdf")}
        )
    
    if response.status_code == 200:
        data = response.json()["data"]
        doc_id = data["doc_id"]
        file_name = data["file_name"]
        console.print(f"[green]✓ File uploaded: {file_name} ({doc_id})[/green]")
        return doc_id, file_name
    else:
        console.print(f"[red]✗ Upload failed: {response.text}[/red]")
        exit(1)


def create_tasks(dataset_id, doc_id):
    """Step 3: Create processing tasks."""
    console.print("\n[bold cyan]Step 3: Create Processing Tasks[/bold cyan]")
    
    mode_choice = Prompt.ask(
        "Select processing mode",
        choices=["classic", "lod", "both"],
        default="both"
    )
    
    task_ids = []
    
    if mode_choice in ["classic", "both"]:
        console.print("\n[yellow]Creating Classic mode task...[/yellow]")
        response = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}/tasks",
            params={"mode": "classic"}
        )
        if response.status_code == 200:
            task_id = response.json()["data"]["task_id"]
            console.print(f"[green]✓ Classic task created: {task_id}[/green]")
            task_ids.append(("classic", task_id))
        else:
            console.print(f"[red]✗ Failed: {response.text}[/red]")
    
    if mode_choice in ["lod", "both"]:
        console.print("\n[yellow]Creating LOD mode task...[/yellow]")
        response = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}/tasks",
            params={"mode": "lod"}
        )
        if response.status_code == 200:
            task_id = response.json()["data"]["task_id"]
            console.print(f"[green]✓ LOD task created: {task_id}[/green]")
            task_ids.append(("lod", task_id))
        else:
            console.print(f"[red]✗ Failed: {response.text}[/red]")
    
    return task_ids


def monitor_tasks(dataset_id, task_ids, auto_monitor=True):
    """Step 4: Monitor task progress."""
    if not auto_monitor:
        return
    
    console.print("\n[bold cyan]Step 4: Monitor Tasks[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop monitoring[/dim]\n")
    
    try:
        while True:
            all_done = True
            
            for mode, task_id in task_ids:
                response = requests.get(f"{BASE_URL}/tasks/{task_id}")
                if response.status_code == 200:
                    task = response.json()["data"]
                    status = task["status"]
                    progress = task.get("progress", 0)
                    
                    status_color = {
                        "PENDING": "yellow",
                        "PROCESSING": "blue",
                        "COMPLETED": "green",
                        "FAILED": "red"
                    }.get(status, "white")
                    
                    console.print(
                        f"[{status_color}]{mode.upper():8s}[/{status_color}] "
                        f"Task {task_id}: {status:12s} [{progress:3d}%]"
                    )
                    
                    if status not in ["COMPLETED", "FAILED"]:
                        all_done = False
            
            if all_done:
                console.print("\n[green]✓ All tasks completed![/green]")
                break
            
            time.sleep(2)
            console.print()
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped[/yellow]")


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]End-to-End Workflow[/bold]\n"
        "Create dataset → Upload document → Create tasks → Monitor",
        border_style="cyan"
    ))
    
    # Step 1: Dataset
    dataset_id, dataset_name = create_or_select_dataset()
    
    # Step 2: Upload
    doc_id, file_name = upload_document(dataset_id)
    
    # Step 3: Create tasks
    task_ids = create_tasks(dataset_id, doc_id)
    
    # Summary
    console.print("\n" + "="*60)
    console.print(Panel(
        f"[green]Tasks Created Successfully![/green]\n\n"
        f"Dataset: {dataset_name} ({dataset_id})\n"
        f"Document: {file_name} ({doc_id})\n"
        f"Tasks: {len(task_ids)} task(s) created\n\n"
        f"[yellow]Next:[/yellow] Start worker to process tasks:\n"
        f"  python -m rag-service.worker.daemon",
        title="Summary",
        border_style="green"
    ))
    
    # Ask if monitor
    if Confirm.ask("\nMonitor task progress now?", default=True):
        monitor_tasks(dataset_id, task_ids, auto_monitor=True)
    else:
        console.print("\n[dim]You can monitor tasks later with test_tasks.py[/dim]")
