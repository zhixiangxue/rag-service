"""
Batch retry all FAILED tasks (concurrent mode).
Usage: 
  - Dry run (view errors): python retry_failed_tasks.py --dry-run
  - Retry tasks: python retry_failed_tasks.py --concurrency 10
"""
import asyncio
import aiohttp
import argparse
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()


async def retry_single_task(session: aiohttp.ClientSession, api_url: str, task_id: str) -> dict:
    """Retry a single task."""
    try:
        async with session.post(
            f"{api_url}/tasks/{task_id}/retry",
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            response.raise_for_status()
            return {"task_id": task_id, "status": "success"}
    except Exception as e:
        return {"task_id": task_id, "status": "error", "error": str(e)}


async def analyze_failed_tasks(api_url: str):
    """Analyze FAILED tasks and display error messages."""
    console.print("\n[cyan]Fetching FAILED tasks...[/cyan]")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{api_url}/tasks",
            params={"status": "FAILED", "limit": 10000}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            failed_tasks = data.get("data", [])
    
    if not failed_tasks:
        console.print("[green]No FAILED tasks found![/green]\n")
        return
    
    console.print(f"[yellow]Found {len(failed_tasks)} FAILED tasks[/yellow]\n")
    console.print("=" * 80)
    
    # Print all error messages
    for i, task in enumerate(failed_tasks, 1):
        task_id = task["task_id"]
        error_msg = task.get("error_message")
        
        console.print(f"\n[bold cyan]{i}. Task: {task_id}[/bold cyan]")
        console.print(error_msg)
    
    console.print("\n" + "=" * 80)
    console.print(f"\n[bold]Total: {len(failed_tasks)} failed tasks[/bold]\n")


async def retry_failed_tasks(api_url: str, concurrency: int):
    """Retry all FAILED tasks with concurrent requests."""
    # Step 1: Get all FAILED tasks
    console.print("\n[cyan]Step 1: Fetching FAILED tasks...[/cyan]")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{api_url}/tasks",
            params={"status": "FAILED", "limit": 10000}
        ) as response:
            response.raise_for_status()
            data = await response.json()
            failed_tasks = data.get("data", [])
    
    if not failed_tasks:
        console.print("[green]No FAILED tasks found![/green]\n")
        return
    
    console.print(f"[yellow]Found {len(failed_tasks)} FAILED tasks[/yellow]")
    console.print(f"[cyan]Concurrency: {concurrency} workers[/cyan]\n")
    
    # Step 2: Retry tasks concurrently
    console.print("[cyan]Step 2: Retrying FAILED tasks...[/cyan]\n")
    
    success_count = 0
    error_count = 0
    errors = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Retrying tasks...", total=len(failed_tasks))
        
        async with aiohttp.ClientSession() as session:
            # Use semaphore to control concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def retry_with_semaphore(task_id: str):
                async with semaphore:
                    result = await retry_single_task(session, api_url, task_id)
                    progress.update(task, advance=1)
                    return result
            
            # Create all tasks
            retry_tasks = [
                retry_with_semaphore(failed_task["task_id"])
                for failed_task in failed_tasks
            ]
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*retry_tasks)
    
    # Count results
    for result in results:
        if result["status"] == "success":
            success_count += 1
        else:
            error_count += 1
            errors.append({"task_id": result["task_id"], "error": result.get("error", "Unknown error")})
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold]Retry Summary[/bold]")
    console.print("=" * 60)
    console.print(f"Total FAILED tasks: {len(failed_tasks)}")
    console.print(f"[green]Successfully retried: {success_count}[/green]")
    console.print(f"[red]Failed to retry: {error_count}[/red]")
    
    if errors:
        console.print("\n[red]Errors:[/red]")
        for err in errors[:10]:  # Show first 10 errors
            console.print(f"  - Task {err['task_id']}: {err['error']}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more errors")
    
    console.print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Batch retry all FAILED tasks (concurrent)")
    parser.add_argument(
        "--api-url",
        default="http://13.56.109.233:8000",
        help="API base URL (default: http://13.56.109.233:8000)"
    )
    parser.add_argument(
        "--concurrency", "-c",
        type=int,
        default=10,
        help="Number of concurrent workers (default: 10)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze errors only, do not retry tasks"
    )
    
    args = parser.parse_args()
    
    try:
        if args.dry_run:
            # Dry run mode: only analyze errors
            asyncio.run(analyze_failed_tasks(args.api_url))
        else:
            # Retry mode: retry all failed tasks
            asyncio.run(retry_failed_tasks(args.api_url, args.concurrency))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]\n")
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    main()
