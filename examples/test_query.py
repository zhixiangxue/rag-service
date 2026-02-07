"""Test query APIs."""
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


def test_vector_query(dataset_id: str, query: str, top_k: int = 5):
    """Test: POST /datasets/{dataset_id}/query - Vector query"""
    console.print(f"\n[bold cyan]Test 1: Vector Query[/bold cyan]")
    console.print(f"Query: {query}")
    console.print(f"Top K: {top_k}")
    
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/query",
        json={
            "query": query,
            "top_k": top_k
        }
    )
    
    console.print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()["data"]
        
        console.print(f"\n[green]✓ Found {len(data)} results[/green]")
        
        for idx, result in enumerate(data, 1):
            console.print(f"\n[bold yellow]Result {idx}[/bold yellow]")
            console.print(Panel(
                result['content'][:500] + ("..." if len(result['content']) > 500 else ""),
                title=f"Score: {result['score']:.4f}",
                border_style="green"
            ))
            console.print(f"  Unit ID: {result['unit_id']}")
            console.print(f"  Doc ID: {result.get('doc_id', 'N/A')}")
    else:
        console.print(f"[red]✗ Failed: {response.json()}[/red]")


def test_web_query(dataset_id: str, query: str):
    """Test: POST /query/web - Complete RAG pipeline with LLM"""
    console.print(f"\n[bold cyan]Test: Web Query (RAG Pipeline)[/bold cyan]")
    console.print(f"Query: {query}")
    
    response = requests.post(
        f"{BASE_URL}/query/web",
        json={
            "dataset_id": dataset_id,
            "query": query,
            "top_k": 5
        }
    )
    
    console.print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        console.print(f"\n[bold yellow]Retrieved Results ({len(data['results'])} items):[/bold yellow]")
        for idx, result in enumerate(data['results'], 1):
            console.print(f"\n[dim]Result {idx} (Score: {result['score']:.4f})[/dim]")
            console.print(result['content'][:300] + "...")
            if result.get('analysis'):
                console.print(f"  Relevant: {result['analysis']['is_relevant']}")
                console.print(f"  Confidence: {result['analysis']['confidence']}")
    else:
        console.print(f"[red]✗ Failed: {response.json()}[/red]")


def test_tree_simple(dataset_id: str, query: str, unit_id: str, max_depth: int = 3):
    """Test: POST /datasets/{dataset_id}/query/tree/simple - Tree query with SimpleRetriever"""
    console.print(f"\n[bold cyan]Test: Tree Query (Simple)[/bold cyan]")
    console.print(f"Query: {query}")
    console.print(f"Unit ID: {unit_id}")
    console.print(f"Max Depth: {max_depth}")
    
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/query/tree/simple",
        json={
            "query": query,
            "unit_id": unit_id,
            "max_depth": max_depth
        }
    )
    
    console.print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()["data"]
        
        console.print(f"\n[green]✓ Retrieved {len(data['nodes'])} nodes[/green]")
        console.print(f"Path: {' → '.join(data['path'])}")
        
        for idx, node in enumerate(data['nodes'], 1):
            console.print(f"\n[bold yellow]Node {idx}[/bold yellow]")
            console.print(f"  Node ID: {node.get('node_id', 'N/A')}")
            console.print(f"  Title: {node.get('title', 'N/A')}")
            
            # Print full content
            content = node.get('content', '')
            if content:
                console.print(f"\n  Content:")
                console.print(Panel(content, border_style="blue", expand=False))
            
            # Print summary if exists
            summary = node.get('summary', '')
            if summary:
                console.print(f"\n  Summary:")
                console.print(Panel(summary, border_style="green", expand=False))
            
            # Print children info
            children = node.get('children', [])
            if children:
                console.print(f"\n  Children: {len(children)} nodes")
    else:
        console.print(f"[red]✗ Failed: {response.json()}[/red]")


def test_tree_mcts(dataset_id: str, query: str, unit_id: str, preset: str = "balanced"):
    """Test: POST /datasets/{dataset_id}/query/tree/mcts - Tree query with MCTSRetriever"""
    console.print(f"\n[bold cyan]Test: Tree Query (MCTS)[/bold cyan]")
    console.print(f"Query: {query}")
    console.print(f"Unit ID: {unit_id}")
    console.print(f"Preset: {preset}")
    
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/query/tree/mcts",
        json={
            "query": query,
            "unit_id": unit_id,
            "preset": preset
        }
    )
    
    console.print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()["data"]
        
        console.print(f"\n[green]✓ Retrieved {len(data['nodes'])} nodes[/green]")
        console.print(f"Path: {' → '.join(data['path'])}")
        
        for idx, node in enumerate(data['nodes'], 1):
            console.print(f"\n[bold yellow]Node {idx}[/bold yellow]")
            console.print(f"  Node ID: {node.get('node_id', 'N/A')}")
            console.print(f"  Title: {node.get('title', 'N/A')}")
            
            # Print full content
            content = node.get('content', '')
            if content:
                console.print(f"\n  Content:")
                console.print(Panel(content, border_style="blue", expand=False))
            
            # Print summary if exists
            summary = node.get('summary', '')
            if summary:
                console.print(f"\n  Summary:")
                console.print(Panel(summary, border_style="green", expand=False))
            
            # Print children info
            children = node.get('children', [])
            if children:
                console.print(f"\n  Children: {len(children)} nodes")
    else:
        console.print(f"[red]✗ Failed: {response.json()}[/red]")


if __name__ == "__main__":
    console.print("\n[bold]========================================[/bold]")
    console.print("[bold]  Query API Tests[/bold]")
    console.print("[bold]========================================[/bold]")
    
    # Get inputs
    dataset_id = input("\nEnter dataset_id: ").strip()
    
    if not dataset_id:
        console.print("[red]dataset_id is required[/red]")
        exit(1)
    
    query = input("Enter query: ").strip()
    
    if not query:
        console.print("[red]query is required[/red]")
        exit(1)
    
    # Select test type
    console.print("\n[bold]Select test type:[/bold]")
    console.print("  1. Vector Query (fast, no LLM)")
    console.print("  2. Web Query (slow, with LLM analysis)")
    console.print("  3. Tree Query - Simple (requires unit_id)")
    console.print("  4. Tree Query - MCTS (requires unit_id)")
    
    test_type = input("\nEnter choice (1-4) [1]: ").strip() or "1"
    
    if test_type == "1":
        test_vector_query(dataset_id, query)
    elif test_type == "2":
        console.print("\n[yellow]Web query uses LLM (requires API key and costs money)[/yellow]")
        test_web_query(dataset_id, query)
    elif test_type in ["3", "4"]:
        unit_id = input("\nEnter unit_id: ").strip()
        if not unit_id:
            console.print("[red]unit_id is required for tree queries[/red]")
            exit(1)
        
        if test_type == "3":
            max_depth = input("Enter max_depth [3]: ").strip() or "3"
            test_tree_simple(dataset_id, query, unit_id, int(max_depth))
        else:
            preset = input("Enter preset (fast/balanced/thorough) [balanced]: ").strip() or "balanced"
            test_tree_mcts(dataset_id, query, unit_id, preset)
    else:
        console.print("[red]Invalid choice[/red]")
        exit(1)
    
    console.print("\n[bold green]Test completed![/bold green]\n")
