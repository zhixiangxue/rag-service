"""
Graph Indexer - Index mortgage documents into graph database.

This module provides graph-based indexing functionality for mortgage program documents.
It extracts program structure using LLM and stores the result in a graph database.
"""
import time
from pathlib import Path
from rich.console import Console
from typing import Dict, Any, Optional

from .extractor import MortgageProgramExtractor
from domain.mortgage.graph import FullExtractionResult
from worker.constants import ProcessingMode
from worker.config import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
    FALKORDB_HOST, FALKORDB_PORT,
)
from worker.utils.manifest import prepare_single_file_manifest

console = Console()


async def index_graph(
    file_path: Path,
    workspace_dir: Path,
    graph_name: str = "mortgage_programs",
    custom_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Graph-based indexing: Extract program structure and store in graph database.
    
    Args:
        file_path: Path to the document file (PDF, MD, etc.)
        workspace_dir: Working directory for this file
        graph_name: Name of the graph in the database
        custom_metadata: Optional business metadata
        on_progress: Optional progress callback function(progress: int)
    
    Returns:
        Processing result with extraction stats
    """
    console.print(f"\n[bold cyan]🚀 Graph Indexing[/bold cyan]")
    console.print(f"   File: {file_path}")
    console.print(f"   Workspace: {workspace_dir}")
    console.print(f"   Graph: {graph_name}\n")
    
    start_time = time.time()
    
    # Helper function to report progress
    async def report_progress(progress: int):
        if on_progress:
            try:
                import inspect
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(progress)
                else:
                    on_progress(progress)
            except Exception as e:
                console.print(f"[yellow]⚠ Progress callback error: {e}[/yellow]")
    
    # Step 1: Prepare manifest (10-15%)
    await report_progress(10)
    
    # Inject mode into custom_metadata
    metadata_with_mode = {**(custom_metadata or {}), "mode": ProcessingMode.GRAPH}
    
    manifest = await prepare_single_file_manifest(
        file_path=file_path,
        workspace_dir=workspace_dir,
        custom_metadata=metadata_with_mode
    )
    await report_progress(15)
    
    task = manifest["tasks"][0]
    source_hash = task["source_hash"]
    parts = task["parts"]
    metadata = task["metadata"]
    
    # Create output directory using source hash
    output_dir = workspace_dir / "output" / source_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_products = 0
    total_rules = 0
    total_matrices = 0
    success_count = 0
    error_count = 0
    
    # Step 2: Process each part
    for part_idx, part_info in enumerate(parts, 1):
        part_file = Path(part_info['file'])
        page_range = part_info['page_range']
        
        # Calculate progress range for this part (15% - 95% divided by parts)
        part_progress_start = 15 + int((part_idx - 1) * 80 / len(parts))
        part_progress_end = 15 + int(part_idx * 80 / len(parts))
        
        console.print(f"\n[cyan]--- Part {part_idx}/{len(parts)} ---[/cyan]")
        console.print(f"📄 File: {part_file.name}")
        console.print(f"📚 Pages: {page_range}")
        
        if not part_file.exists():
            console.print(f"[red]❌ Part file not found: {part_file}[/red]")
            error_count += 1
            continue
        
        try:
            # Step 2.1: Read document (0-20% of part range)
            await report_progress(part_progress_start)
            console.print(f"[dim]  Step 1: Reading document...[/dim]")
            
            # Read document content
            content = await _read_document(part_file)
            console.print(f"  ✅ Document loaded: {len(content):,} characters")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.2))
            
            # Step 2.2: Extract program structure (20-70% of part range)
            console.print(f"[dim]  Step 2: Extracting program structure...[/dim]")
            llm_uri = f"{LLM_PROVIDER}/{LLM_MODEL}"
            
            extractor = MortgageProgramExtractor(
                llm_uri=llm_uri,
                api_key=LLM_API_KEY,
            )
            
            result = await extractor.extract(
                document_content=content,
                source_file=str(part_file),
            )
            console.print(f"  ✅ Extraction complete: {len(result.stages_completed)} stages")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.7))
            
            # Step 2.3: Store in graph database (70-100% of part range)
            console.print(f"[dim]  Step 3: Storing in graph database...[/dim]")
            
            # Import graph storage
            from zag.storages.graph import FalkorDBGraphStorage
            from .storage import MortgageGraphStorage
            
            base_storage = FalkorDBGraphStorage(
                host=FALKORDB_HOST,
                port=FALKORDB_PORT,
                graph_name=graph_name,
            )
            
            with base_storage:
                # Use mortgage-specific storage
                mortgage_storage = MortgageGraphStorage(base_storage)
                program_id = mortgage_storage.store_program(result.to_graph_data())
                console.print(f"  ✅ Stored program: {program_id}")
            
            total_products += len(result.products)
            total_rules += len(result.rules)
            total_matrices += len(result.matrices)
            success_count += 1
            
            console.print(f"[green]✓ Part {part_idx} completed[/green]")
            console.print(f"    Products: {len(result.products)}, Rules: {len(result.rules)}, Matrices: {len(result.matrices)}")
            await report_progress(part_progress_end)
            
        except Exception as e:
            console.print(f"[red]✗ Part {part_idx} failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    elapsed = time.time() - start_time
    
    # Report near completion
    await report_progress(95)
    
    # Summary
    console.print(f"\n[bold green]📊 Processing Summary[/bold green]")
    console.print(f"  Total parts: {len(parts)}")
    console.print(f"  Successful: {success_count}")
    console.print(f"  Failed: {error_count}")
    console.print(f"  Total products: {total_products}")
    console.print(f"  Total rules: {total_rules}")
    console.print(f"  Total matrices: {total_matrices}")
    console.print(f"  Time: {elapsed:.2f}s\n")
    
    if error_count > 0:
        raise Exception(f"Processing failed: {error_count}/{len(parts)} parts failed")
    
    # Report completion
    await report_progress(98)
    
    return {
        "products": total_products,
        "rules": total_rules,
        "matrices": total_matrices,
        "parts_processed": success_count,
        "elapsed_seconds": elapsed,
        "source_hash": source_hash
    }


async def _read_document(file_path: Path) -> str:
    """Read document content from file."""
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        # Use PDF reader
        from zag.readers import PDFReader
        reader = PDFReader(use_gpu=False)
        doc = await reader.read(str(file_path))
        return doc.content
    elif suffix in [".md", ".txt"]:
        # Read text file directly
        return file_path.read_text(encoding="utf-8")
    else:
        # Try to read as text
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Unsupported file format: {suffix}") from e
