"""
Graph Indexer - Index mortgage documents into graph database.

This module provides graph-based indexing functionality for mortgage program documents.
It extracts program structure using LLM and stores the result in a graph database.

Architecture (after refactor):
- Uses page-range-based reading instead of physical PDF splitting
- Similar to classic.py: read in page ranges → extract → store
"""
import time
from pathlib import Path
from rich.console import Console
from typing import Dict, Any, Optional, List, Tuple

from .extractor import MortgageProgramExtractor
from domain.mortgage.graph import FullExtractionResult
from worker.constants import ProcessingMode
from worker.config import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
    FALKORDB_HOST, FALKORDB_PORT,
    MAX_PAGES_PER_PART,
)
from zag.utils.hash import calculate_file_hash

console = Console()


def calculate_page_ranges(total_pages: int, pages_per_part: int) -> List[Tuple[int, int]]:
    """
    Calculate page ranges for large file processing.
    
    Args:
        total_pages: Total number of pages in the PDF
        pages_per_part: Maximum pages per part
        
    Returns:
        List of (start, end) tuples (1-based, inclusive)
    """
    ranges = []
    start = 1
    while start <= total_pages:
        end = min(start + pages_per_part - 1, total_pages)
        ranges.append((start, end))
        start = end + 1
    return ranges


async def index_graph(
    file_path: Path,
    workspace_dir: Path,
    graph_name: str = "mortgage_programs",
    custom_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Graph-based indexing: Extract program structure and store in graph database.
    
    Uses page-range-based reading instead of physical PDF splitting:
    1. Calculate source hash and page count
    2. Read document in page ranges
    3. Extract program structure from each range
    4. Store in graph database
    
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
    
    # Step 1: Calculate source hash and page count (10-15%)
    await report_progress(10)
    
    source_hash = calculate_file_hash(file_path)
    console.print(f"🔑 Source hash: {source_hash}")
    
    # Check file type and get page count for PDFs
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            console.print(f"📖 Total pages: {total_pages}")
        except Exception as e:
            raise Exception(f"Failed to read PDF: {e}")
        
        # Calculate page ranges
        if total_pages > MAX_PAGES_PER_PART:
            page_ranges = calculate_page_ranges(total_pages, MAX_PAGES_PER_PART)
            console.print(f"✂️  Large file, will read in {len(page_ranges)} page ranges: {page_ranges}")
        else:
            page_ranges = [(1, total_pages)]
            console.print(f"✅ Small file, reading all pages at once")
    else:
        # For non-PDF files, treat as single "page"
        page_ranges = None
        total_pages = 1
        console.print(f"📄 Non-PDF file, processing as single part")
    
    # Inject mode into custom_metadata
    metadata_with_mode = {**(custom_metadata or {}), "mode": ProcessingMode.GRAPH}
    
    # Create output directory using source hash
    output_dir = workspace_dir / "output" / source_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_products = 0
    total_rules = 0
    total_matrices = 0
    success_count = 0
    error_count = 0
    
    # Step 2: Process each page range
    if page_ranges:
        # PDF processing with page ranges
        for range_idx, (page_start, page_end) in enumerate(page_ranges, 1):
            range_progress = 15 + int((range_idx - 0.5) / len(page_ranges) * 80)
            await report_progress(range_progress)
            
            console.print(f"\n[cyan]--- Part {range_idx}/{len(page_ranges)}: pages {page_start}-{page_end} ---[/cyan]")
            
            try:
                # Step 2.1: Read document content for this page range
                console.print(f"[dim]  Step 1: Reading document...[/dim]")
                content = await _read_document(file_path, page_range=(page_start, page_end))
                console.print(f"  ✅ Document loaded: {len(content):,} characters")
                
                # Step 2.2: Extract program structure
                console.print(f"[dim]  Step 2: Extracting program structure...[/dim]")
                llm_uri = f"{LLM_PROVIDER}/{LLM_MODEL}"
                
                extractor = MortgageProgramExtractor(
                    llm_uri=llm_uri,
                    api_key=LLM_API_KEY,
                )
                
                result = await extractor.extract(
                    document_content=content,
                    source_file=str(file_path),
                )
                console.print(f"  ✅ Extraction complete: {len(result.stages_completed)} stages")
                
                # Step 2.3: Store in graph database
                console.print(f"[dim]  Step 3: Storing in graph database...[/dim]")
                
                from zag.storages.graph import FalkorDBGraphStorage
                from .storage import MortgageGraphStorage
                
                base_storage = FalkorDBGraphStorage(
                    host=FALKORDB_HOST,
                    port=FALKORDB_PORT,
                    graph_name=graph_name,
                )
                
                with base_storage:
                    mortgage_storage = MortgageGraphStorage(base_storage)
                    program_id = mortgage_storage.store_program(result.to_graph_data())
                    console.print(f"  ✅ Stored program: {program_id}")
                
                total_products += len(result.products)
                total_rules += len(result.rules)
                total_matrices += len(result.matrices)
                success_count += 1
                
                console.print(f"[green]✓ Part {range_idx} completed[/green]")
                console.print(f"    Products: {len(result.products)}, Rules: {len(result.rules)}, Matrices: {len(result.matrices)}")
                
            except Exception as e:
                console.print(f"[red]✗ Part {range_idx} failed: {e}[/red]")
                import traceback
                traceback.print_exc()
                error_count += 1
    else:
        # Non-PDF processing (single file)
        await report_progress(50)
        
        try:
            console.print(f"[dim]  Step 1: Reading document...[/dim]")
            content = await _read_document(file_path)
            console.print(f"  ✅ Document loaded: {len(content):,} characters")
            
            console.print(f"[dim]  Step 2: Extracting program structure...[/dim]")
            llm_uri = f"{LLM_PROVIDER}/{LLM_MODEL}"
            
            extractor = MortgageProgramExtractor(
                llm_uri=llm_uri,
                api_key=LLM_API_KEY,
            )
            
            result = await extractor.extract(
                document_content=content,
                source_file=str(file_path),
            )
            console.print(f"  ✅ Extraction complete: {len(result.stages_completed)} stages")
            
            console.print(f"[dim]  Step 3: Storing in graph database...[/dim]")
            
            from zag.storages.graph import FalkorDBGraphStorage
            from .storage import MortgageGraphStorage
            
            base_storage = FalkorDBGraphStorage(
                host=FALKORDB_HOST,
                port=FALKORDB_PORT,
                graph_name=graph_name,
            )
            
            with base_storage:
                mortgage_storage = MortgageGraphStorage(base_storage)
                program_id = mortgage_storage.store_program(result.to_graph_data())
                console.print(f"  ✅ Stored program: {program_id}")
            
            total_products += len(result.products)
            total_rules += len(result.rules)
            total_matrices += len(result.matrices)
            success_count += 1
            
        except Exception as e:
            console.print(f"[red]✗ Processing failed: {e}[/red]")
            import traceback
            traceback.print_exc()
            error_count += 1
    
    elapsed = time.time() - start_time
    
    # Report near completion
    await report_progress(95)
    
    # Summary
    parts_count = len(page_ranges) if page_ranges else 1
    console.print(f"\n[bold green]📊 Processing Summary[/bold green]")
    console.print(f"  Total parts: {parts_count}")
    console.print(f"  Successful: {success_count}")
    console.print(f"  Failed: {error_count}")
    console.print(f"  Total products: {total_products}")
    console.print(f"  Total rules: {total_rules}")
    console.print(f"  Total matrices: {total_matrices}")
    console.print(f"  Time: {elapsed:.2f}s\n")
    
    if error_count > 0:
        raise Exception(f"Processing failed: {error_count}/{parts_count} parts failed")
    
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


async def _read_document(file_path: Path, page_range: Optional[Tuple[int, int]] = None) -> str:
    """
    Read document content from file.
    
    Args:
        file_path: Path to the document file
        page_range: Optional page range (start, end) for PDFs (1-based, inclusive)
    
    Returns:
        Document content as string
    """
    suffix = file_path.suffix.lower()
    
    if suffix == ".pdf":
        # Use DoclingReader for PDF (supports page_range)
        from zag.readers import DoclingReader
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
        
        pdf_options = PdfPipelineOptions()
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CPU  # Use CPU for stability
        )
        
        reader = DoclingReader(pdf_pipeline_options=pdf_options)
        doc = reader.read(str(file_path), page_range=page_range)
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
