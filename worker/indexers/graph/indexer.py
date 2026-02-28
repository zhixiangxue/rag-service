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
    
    # Step 1: Calculate source hash and page ranges
    await report_progress(10)
    console.print("\n[bold white on blue] Step 1 [/bold white on blue] Preparing...")

    source_hash = calculate_file_hash(file_path)
    console.print(f"  🔑 Source hash: {source_hash}")

    # Check file type and get page count for PDFs
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            console.print(f"  📖 Total pages: {total_pages}")
        except Exception as e:
            raise Exception(f"Failed to read PDF: {e}")

        # Calculate page ranges
        if total_pages > MAX_PAGES_PER_PART:
            page_ranges = calculate_page_ranges(total_pages, MAX_PAGES_PER_PART)
            console.print(f"  ✂️  Large file → {len(page_ranges)} parts: {page_ranges}")
        else:
            page_ranges = [(1, total_pages)]
            console.print(f"  ✅ Small file → single part")
    else:
        page_ranges = None
        total_pages = 1
        console.print(f"  📄 Non-PDF → single part")

    # Inject mode into custom_metadata
    metadata_with_mode = {**(custom_metadata or {}), "mode": ProcessingMode.GRAPH}

    # Create output directory using source hash
    output_dir = workspace_dir / "output" / source_hash
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Read document content
    await report_progress(15)
    console.print("\n[bold white on blue] Step 2 [/bold white on blue] Reading document...")

    if page_ranges:
        # PDF: read parts and merge into a single doc
        doc = None
        for idx, (page_start, page_end) in enumerate(page_ranges, 1):
            console.print(f"  [{idx}/{len(page_ranges)}] Reading pages {page_start}-{page_end}...")
            part_doc = await _read_document(file_path, page_range=(page_start, page_end))
            doc = part_doc if doc is None else doc + part_doc
            await report_progress(15 + int(idx / len(page_ranges) * 40))
        content = doc.content
    else:
        # Non-PDF: read text directly
        console.print("  [1/1] Reading full document...")
        content = file_path.read_text(encoding="utf-8")
        await report_progress(55)

    console.print(f"  ✅ {len(content):,} chars loaded")

    # Step 3: Extract program structure
    await report_progress(60)
    console.print("\n[bold white on blue] Step 3 [/bold white on blue] Extracting program structure...")

    extractor = MortgageProgramExtractor(
        llm_uri=f"{LLM_PROVIDER}/{LLM_MODEL}",
        api_key=LLM_API_KEY,
    )
    result = await extractor.extract(
        document_content=content,
        source_file=str(file_path),
    )
    console.print(f"  ✅ {len(result.stages_completed)} stages — Products: {len(result.products)}, Rules: {len(result.rules)}, Matrices: {len(result.matrices)}")

    # Step 4: Store in graph database
    await report_progress(90)
    console.print("\n[bold white on blue] Step 4 [/bold white on blue] Storing in graph database...")

    from zag.storages.graph import FalkorDBGraphStorage
    from domain.mortgage import MortgageGraphStorage

    base_storage = FalkorDBGraphStorage(
        host=FALKORDB_HOST,
        port=FALKORDB_PORT,
        graph_name=graph_name,
    )
    with base_storage:
        mortgage_storage = MortgageGraphStorage(base_storage)
        program_id = mortgage_storage.store_program(result.to_graph_data())
        console.print(f"[green]✓ Stored program[/green]: {program_id}")

    total_products = len(result.products)
    total_rules = len(result.rules)
    total_matrices = len(result.matrices)

    # Report near completion
    await report_progress(95)

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]✓ Done[/bold green] in {elapsed:.2f}s")
    console.print(f"  Products: {total_products}, Rules: {total_rules}, Matrices: {total_matrices}")

    # Report completion
    await report_progress(98)

    return {
        "products": total_products,
        "rules": total_rules,
        "matrices": total_matrices,
        "elapsed_seconds": elapsed,
        "source_hash": source_hash
    }


async def _read_document(file_path: Path, page_range: Optional[Tuple[int, int]] = None):
    """
    Read a PDF document (optionally a specific page range) and apply heading correction.

    Returns the doc object (for PDF, supports `+` merging).
    Only PDF is supported; non-PDF files are read inline in index_graph.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from zag.readers import MinerUReader
        from zag.postprocessors.correctors import HeadingCorrector

        reader = MinerUReader(backend="pipeline")
        doc = reader.read(str(file_path), page_range=page_range)

        # Apply heading correction
        console.print("  🔧 Correcting headings...")
        corrector = HeadingCorrector(
            llm_uri=f"{LLM_PROVIDER}/{LLM_MODEL}",
            api_key=LLM_API_KEY,
            llm_correction=True
        )
        doc = await corrector.acorrect_document(doc)
        console.print("  ✅ Headings corrected")

        return doc
    else:
        raise ValueError(f"_read_document only supports PDF; got: {suffix}")
