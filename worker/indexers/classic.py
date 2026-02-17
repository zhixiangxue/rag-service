"""Classic RAG indexer - Split documents into chunks and index."""
import time
from pathlib import Path
from rich.console import Console
from typing import Dict, Any, Optional, List, Tuple

from ..processors.classic_processor import ClassicDocumentProcessor
from ..constants import ProcessingMode
from ..config import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
    EMBEDDING_URI, OPENAI_API_KEY,
    VECTOR_STORE_HOST, VECTOR_STORE_PORT,
    MEILISEARCH_HOST, MEILISEARCH_API_KEY,
    USE_GPU, NUM_THREADS,
    MAX_CHUNK_TOKENS, TABLE_MAX_TOKENS, TARGET_TOKEN_SIZE,
    NUM_KEYWORDS, MAX_PAGES_PER_PART
)
from zag.schemas.pdf import PDF
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


async def index_classic(
    file_path: Path,
    workspace_dir: Path,
    collection_name: str,
    meilisearch_index_name: str,
    custom_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[callable] = None,
    vector_store_grpc_port: int = 6334
) -> Dict[str, Any]:
    """
    Classic RAG indexing: split document into chunks and index.
    
    For large files, uses page-range-based reading instead of physical splitting:
    1. Read file in page ranges (e.g., 1-100, 101-200, ...)
    2. Apply HeadingCorrection to each range
    3. Merge all ranges into a single PDF document
    4. Process the merged document (split, tables, metadata, indexing)
    
    Args:
        file_path: Path to the PDF file
        workspace_dir: Working directory for this file
        collection_name: Qdrant collection name
        meilisearch_index_name: MeiliSearch index name
        custom_metadata: Optional business metadata
        on_progress: Optional progress callback function(progress: int)
        vector_store_grpc_port: Qdrant gRPC port
    
    Returns:
        Processing result with unit_count and other stats
    """
    console.print(f"\n[bold cyan]🚀 Classic RAG Indexing[/bold cyan]")
    console.print(f"   File: {file_path}")
    console.print(f"   Workspace: {workspace_dir}\n")
    
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
    
    console.print(f"\n[black on cyan] Step 1/9: Analyzing document [/black on cyan]")
    
    source_hash = calculate_file_hash(file_path)
    console.print(f"   🔑 Source hash: {source_hash}")
    
    # Get page count
    from pypdf import PdfReader
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        console.print(f"   📖 Total pages: {total_pages}")
    except Exception as e:
        raise Exception(f"Failed to read PDF: {e}")
    
    # Calculate page ranges
    if total_pages > MAX_PAGES_PER_PART:
        page_ranges = calculate_page_ranges(total_pages, MAX_PAGES_PER_PART)
        console.print(f"   ✂️  Large file, will read in {len(page_ranges)} parts: {page_ranges}")
    else:
        page_ranges = [(1, total_pages)]
        console.print(f"   ✅ Small file, reading all pages at once")
    
    # Inject mode into custom_metadata
    metadata_with_mode = {**(custom_metadata or {}), "mode": ProcessingMode.CLASSIC}
    
    # Create output directory using source hash
    output_dir = workspace_dir / "output" / source_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processor
    processor = ClassicDocumentProcessor(output_root=output_dir)
    
    # Step 2: Read document in parts and merge (15-25%)
    console.print(f"\n[black on cyan] Step 2/9: Reading document [/black on cyan]")
    merged_doc: Optional[PDF] = None
    
    for part_idx, (page_start, page_end) in enumerate(page_ranges, 1):
        part_progress = 15 + int((part_idx - 0.5) / len(page_ranges) * 10)
        await report_progress(part_progress)
        
        console.print(f"\n[cyan]--- Reading part {part_idx}/{len(page_ranges)}: pages {page_start}-{page_end} ---[/cyan]")
        
        # Read this page range
        doc = await processor.read_document(
            pdf_path=file_path,
            use_gpu=USE_GPU,
            num_threads=NUM_THREADS,
            page_range=(page_start, page_end)
        )
        
        # Merge documents
        if merged_doc is None:
            merged_doc = doc
        else:
            console.print(f"  🔗 Merging with previous document...")
            merged_doc = merged_doc + doc
            console.print(f"  ✅ Merged: {len(merged_doc.pages)} pages total")
    
    # Step 3: Set merged document and update doc_id (25%)
    await report_progress(25)
    
    # Update doc_id to use source_hash (consistent across all parts)
    merged_doc.doc_id = source_hash
    merged_doc.metadata.md5 = source_hash
    processor.set_document(merged_doc)
    
    console.print(f"\n[black on cyan] Step 3/9: Document ready [/black on cyan]")
    console.print(f"   Pages: {len(merged_doc.pages)}, Characters: {len(merged_doc.content):,}")
    console.print(f"   doc_id: {merged_doc.doc_id}")
    
    # Dump to cache for reuse
    cache_dir = Path.home() / ".zag" / "cache" / "readers" / "docling"
    archive_path = merged_doc.dump(cache_dir)
    console.print(f"   💾 Cached: {archive_path}")
    
    # Step 4: Inject custom metadata (30%)
    await report_progress(30)
    if metadata_with_mode:
        console.print(f"\n[black on cyan] Step 4/9: Injecting metadata [/black on cyan]")
        processor.set_business_context(custom_metadata=metadata_with_mode)
        console.print(f"  ✅ Metadata injected")
    
    # Step 5: Split document (30-45%)
    await report_progress(35)
    console.print(f"\n[black on cyan] Step 5/9: Splitting document [/black on cyan]")
    await processor.split_document(
        max_chunk_tokens=MAX_CHUNK_TOKENS,
        table_max_tokens=TABLE_MAX_TOKENS,
        target_token_size=TARGET_TOKEN_SIZE,
        export_visualization=True
    )
    console.print(f"  ✅ Split into {len(processor.units)} units")
    await report_progress(45)
    
    # Step 6: Process tables (45-65%)
    await report_progress(50)
    console.print(f"\n[black on cyan] Step 6/9: Processing tables [/black on cyan]")
    llm_uri = f"{LLM_PROVIDER}/{LLM_MODEL}"
    await processor.process_tables(
        llm_uri=llm_uri,
        api_key=LLM_API_KEY,
    )
    console.print(f"  ✅ Tables processed")
    await report_progress(65)
    
    # Step 7: Extract metadata (65-75%)
    await report_progress(70)
    console.print(f"\n[black on cyan] Step 7/9: Extracting metadata [/black on cyan]")
    await processor.extract_metadata(
        llm_uri=llm_uri,
        api_key=LLM_API_KEY,
        num_keywords=NUM_KEYWORDS,
    )
    console.print(f"  ✅ Metadata extracted")
    await report_progress(75)
    
    # Step 8: Build vector index (75-90%)
    await report_progress(80)
    console.print(f"\n[black on cyan] Step 8/9: Building vector index [/black on cyan]")
    await processor.build_vector_index(
        embedding_uri=EMBEDDING_URI,
        qdrant_host=VECTOR_STORE_HOST,
        qdrant_port=VECTOR_STORE_PORT,
        qdrant_grpc_port=vector_store_grpc_port,
        collection_name=collection_name,
        api_key=OPENAI_API_KEY
    )
    console.print(f"  ✅ Vector index built")
    await report_progress(90)
    
    # Step 9: Build fulltext index (90-98%) - Optional
    await report_progress(92)
    console.print(f"\n[black on cyan] Step 9/9: Building fulltext index [/black on cyan]")
    try:
        await processor.build_fulltext_index(
            meilisearch_url=MEILISEARCH_HOST,
            index_name=meilisearch_index_name
        )
        console.print(f"  ✅ Fulltext index built")
    except Exception as e:
        console.print(f"  [yellow]⚠ Fulltext index skipped: {str(e)[:80]}[/yellow]")
        console.print(f"  [dim]Task will continue without fulltext search capability[/dim]")
    await report_progress(98)
    
    elapsed = time.time() - start_time
    
    # Summary
    console.print(f"\n[bold green]📊 Processing Summary[/bold green]")
    console.print(f"  Parts: {len(page_ranges)}")
    console.print(f"  Total units: {len(processor.units)}")
    console.print(f"  Time: {elapsed:.2f}s\n")
    
    return {
        "unit_count": len(processor.units),
        "parts": len(page_ranges),
        "elapsed_seconds": elapsed,
        "source_hash": source_hash
    }
