"""Classic RAG indexer - Split documents into chunks and index."""
import time
from pathlib import Path
from rich.console import Console
from typing import Dict, Any, Optional

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
from zag.utils.hash import calculate_file_hash

console = Console()


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
    
    The processor now handles large file splitting internally with part-level
    checkpoint recovery. This function just orchestrates the high-level pipeline.
    
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
    
    # Step 1: Calculate source hash (10%)
    await report_progress(10)
    console.print(f"\n[black on cyan] Step 1/9: Analyzing document [/black on cyan]")
    
    source_hash = calculate_file_hash(file_path)
    console.print(f"   🔑 Source hash: {source_hash}")
    
    # Inject mode into custom_metadata
    metadata_with_mode = {**(custom_metadata or {}), "mode": ProcessingMode.CLASSIC}
    
    # Create output directory using source hash
    output_dir = workspace_dir / source_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create processor (will auto-restore checkpoint if exists)
    processor = ClassicDocumentProcessor.from_checkpoint(output_root=output_dir)
    
    # Step 2: Read document (15-25%)
    # Processor handles large file splitting + part-level checkpoint internally
    console.print(f"\n[black on cyan] Step 2/9: Reading document [/black on cyan]")
    await report_progress(15)
    
    doc = await processor.read_document(
        pdf_path=file_path,
        max_pages_per_part=MAX_PAGES_PER_PART
    )
    
    await report_progress(25)
    
    # Step 3: Update doc_id (30%)
    console.print(f"\n[black on cyan] Step 3/9: Document ready [/black on cyan]")
    
    # Update doc_id to use source_hash (consistent)
    doc.doc_id = source_hash
    doc.metadata.md5 = source_hash
    
    console.print(f"   Pages: {len(doc.pages)}, Characters: {len(doc.content):,}")
    console.print(f"   doc_id: {doc.doc_id}")
    
    await report_progress(30)
    
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
    console.print(f"  Total units: {len(processor.units)}")
    console.print(f"  Time: {elapsed:.2f}s\n")
    
    return {
        "unit_count": len(processor.units),
        "elapsed_seconds": elapsed,
        "source_hash": source_hash
    }
