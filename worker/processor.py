"""Document processing pipeline - integrates prepare_manifest and import logic."""
import json
import time
from pathlib import Path
from PyPDF2 import PdfReader
from rich.console import Console
from typing import Dict, Any, Optional

from .document_processor import MortgageDocumentProcessor
from .extract_pdf_pages import split_pdf_with_overlap
from .config import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
    EMBEDDING_URI, OPENAI_API_KEY,
    VECTOR_STORE_HOST, VECTOR_STORE_PORT,
    MEILISEARCH_HOST, MEILISEARCH_API_KEY,
    USE_GPU, NUM_THREADS,
    MAX_CHUNK_TOKENS, TABLE_MAX_TOKENS, TARGET_TOKEN_SIZE,
    MAX_PAGES_PER_PART, NUM_KEYWORDS
)
from zag.utils.hash import calculate_file_hash

console = Console()


async def prepare_single_file_manifest(
    file_path: Path,
    workspace_dir: Path,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Prepare manifest for a single file (adapted from prepare_manifest.py).
    
    Args:
        file_path: Path to the PDF file
        workspace_dir: Working directory for this file
        custom_metadata: Optional business metadata
    
    Returns:
        Manifest dict with processing tasks
    """
    console.print(f"\n[cyan]📋 Preparing manifest for: {file_path.name}[/cyan]")
    
    # Calculate source file hash
    source_hash = calculate_file_hash(file_path)
    console.print(f"   🔑 Hash: {source_hash}")
    
    # Check page count
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        console.print(f"   📖 Pages: {total_pages}")
    except Exception as e:
        raise Exception(f"Failed to read PDF: {e}")
    
    # Create temp_parts directory inside workspace
    temp_dir = workspace_dir / "temp_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create task
    task = {
        "source_file": str(file_path.absolute()),
        "source_hash": source_hash,
        "metadata": custom_metadata or {},
        "total_pages": total_pages,
        "parts": []
    }
    
    # Split if needed
    if total_pages > MAX_PAGES_PER_PART:
        console.print(f"   ✂️  Large file, splitting into parts...")
        
        try:
            part_files = split_pdf_with_overlap(
                input_pdf=str(file_path),
                output_dir=str(temp_dir),
                pages_per_part=MAX_PAGES_PER_PART,
                overlap_pages=0
            )
            
            for part_file in part_files:
                part_name = Path(part_file).stem
                page_range = part_name.split('_pages_')[-1] if '_pages_' in part_name else "unknown"
                
                task["parts"].append({
                    "file": str(part_file),
                    "page_range": page_range
                })
            
            console.print(f"   ✅ Split into {len(part_files)} parts")
            
        except Exception as e:
            console.print(f"   [yellow]⚠️  Failed to split, using original: {e}[/yellow]")
            # Fallback: use original file
            task["parts"].append({
                "file": str(file_path.absolute()),
                "page_range": f"1-{total_pages}"
            })
    else:
        console.print(f"   ✅ Small file, no splitting needed")
        task["parts"].append({
            "file": str(file_path.absolute()),
            "page_range": f"1-{total_pages}"
        })
    
    # Save manifest to workspace
    manifest = {"tasks": [task]}
    manifest_path = workspace_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    console.print(f"   💾 Manifest saved: {manifest_path}")
    console.print(f"   📦 Parts to process: {len(task['parts'])}\n")
    
    return manifest


async def process_document(
    file_path: Path,
    workspace_dir: Path,
    collection_name: str,
    meilisearch_index_name: str,
    custom_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[callable] = None,
    vector_store_grpc_port: int = 6334
) -> Dict[str, Any]:
    """
    Process document (adapted from import.py).
    
    Args:
        file_path: Path to the PDF file
        workspace_dir: Working directory for this file
        custom_metadata: Optional business metadata
        on_progress: Optional progress callback function(progress: int)
    
    Returns:
        Processing result with unit_count and other stats
    """
    console.print(f"\n[bold cyan]🚀 Processing Document[/bold cyan]")
    console.print(f"   File: {file_path}")
    console.print(f"   Workspace: {workspace_dir}\n")
    
    start_time = time.time()
    
    # Helper function to report progress (支持 async 回调)
    async def report_progress(progress: int):
        if on_progress:
            try:
                # 检查是否是 async 函数
                import inspect
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(progress)
                else:
                    on_progress(progress)
            except Exception as e:
                console.print(f"[yellow]⚠ Progress callback error: {e}[/yellow]")
    
    # Step 1: Prepare manifest (10-15%)
    await report_progress(10)
    manifest = await prepare_single_file_manifest(
        file_path=file_path,
        workspace_dir=workspace_dir,
        custom_metadata=custom_metadata
    )
    await report_progress(15)
    
    task = manifest["tasks"][0]
    source_hash = task["source_hash"]
    parts = task["parts"]
    metadata = task["metadata"]
    
    # Create output directory using source hash
    output_dir = workspace_dir / "output" / source_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_units = 0
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
            # Create processor
            processor = MortgageDocumentProcessor(output_root=output_dir)
            
            # Step 2.1: Read document (0-10% of part range)
            await report_progress(part_progress_start)
            console.print(f"[dim]  Step 1: Reading document...[/dim]")
            await processor.read_document(
                pdf_path=part_file,
                use_gpu=USE_GPU,
                num_threads=NUM_THREADS,
                source_hash=source_hash
            )
            console.print(f"  ✅ Document loaded: {len(processor.document.content):,} characters")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.1))
            
            # Step 2.2: Inject custom metadata (10-15% of part range)
            if metadata:
                console.print(f"[dim]  Step 2: Injecting metadata...[/dim]")
                processor.set_business_context(custom_metadata=metadata)
                console.print(f"  ✅ Metadata injected")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.15))
            
            # Step 2.3: Split document (15-30% of part range)
            console.print(f"[dim]  Step 3: Splitting document...[/dim]")
            await processor.split_document(
                max_chunk_tokens=MAX_CHUNK_TOKENS,
                table_max_tokens=TABLE_MAX_TOKENS,
                target_token_size=TARGET_TOKEN_SIZE,
                export_visualization=True
            )
            part_units = len(processor.units)
            console.print(f"  ✅ Split into {part_units} units")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.3))
            
            # Step 2.4: Process tables (30-50% of part range)
            console.print(f"[dim]  Step 4: Processing tables...[/dim]")
            llm_uri = f"{LLM_PROVIDER}/{LLM_MODEL}"
            await processor.process_tables(
                llm_uri=llm_uri,
                api_key=LLM_API_KEY,
            )
            console.print(f"  ✅ Tables processed")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.5))
            
            # Step 2.5: Extract metadata (50-70% of part range)
            console.print(f"[dim]  Step 5: Extracting metadata...[/dim]")
            await processor.extract_metadata(
                llm_uri=llm_uri,
                api_key=LLM_API_KEY,
                num_keywords=NUM_KEYWORDS,
            )
            console.print(f"  ✅ Metadata extracted")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.7))
            
            # Step 2.6: Build vector index (70-85% of part range)
            console.print(f"[dim]  Step 6: Building vector index...[/dim]")
            await processor.build_vector_index(
                embedding_uri=EMBEDDING_URI,
                qdrant_host=VECTOR_STORE_HOST,
                qdrant_port=VECTOR_STORE_PORT,
                qdrant_grpc_port=vector_store_grpc_port,
                collection_name=collection_name,
                clear_existing=False,  # Accumulate, don't clear
                api_key=OPENAI_API_KEY
            )
            console.print(f"  ✅ Vector index built")
            await report_progress(part_progress_start + int((part_progress_end - part_progress_start) * 0.85))
            
            # Step 2.7: Build fulltext index (85-100% of part range)
            console.print(f"[dim]  Step 7: Building fulltext index...[/dim]")
            await processor.build_fulltext_index(
                meilisearch_url=MEILISEARCH_HOST,
                index_name=meilisearch_index_name,
                clear_existing=False  # Accumulate, don't clear
            )
            console.print(f"  ✅ Fulltext index built")
            await report_progress(part_progress_end)
            
            total_units += part_units
            success_count += 1
            console.print(f"[green]✓ Part {part_idx} completed ({part_units} units)[/green]")
            
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
    console.print(f"  Total units: {total_units}")
    console.print(f"  Time: {elapsed:.2f}s\n")
    
    if error_count > 0:
        raise Exception(f"Processing failed: {error_count}/{len(parts)} parts failed")
    
    # Report completion - will be set to 100 by caller
    await report_progress(98)
    
    return {
        "unit_count": total_units,
        "parts_processed": success_count,
        "elapsed_seconds": elapsed,
        "source_hash": source_hash
    }
