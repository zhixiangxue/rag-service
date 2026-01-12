#!/usr/bin/env python3
"""
Command-line runner for MortgageDocumentProcessor

This script demonstrates how to use the refactored processor.
All logic is now in MortgageDocumentProcessor class for easy Celery integration.

Usage:
    python run_mortgage_pipeline.py

Configuration:
    Edit constants below or pass via environment variables
"""

import os
import sys
import time
import shutil
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from mortgage_document_processor import MortgageDocumentProcessor, console
from rich.table import Table
from rich.panel import Panel

# Load environment variables
load_dotenv()

# Configuration
API_KEY = os.getenv("BAILIAN_API_KEY")  # Optional for remote LLM
OLLAMA_BASE_URL = "http://localhost:11434"
MEILISEARCH_URL = "http://127.0.0.1:7700"
EMBEDDING_MODEL = "jina/jina-embeddings-v2-base-en:latest"
LLM_MODEL = "qwen2.5:7b"
EMBEDDING_URI = f"ollama/{EMBEDDING_MODEL}"
LLM_URI = f"ollama/{LLM_MODEL}"
FILES_DIR = Path(__file__).parent / "files"
OUTPUT_ROOT = Path(__file__).parent / "output"
CHROMA_PERSIST_DIR = OUTPUT_ROOT / "chroma_db"


def print_section(title: str, char: str = "="):
    """Print section header"""
    console.print(f"\n{char * 70}")
    console.print(f"  {title}", style="bold cyan")
    console.print(f"{char * 70}\n")


def calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Calculate xxhash of a file
    
    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read (default 8KB)
        
    Returns:
        Hexadecimal hash string
    """
    try:
        import xxhash
    except ImportError:
        raise ImportError(
            "xxhash is required for file hashing. "
            "Install it with: pip install xxhash"
        )
    
    # Use xxh64 for good balance of speed and collision resistance
    hash_func = xxhash.xxh64()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def check_prerequisites():
    """Check prerequisites before running"""
    print_section("🔍 Checking Prerequisites")
    
    issues = []
    
    # Check LLM
    if LLM_URI.startswith("bailian/"):
        if not API_KEY:
            issues.append("❌ .env file missing BAILIAN_API_KEY (Bailian LLM required)")
        else:
            console.print(f"✅ API Key found: {API_KEY[:10]}...")
    else:
        console.print(f"📦 Using local LLM: {LLM_URI} (no API key needed)")
    
    # Check Ollama
    try:
        import httpx
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            console.print(f"✅ Ollama running: {OLLAMA_BASE_URL}")
        else:
            issues.append(f"❌ Ollama error: {response.status_code}")
    except Exception as e:
        issues.append(f"❌ Cannot connect to Ollama: {e}")
        console.print(f"   💡 Hint: Start Ollama with 'ollama serve'")
    
    # Check GPU/CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"✅ GPU available: {gpu_name}")
            console.print(f"   CUDA version: {torch.version.cuda}")
        else:
            console.print(f"⚠️  GPU not available, using CPU (slower)", style="yellow")
            console.print(f"   💡 Hint: Install CUDA PyTorch for acceleration")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    # Check PDF files
    pdf_files = list(FILES_DIR.glob("*.pdf"))
    if not pdf_files:
        issues.append(f"❌ No PDF files found: {FILES_DIR}")
    else:
        console.print(f"✅ Found {len(pdf_files)} PDF file(s):")
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            console.print(f"   - {pdf.name} ({size_mb:.1f} MB)")
    
    # Check Meilisearch
    try:
        import meilisearch
        client = meilisearch.Client(MEILISEARCH_URL)
        health = client.health()
        if health.get("status") == "available":
            console.print(f"✅ Meilisearch running: {MEILISEARCH_URL}")
        else:
            issues.append("❌ Meilisearch unavailable")
    except Exception as e:
        issues.append(f"❌ Cannot connect to Meilisearch: {e}")
    
    if issues:
        console.print("\n⚠️  Issues found:", style="bold yellow")
        for issue in issues:
            console.print(f"  {issue}")
        return False
    
    console.print("\n✅ All prerequisites OK!", style="bold green")
    return True


async def main():
    """Main pipeline execution"""
    console.print("\n" + "=" * 70)
    console.print("  🚀 Mortgage Document Pipeline", style="bold cyan")
    console.print("=" * 70 + "\n")
    
    # Check prerequisites
    if not check_prerequisites():
        console.print("\n❌ Prerequisites not met. Please fix issues above.", style="bold red")
        sys.exit(1)
    
    # Determine which PDFs to process
    if len(sys.argv) > 1:
        # Command-line argument provided: process specified file
        pdf_path = Path(sys.argv[1])
        pdf_files = [pdf_path]
        console.print(f"\n📌 Processing specified file: {pdf_path}")
    else:
        # No argument: process all PDFs in files directory
        pdf_files = sorted(FILES_DIR.glob("*.pdf"))
        if not pdf_files:
            console.print("\n❌ No PDF files found in files directory", style="bold red")
            sys.exit(1)
        console.print(f"\n📌 Processing all PDFs in {FILES_DIR}: {len(pdf_files)} file(s)\n")
    
    # Process each PDF
    for pdf_path in pdf_files:
        # Calculate file hash for output directory name
        file_hash = calculate_file_hash(pdf_path)
        output_dir = OUTPUT_ROOT / file_hash
        
        console.print(f"\n{'=' * 70}")
        console.print(f"📄 Processing: {pdf_path.name}", style="bold yellow")
        console.print(f"🔑 File hash: {file_hash}")
        console.print(f"📁 Output directory: {output_dir}")
        console.print(f"{'=' * 70}\n")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy source file to output directory for redundancy
        source_copy = output_dir / pdf_path.name
        if not source_copy.exists():
            shutil.copy2(pdf_path, source_copy)
            console.print(f"📋 Source file copied to output directory: {source_copy.name}\n")
        else:
            console.print(f"📋 Source file already exists in output directory\n")
        
        # Create processor
        processor = MortgageDocumentProcessor(output_root=output_dir)
        
        start_time = time.time()
        
        try:
            # Step 1: Read document
            print_section("📄 Step 1: Read Document", "-")
            await processor.read_document(
                pdf_path=pdf_path,
                use_gpu=True,
                num_threads=8,
                quality_threshold=90.0
            )
            console.print(f"✅ Document loaded: {len(processor.document.content):,} characters\n")
            
            # Step 2: Split document
            print_section("🔪 Step 2: Split Document", "-")
            await processor.split_document(
                max_chunk_tokens=1200,
                table_max_tokens=1500,
                target_token_size=800,
                export_visualization=True
            )
            console.print(f"✅ Split into {len(processor.units)} units\n")
            
            # Step 3: Process tables
            print_section("📊 Step 3: Process Tables", "-")
            await processor.process_tables(
                llm_uri=LLM_URI,
                api_key=API_KEY,
                use_cache=True
            )
            console.print(f"✅ Tables processed\n")
            
            # Step 4: Extract metadata
            print_section("🏷️  Step 4: Extract Metadata", "-")
            await processor.extract_metadata(
                llm_uri=LLM_URI,
                api_key=API_KEY,
                num_keywords=5,
                use_cache=True
            )
            console.print(f"✅ Metadata extracted\n")
            
            # Step 5: Build indices
            print_section("📚 Step 5: Build Indices", "-")
            
            # Vector index
            console.print("Building vector index...")
            await processor.build_vector_index(
                embedding_uri=EMBEDDING_URI,
                chroma_persist_dir=CHROMA_PERSIST_DIR,
                collection_name="mortgage_guidelines",
                clear_existing=True
            )
            
            # Fulltext index
            console.print("\nBuilding fulltext index...")
            await processor.build_fulltext_index(
                meilisearch_url=MEILISEARCH_URL,
                index_name="mortgage_guidelines",
                clear_existing=True
            )
            console.print(f"\n✅ Indices built\n")
            
            # Step 6: Test retrieval
            print_section("🔍 Step 6: Test Retrieval", "-")
            
            test_queries = [
                "FHA 贷款的首付要求是什么？",
                "VA 贷款的资格条件有哪些？",
                "Fannie Mae 的 LTV 要求",
            ]
            
            for i, query in enumerate(test_queries, 1):
                console.print(f"\n[bold cyan]{i}. Query:[/bold cyan] {query}")
                
                start = time.time()
                results = await processor.retrieve_fusion(
                    query=query,
                    embedding_uri=EMBEDDING_URI,
                    chroma_persist_dir=CHROMA_PERSIST_DIR,
                    meilisearch_url=MEILISEARCH_URL,
                    top_k=3
                )
                elapsed = time.time() - start
                
                console.print(f"   ✅ Found {len(results)} results ({elapsed*1000:.0f}ms)")
                
                if results:
                    top_result = results[0]
                    preview = top_result.content[:80].replace("\n", " ")
                    
                    source_info = "N/A"
                    if top_result.metadata and top_result.metadata.context_path:
                        source_info = top_result.metadata.context_path.split('/')[0]
                    
                    console.print(f"   📄 Source: {source_info}")
                    console.print(f"   💯 Score: {top_result.score:.4f}")
                    console.print(f"   📝 Preview: {preview}...")
            
            console.print(f"\n✅ Retrieval tests complete\n")
            
            # Step 7: Test postprocessing
            print_section("🔄 Step 7: Test Postprocessing", "-")
            
            query = "不同贷款产品的利率和首付要求比较"
            console.print(f"Query: [bold]{query}[/bold]\n")
            
            # Raw retrieval
            raw_results = await processor.retrieve_fusion(
                query=query,
                embedding_uri=EMBEDDING_URI,
                chroma_persist_dir=CHROMA_PERSIST_DIR,
                meilisearch_url=MEILISEARCH_URL,
                top_k=10
            )
            console.print(f"Raw results: {len(raw_results)} units")
            
            # Apply postprocessing
            processed_results = await processor.postprocess(
                query=query,
                results=raw_results,
                similarity_threshold=0.6,
                dedup_strategy="exact",
                context_window=1
            )
            console.print(f"Processed results: {len(processed_results)} units")
            
            # Display results
            if processed_results:
                table = Table(title="Postprocessing Results", show_header=True, header_style="bold magenta")
                table.add_column("#", style="cyan", width=4)
                table.add_column("Score", style="green", width=10)
                table.add_column("Source", style="yellow", width=30)
                table.add_column("Preview", style="white", width=50)
                
                for i, unit in enumerate(processed_results[:5], 1):
                    score = f"{unit.score:.4f}" if hasattr(unit, 'score') and unit.score else "N/A"
                    
                    source = "N/A"
                    if unit.metadata and unit.metadata.context_path:
                        source = unit.metadata.context_path.split('/')[0]
                    
                    preview = unit.content[:47].replace("\n", " ") + "..."
                    table.add_row(str(i), score, source, preview)
                
                console.print("\n")
                console.print(table)
            
            console.print(f"\n✅ Postprocessing complete\n")
            
            # Summary for this document
            total_time = time.time() - start_time
            print_section("📊 Document Summary")
            
            console.print(f"✅ Document complete in {total_time:.2f}s", style="bold green")
            console.print(f"\nKey metrics:")
            if processor.document:
                console.print(f"   - Document: {processor.document.metadata.file_name}")
            if processor.units:
                console.print(f"   - Units: {len(processor.units)}")
            if processor.vector_indexer:
                console.print(f"   - Vector index: {processor.vector_indexer.count()} units")
            if processor.fulltext_indexer:
                console.print(f"   - Fulltext index: {processor.fulltext_indexer.count()} units")
            console.print(f"   - Output directory: {output_dir}")
            
        except Exception as e:
            console.print(f"\n❌ Failed to process {pdf_path.name}: {e}", style="bold red")
            import traceback
            traceback.print_exc()
            # Continue to next file
        
    # Final summary
    console.print("\n" + "=" * 70)
    console.print("✅ All documents processed successfully!", style="bold green")
    console.print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n❌ Pipeline failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        sys.exit(1)
