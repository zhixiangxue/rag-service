#!/usr/bin/env python3
import os
import sys
import time
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from document_processor import MortgageDocumentProcessor, console
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers import VectorRetriever
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For OpenAI embeddings
BAILIAN_API_KEY = os.getenv("BAILIAN_API_KEY")  # For Bailian LLM (optional)
MEILISEARCH_URL = "http://127.0.0.1:7700"
QDRANT_HOST = "localhost"
QDRANT_PORT = 16333  # HTTP REST API port
QDRANT_GRPC_PORT = 16334  # gRPC port (much faster!)
EMBEDDING_URI = "openai/text-embedding-3-small"
LLM_URI = "openai/gpt-4o-mini"
OUTPUT_ROOT = Path(__file__).parent / "output"


def print_section(title: str, char: str = "="):
    """Print section header"""
    console.print(f"\n{char * 70}")
    console.print(f"  {title}", style="bold cyan")
    console.print(f"{char * 70}\n")


def check_prerequisites():
    """Check prerequisites before running"""
    print_section("🔍 Checking Prerequisites")
    
    issues = []
    
    # Check LLM
    if LLM_URI.startswith("bailian/"):
        if not BAILIAN_API_KEY:
            issues.append("❌ .env file missing BAILIAN_API_KEY (Bailian LLM required)")
        else:
            console.print(f"✅ Bailian API Key found: {BAILIAN_API_KEY[:10]}...")
    else:
        console.print(f"📦 Using local LLM: {LLM_URI} (no API key needed)")
    
    # Check Embedding API key
    if EMBEDDING_URI.startswith("openai/"):
        if not OPENAI_API_KEY:
            issues.append("❌ .env file missing OPENAI_API_KEY (OpenAI embeddings required)")
        else:
            console.print(f"✅ OpenAI API Key found: {OPENAI_API_KEY[:10]}...")
    else:
        console.print(f"📦 Using local embeddings: {EMBEDDING_URI}")
    
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
    
    # Check Qdrant
    try:
        import httpx
        response = httpx.get(f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz", timeout=5.0)
        if response.status_code == 200:
            console.print(f"✅ Qdrant running: {QDRANT_HOST}:{QDRANT_PORT}")
        else:
            issues.append(f"❌ Qdrant error: {response.status_code}")
    except Exception as e:
        issues.append(f"❌ Cannot connect to Qdrant: {e}")
        console.print(f"   💡 Hint: Start Qdrant with './playground/start_qdrant.bat'")
    
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
    console.print("  🚀 Mortgage Document Ingestion - Batch Processing", style="bold cyan")
    console.print("=" * 70 + "\n")
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        console.print("📝 No manifest file provided", style="yellow")
        console.print("\nUsage:")
        console.print("  python import.py <manifest.json>")
        console.print("\nExample:")
        console.print("  python import.py ./manifest.json")
        console.print()
        
        # Interactive input
        try:
            file_input = input("Please enter manifest file path: ").strip()
            if not file_input:
                console.print("\n❌ No file path provided", style="bold red")
                sys.exit(1)
            manifest_path = Path(file_input)
        except KeyboardInterrupt:
            console.print("\n\n❌ Cancelled by user", style="bold red")
            sys.exit(1)
    else:
        manifest_path = Path(sys.argv[1])
    
    if not manifest_path.exists():
        console.print(f"\n❌ Manifest file not found: {manifest_path}", style="bold red")
        sys.exit(1)
    
    # Load manifest
    console.print(f"📄 Loading manifest: {manifest_path}")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    tasks = manifest.get('tasks', [])
    if not tasks:
        console.print("\n❌ No tasks found in manifest", style="bold red")
        sys.exit(1)
    
    console.print(f"   ✅ Loaded {len(tasks)} tasks\n")
    
    # Check prerequisites
    if not check_prerequisites():
        console.print("\n❌ Prerequisites not met. Please fix issues above.", style="bold red")
        sys.exit(1)
    
    # Process each task
    total_parts = sum(len(task['parts']) for task in tasks)
    console.print(f"\n{'=' * 70}")
    console.print(f"📦 Processing {len(tasks)} tasks ({total_parts} parts total)", style="bold yellow")
    console.print(f"{'=' * 70}\n")
    
    success_count = 0
    error_count = 0
    
    for task_idx, task in enumerate(tasks, 1):
        source_file = task['source_file']
        source_hash = task['source_hash']
        metadata_file = task.get('metadata_file')
        total_pages = task['total_pages']
        parts = task['parts']
        
        console.print(f"\n{'=' * 70}")
        console.print(f"📄 Task {task_idx}/{len(tasks)}: {Path(source_file).name}", style="bold cyan")
        console.print(f"   🔑 Source hash: {source_hash}")
        console.print(f"   📚 Total pages: {total_pages}")
        console.print(f"   📦 Parts: {len(parts)}")
        console.print(f"{'=' * 70}\n")
        
        # Load custom metadata from JSON
        custom_metadata = {}
        if metadata_file and Path(metadata_file).exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    custom_metadata = json.load(f)
                console.print(f"📋 Loaded metadata from: {Path(metadata_file).name}")
                for key, value in custom_metadata.items():
                    console.print(f"   - {key}: {value}")
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to load metadata: {e}[/yellow]")
        else:
            console.print(f"[yellow]⚠️  No metadata file found[/yellow]")
        
        # Create output directory using source hash
        output_dir = OUTPUT_ROOT / source_hash
        output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print(f"\n📁 Output directory: {output_dir}\n")
        
        # Process each part
        for part_idx, part_info in enumerate(parts, 1):
            part_file = Path(part_info['file'])
            page_range = part_info['page_range']
            
            console.print(f"\n--- Part {part_idx}/{len(parts)} ---")
            console.print(f"📄 File: {part_file.name}")
            console.print(f"📚 Pages: {page_range}")
            
            if not part_file.exists():
                console.print(f"[red]❌ Part file not found: {part_file}[/red]")
                error_count += 1
                continue
            
            try:
                # Create processor
                processor = MortgageDocumentProcessor(output_root=output_dir)
                
                start_time = time.time()
                
                # Step 1: Read document (with source_hash)
                print_section(f"📄 Step 1: Read Document (Part {part_idx})", "-")
                await processor.read_document(
                    pdf_path=part_file,
                    use_gpu=True,
                    num_threads=8,
                    source_hash=source_hash  # ← 关键：使用源文件 hash
                )
                console.print(f"✅ Document loaded: {len(processor.document.content):,} characters\n")
                
                # Step 2: Inject custom metadata
                print_section("📋 Step 2: Inject Custom Metadata", "-")
                processor.set_business_context(custom_metadata=custom_metadata)
                console.print(f"\n✅ Custom metadata injected\n")
                
                # Step 3: Split document
                print_section("🔪 Step 3: Split Document", "-")
                await processor.split_document(
                    max_chunk_tokens=1200,
                    table_max_tokens=1500,
                    target_token_size=800,
                    export_visualization=True
                )
                console.print(f"✅ Split into {len(processor.units)} units\n")
                
                # Step 4: Process tables
                print_section("📊 Step 4: Process Tables", "-")
                await processor.process_tables(
                    llm_uri=LLM_URI,
                    api_key=BAILIAN_API_KEY,
                )
                console.print(f"✅ Tables processed\n")
                
                # Step 5: Extract metadata
                print_section("🏷️  Step 5: Extract Metadata", "-")
                await processor.extract_metadata(
                    llm_uri=LLM_URI,
                    api_key=BAILIAN_API_KEY,
                    num_keywords=5,
                )
                console.print(f"✅ Metadata extracted\n")
                
                # Step 6: Build indices
                print_section("📚 Step 6: Build Indices", "-")
                
                # Vector index
                console.print("Building vector index...")
                await processor.build_vector_index(
                    embedding_uri=EMBEDDING_URI,
                    qdrant_host=QDRANT_HOST,
                    qdrant_port=QDRANT_PORT,
                    qdrant_grpc_port=QDRANT_GRPC_PORT,
                    collection_name="mortgage_guidelines",
                    clear_existing=False,  # 不清空，累加
                    api_key=OPENAI_API_KEY
                )
                
                # Fulltext index
                console.print("\nBuilding fulltext index...")
                await processor.build_fulltext_index(
                    meilisearch_url=MEILISEARCH_URL,
                    index_name="mortgage_guidelines",
                    clear_existing=False  # 不清空，累加
                )
                console.print(f"\n✅ Indices built\n")
                
                elapsed = time.time() - start_time
                console.print(f"✅ Part {part_idx} processed in {elapsed:.2f}s", style="bold green")
                success_count += 1
                
            except Exception as e:
                console.print(f"\n❌ Failed to process part {part_idx}: {e}", style="bold red")
                import traceback
                traceback.print_exc()
                error_count += 1
    
    # Final summary
    console.print("\n" + "=" * 70)
    console.print("  📊 Batch Processing Summary", style="bold green")
    console.print("=" * 70 + "\n")
    
    console.print(f"✅ Total tasks: {len(tasks)}")
    console.print(f"✅ Total parts: {total_parts}")
    console.print(f"✅ Successful: {success_count}")
    if error_count > 0:
        console.print(f"❌ Failed: {error_count}", style="bold red")
    
    console.print(f"\n🎉 Batch processing complete!", style="bold green")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n❌ Pipeline failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        sys.exit(1)
