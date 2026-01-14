#!/usr/bin/env python3
import os
import sys
import time
import shutil
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from mortgage_document_processor import MortgageDocumentProcessor, console
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers import VectorRetriever
from zag.utils.retry import aretry
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
LLM_URI = "ollama/qwen2.5:7b"
OUTPUT_ROOT = Path(__file__).parent / "output"

# ============================================================================
# BUSINESS INTEGRATION: Document Configuration
# ============================================================================
# This dict demonstrates what business systems should provide
# In production, this would come from your backend API/database
DOCUMENT_CONFIG = {
    "file_path": None,  # Required: will be set from command line
    "custom_metadata": {
        # Business IDs (NOT framework IDs)
        "lender_id": "USDA_RD_2024",
        "doc_id": "USDA_RURAL_HOUSING_001",
        "agent_id": "AGENT_USDA_001",
        "loan_id": "USDA_SAMPLE_2024",
        
        # Business attributes
        "department": "USDA Rural Development",
        "product_type": "USDA",  # FHA, VA, conventional, jumbo, USDA
        "region": "US-Rural",
        "version": "2024.1",
        "status": "active",
        "effective_date": "2024-01-01",
        "expiration_date": "2025-12-31",
        "owner": "USDA Rural Housing Service",
        "approval_status": "approved",
        
        # Additional tracking
        "source_system": "usda_rural_housing_portal",
        "ingestion_timestamp": "2024-01-14T18:40:00Z",
        "program_type": "Section 502 Direct Loan",  # USDA specific
        "eligibility_scope": "rural_area",
        "income_limit_category": "low_to_moderate"
    }
}


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
    console.print("  🚀 Mortgage Document Ingestion - Business Integration Demo", style="bold cyan")
    console.print("=" * 70 + "\n")
    
    # Parse command line arguments or prompt for input
    if len(sys.argv) < 2:
        console.print("📝 No file path provided in command line", style="yellow")
        console.print("\nUsage:")
        console.print("  python run_mortgage_pipeline.py <pdf_file_path>")
        console.print("\nExample:")
        console.print("  python run_mortgage_pipeline.py ./files/mortgage_guidelines.pdf")
        console.print()
        
        # Interactive input
        try:
            file_input = input("Please enter PDF file path: ").strip()
            if not file_input:
                console.print("\n❌ No file path provided", style="bold red")
                sys.exit(1)
            pdf_path = Path(file_input)
        except KeyboardInterrupt:
            console.print("\n\n❌ Cancelled by user", style="bold red")
            sys.exit(1)
    else:
        # Get file path from command line
        pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        console.print(f"\n❌ File not found: {pdf_path}", style="bold red")
        sys.exit(1)
    
    # Set file path in config
    DOCUMENT_CONFIG["file_path"] = pdf_path
    
    # Display configuration
    console.print("📄 Document Configuration:", style="bold yellow")
    console.print(f"  File: {pdf_path}")
    console.print(f"  Size: {pdf_path.stat().st_size / (1024*1024):.1f} MB")
    console.print(f"\n📋 Custom Metadata (Business Data):", style="bold yellow")
    for key, value in DOCUMENT_CONFIG["custom_metadata"].items():
        console.print(f"  - {key}: {value}")
    console.print()
    
    # Check prerequisites
    if not check_prerequisites():
        console.print("\n❌ Prerequisites not met. Please fix issues above.", style="bold red")
        sys.exit(1)
    
    # Calculate file hash for output directory
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
        )
        console.print(f"✅ Document loaded: {len(processor.document.content):,} characters\n")
        
        # Step 2: Inject custom metadata (KEY INTEGRATION POINT!)
        print_section("📋 Step 2: Inject Custom Metadata", "-")
        processor.set_business_context(
            custom_metadata=DOCUMENT_CONFIG["custom_metadata"]
        )
        console.print(f"\n✅ Custom metadata injected - all units will inherit these values\n")
        
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
        
        # Vector index with retry
        console.print("Building vector index...")
        try:
            async with aretry(
                max_attempts=5,
                backoff_factor=2.0,
                initial_delay=2.0,
                exceptions=(RuntimeError, Exception)
            ):
                await processor.build_vector_index(
                    embedding_uri=EMBEDDING_URI,
                    qdrant_host=QDRANT_HOST,
                    qdrant_port=QDRANT_PORT,
                    qdrant_grpc_port=QDRANT_GRPC_PORT,
                    collection_name="mortgage_guidelines",
                    clear_existing=False,
                    api_key=OPENAI_API_KEY
                )
        except Exception as e:
            console.print(f"[red]❌ Vector indexing failed after all retries: {e}[/red]")
            raise
        
        # Fulltext index
        console.print("\nBuilding fulltext index...")
        await processor.build_fulltext_index(
            meilisearch_url=MEILISEARCH_URL,
            index_name="mortgage_guidelines",
            clear_existing=False
        )
        console.print(f"\n✅ Indices built\n")
        
        # Step 7: Test retrieval
        print_section("🔍 Step 7: Test Retrieval", "-")
        
        console.print("🔎 Creating vector retriever...\n")
        
        # Create retriever
        embedder = Embedder(EMBEDDING_URI, api_key=OPENAI_API_KEY)
        vector_store = QdrantVectorStore.server(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True,
            collection_name="mortgage_guidelines",
            embedder=embedder
        )
        retriever = VectorRetriever(vector_store=vector_store, top_k=3)
        
        test_queries = [
            "FHA 贷款的首付要求是什么？",
            "VA 贷款的资格条件有哪些？",
            "Fannie Mae 的 LTV 要求",
        ]
        
        for i, query in enumerate(test_queries, 1):
            console.print(f"\n[bold cyan]{i}. Query:[/bold cyan] {query}")
            
            start = time.time()
            results = retriever.retrieve(query)
            elapsed = time.time() - start
            
            console.print(f"   ✅ Found {len(results)} results ({elapsed*1000:.0f}ms)")
            
            if results:
                top_result = results[0]
                rprint(top_result)
        
        console.print(f"\n✅ Retrieval tests complete\n")
        
        # Summary
        total_time = time.time() - start_time
        print_section("📊 Pipeline Summary")
        
        console.print(f"✅ Pipeline complete in {total_time:.2f}s", style="bold green")
        console.print(f"\nKey metrics:")
        if processor.document:
            console.print(f"   - Document: {processor.document.metadata.file_name}")
        if processor.units:
            console.print(f"   - Units: {len(processor.units)}")
            # Show custom metadata from first unit
            first_unit = processor.units[0]
            if first_unit.metadata and first_unit.metadata.custom:
                console.print(f"   - Custom metadata keys: {list(first_unit.metadata.custom.keys())}")
        if processor.vector_indexer:
            console.print(f"   - Vector index: {processor.vector_indexer.count()} units")
        if processor.fulltext_indexer:
            console.print(f"   - Fulltext index: {processor.fulltext_indexer.count()} units")
        console.print(f"   - Output directory: {output_dir}")
        
        console.print(f"\n🎉 Integration demo complete!", style="bold green")
        console.print(f"\n💡 Next steps for business integration:", style="bold cyan")
        console.print("   1. Replace DOCUMENT_CONFIG with your API/database data")
        console.print("   2. Add business-specific filters in retrieval")
        console.print("   3. Implement authentication and access control")
        console.print("   4. Set up monitoring and logging")
        console.print("   5. Deploy as microservice or Celery tasks")
        
    except Exception as e:
        console.print(f"\n❌ Failed to process {pdf_path.name}: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        console.print(f"\n❌ Pipeline failed: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        sys.exit(1)
