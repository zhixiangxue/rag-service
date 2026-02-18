"""LOD indexer - Index entire document with multi-resolution views."""
import time
import uuid
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from typing import Dict, Any, Optional

from zag.readers import MinerUReader, MarkdownTreeReader
from zag.extractors import CompressionExtractor
from zag.postprocessors.correctors import HeadingCorrector
from zag.schemas import BaseUnit, ContentView, LODLevel, TextUnit, UnitMetadata, UnitType
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore

from ..constants import ProcessingMode
from ..config import (
    LLM_PROVIDER, LLM_MODEL, LLM_API_KEY,
    EMBEDDING_URI, OPENAI_API_KEY,
    VECTOR_STORE_HOST, VECTOR_STORE_PORT
)

console = Console()

# Maximum pages allowed for LOD processing
MAX_LOD_PAGES = 200


# Prompts for LOD extraction
QUICK_CHECK_PROMPT = """You are a mortgage document analysis expert. Extract the Quick-Check layer from the following document.

**What is the Quick-Check layer?**
This layer contains hard "one-strike-out" eligibility criteria used to quickly eliminate non-qualifying applications.

**Typical characteristics**:
- Uses negation words like "NOT ELIGIBLE", "ineligible", "prohibited", "not permitted"
- Involves minimum/maximum numeric limits (credit score, LTV, loan amount)
- Clear restrictions on borrower identity, property type, geographic location
- Simple conditions that can be quickly evaluated
- Usually found at the beginning of documents

**Extraction requirements**:
1. Extract only hard rejection criteria
2. Preserve complete logical relationships (AND/OR/IF-THEN/EXCEPT)
3. Preserve all numbers, percentages, entity names
4. Use concise but complete natural language
5. Maintain original section structure
6. Target length: 1-2k tokens

**Now analyze the following document and extract the Quick-Check layer**:

{text}

**Output requirements**:
- Maintain Markdown format
- Organize content with section headings
- Each criterion on a separate line
- Include only hard rejection criteria
"""


SHORTLIST_PROMPT = """You are a professional mortgage document compression expert. 

**CRITICAL REQUIREMENT**: You MUST compress the following text to EXACTLY {target_tokens} tokens or less.

**Absolutely preserve (NEVER delete or modify)**:
1. All numbers, percentages, amounts, dates
2. All placeholders in format {{{{HTML_TABLE_X}}}}
3. All eligibility condition if/then logic relationships
4. All entity names (company names, product names, location names)
5. All negation expressions and restriction clauses

**What to compress (actively remove)**:
- Verbose explanations and redundant descriptions
- Repetitive legal statements and disclaimers
- Multiple similar examples (keep only 1 representative example)
- Overly detailed process descriptions
- Explanatory text outside of tables

**Compression priorities**:
- Priority 1: Preserve core business rules and ALL numbers
- Priority 2: Maintain logical relationship integrity
- Priority 3: Actively remove verbose language and redundant content
- Priority 4: Remove all unnecessary formatting and whitespace

**Output requirements**:
- MUST be {target_tokens} tokens or LESS
- Do NOT add any summary language
- Output compressed content directly without extra explanations
- Maintain Markdown format but remove unnecessary formatting
- Keep all {{{{HTML_TABLE_X}}}} placeholders exactly as they appear

Original text:
{text}

Output compressed text (MUST be <= {target_tokens} tokens):"""


async def index_lod(
    file_path: Path,
    collection_name: str,
    custom_metadata: Optional[Dict[str, Any]] = None,
    on_progress: Optional[callable] = None,
    quick_check_target: int = 2000,
    shortlist_target: int = 60000,
    vector_store_grpc_port: int = 6334
) -> Dict[str, Any]:
    """
    LOD indexing: index entire document with 4 views (LOW, MEDIUM, HIGH, FULL).
    
    Args:
        file_path: Path to the PDF file
        collection_name: Qdrant collection name
        custom_metadata: Optional business metadata
        quick_check_target: Target token count for Quick-Check layer (default: 2000)
        shortlist_target: Target token count for Shortlist layer (default: 60000)
        vector_store_grpc_port: Qdrant gRPC port
    
    Returns:
        Processing result with lod_unit_id and views_count
    """
    console.print(f"\n[bold cyan]🔄 LOD Indexing[/bold cyan]")
    console.print(f"   File: {file_path.name}")
    console.print(f"   Collection: {collection_name}\n")
    
    start_time = time.time()
    
    if not file_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    # Step 1: Check page count first (lightweight check before heavy processing)
    console.print("[yellow]Step 1: Checking document size...[/yellow]")
    
    from pypdf import PdfReader
    pdf_reader = PdfReader(str(file_path))
    page_count = len(pdf_reader.pages)
    
    if page_count > MAX_LOD_PAGES:
        # Display prominent warning
        warning_table = Table(show_header=False, box=None, padding=(1, 2))
        warning_table.add_column(style="bold red")
        warning_table.add_row(f"⚠️  DOCUMENT TOO LARGE ⚠️")
        warning_table.add_row(f"")
        warning_table.add_row(f"Page count: {page_count} pages")
        warning_table.add_row(f"Max allowed: {MAX_LOD_PAGES} pages")
        warning_table.add_row(f"")
        warning_table.add_row(f"LOD mode is not suitable for large documents.")
        warning_table.add_row(f"Please use CLASSIC mode or split the document.")
        
        console.print("")
        console.print(Panel(warning_table, title="[bold red]PROCESSING REJECTED[/bold red]", border_style="red"))
        console.print("")
        
        raise ValueError(f"Document has {page_count} pages, exceeds LOD limit of {MAX_LOD_PAGES} pages. Use CLASSIC mode.")
    
    console.print(f"  Pages: {page_count} (✓ within limit)")
    
    # Step 2: Read PDF with MinerU
    console.print("[yellow]Step 2: Reading PDF with MinerU...[/yellow]")
    if on_progress:
        await on_progress(10)
    
    reader = MinerUReader(backend="pipeline")
    doc = reader.read(str(file_path))

    # Apply heading correction
    console.print("  🔧 Correcting headings...")
    corrector = HeadingCorrector(
        llm_uri=f"{LLM_PROVIDER}/{LLM_MODEL}",
        api_key=LLM_API_KEY,
        llm_correction=True
    )
    doc = await corrector.acorrect_document(doc)
    console.print("  ✅ Headings corrected")
    
    # Dump to cache for reuse
    cache_dir = Path.home() / ".zag" / "cache" / "readers" / "mineru"
    archive_path = doc.dump(cache_dir)
    console.print(f"  💾 Cached: {archive_path}")

    lod_full = doc.content
    
    extractor = CompressionExtractor(f"{LLM_PROVIDER}/{LLM_MODEL}", api_key=LLM_API_KEY)
    original_tokens = extractor.count_tokens(lod_full)
    
    console.print(f"[green]✓ PDF read successfully[/green]")
    console.print(f"  Content: {len(lod_full):,} chars")
    console.print(f"  Tokens: {original_tokens:,}")
    
    # Step 3: Extract Quick-Check layer (lod_low)
    console.print("\n[yellow]Step 3: Extracting Quick-Check layer (LOW)...[/yellow]")
    if on_progress:
        await on_progress(30)
    
    lod_low = extractor.compress(
        text=lod_full,
        prompt=QUICK_CHECK_PROMPT,
        target_tokens=quick_check_target,
        chunk_size=3000,
        max_depth=2
    )
    lod_low_tokens = extractor.count_tokens(lod_low)
    
    console.print(f"[green]✓ Quick-Check extracted[/green]")
    console.print(f"  Tokens: {lod_low_tokens:,} ({lod_low_tokens/original_tokens:.1%})")
    
    # Step 4: Extract Shortlist layer (lod_medium)
    console.print("\n[yellow]Step 4: Extracting Shortlist layer (MEDIUM)...[/yellow]")
    if on_progress:
        await on_progress(50)
    
    lod_medium = extractor.compress(
        text=lod_full,
        prompt=SHORTLIST_PROMPT,
        target_tokens=shortlist_target,
        chunk_size=3000,
        max_depth=2
    )
    lod_medium_tokens = extractor.count_tokens(lod_medium)
    
    console.print(f"[green]✓ Shortlist extracted[/green]")
    console.print(f"  Tokens: {lod_medium_tokens:,} ({lod_medium_tokens/original_tokens:.1%})")
    
    # Step 5: Generate DocTree (lod_high)
    console.print("\n[yellow]Step 5: Building DocTree structure (HIGH)...[/yellow]")
    if on_progress:
        await on_progress(70)
    
    try:
        tree_reader = MarkdownTreeReader(
            llm_uri=f"{LLM_PROVIDER}/{LLM_MODEL}",
            api_key=LLM_API_KEY
        )
        lod_tree = await tree_reader.read(content=lod_full, generate_summaries=True)
        
        all_nodes = lod_tree.collect_all_nodes()
        console.print(f"[green]✓ DocTree built successfully[/green]")
        console.print(f"  Total nodes: {len(all_nodes)}")
        console.print(f"  Root: {lod_tree.doc_name}")
    except Exception as e:
        console.print(f"[red]✗ DocTree building failed: {e}[/red]")
        raise
    
    # Step 6: Create LOD Unit
    console.print("\n[yellow]Step 6: Creating LOD Unit...[/yellow]")
    if on_progress:
        await on_progress(85)
    
    doc_id = doc.doc_id
    lod_unit_id = f"{doc_id}_lod"
    
    # Create TextUnit with views
    # Inject mode into custom_metadata
    metadata_with_mode = {**(custom_metadata or {}), "mode": ProcessingMode.LOD}
    
    lod_unit = TextUnit(
        unit_id=lod_unit_id,
        doc_id=doc_id,
        content=lod_low,  # Primary content for retrieval
        embedding_content=lod_low,  # Embed on low layer
        metadata=UnitMetadata(
            doc_id=doc_id,
            unit_type=UnitType.TEXT,
            source_file=file_path.name,
            custom_metadata=metadata_with_mode
        ),
        views=[
            ContentView(level=LODLevel.LOW, content=lod_low, token_count=lod_low_tokens),
            ContentView(level=LODLevel.MEDIUM, content=lod_medium, token_count=lod_medium_tokens),
            ContentView(level=LODLevel.HIGH, content=lod_tree.to_dict()),
            ContentView(level=LODLevel.FULL, content=lod_full, token_count=original_tokens)
        ]
    )
    
    console.print(f"[green]✓ LOD Unit created[/green]")
    console.print(f"  Unit ID: {lod_unit_id}")
    console.print(f"  Views: {len(lod_unit.views)}")
    
    # Step 7: Index to vector store
    console.print("\n[yellow]Step 7: Indexing to vector store...[/yellow]")
    if on_progress:
        await on_progress(95)
    
    embedder = Embedder(EMBEDDING_URI, api_key=OPENAI_API_KEY)
    
    vector_store = QdrantVectorStore.server(
        host=VECTOR_STORE_HOST,
        port=VECTOR_STORE_PORT,
        prefer_grpc=True,
        grpc_port=vector_store_grpc_port,
        collection_name=collection_name,
        embedder=embedder,
        timeout=60
    )
    
    # Clear existing data for this document before indexing
    console.print(f"  🗑️  Clearing old LOD data for doc_id: {doc_id}")
    await vector_store.aremove({
        "doc_id": doc_id,
        "metadata.custom.mode": ProcessingMode.LOD
    })
    
    # Index the LOD unit
    vector_store.add([lod_unit])
    
    console.print(f"[green]✓ Indexed to vector store[/green]")
    console.print(f"  Collection: {collection_name}")
    
    elapsed = time.time() - start_time
    
    # Print summary
    console.print("\n")
    console.print(Panel.fit(
        f"[bold green]LOD Indexing Complete[/bold green]\n\n" +
        f"Unit ID: {lod_unit_id}\n" +
        f"Views: 4 layers\n\n" +
        f"FULL (Original): {original_tokens:,} tokens\n" +
        f"LOW (Quick-Check): {lod_low_tokens:,} tokens ({lod_low_tokens/original_tokens:.1%})\n" +
        f"MEDIUM (Shortlist): {lod_medium_tokens:,} tokens ({lod_medium_tokens/original_tokens:.1%})\n" +
        f"HIGH (DocTree): {len(all_nodes)} nodes\n\n" +
        f"Time: {elapsed:.2f}s ({elapsed/60:.1f} min)",
        title="Summary",
        border_style="green"
    ))
    
    return {
        "unit_count": 1,
        "lod_unit_id": lod_unit_id,
        "views_count": 4,
        "elapsed_seconds": elapsed,
        "token_stats": {
            "full": original_tokens,
            "low": lod_low_tokens,
            "medium": lod_medium_tokens,
            "tree_nodes": len(all_nodes)
        }
    }
