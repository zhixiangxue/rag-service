#!/usr/bin/env python3
"""
Classic Document Processor

A production-ready document processor for classic RAG (chunk-based) indexing.
Designed for Celery integration with independent, reusable methods.

Key Features:
- Each method is independent and can be called separately
- All parameters passed explicitly (no config objects)
- State stored in instance (units, indexers, paths)
- Full caching support with quality validation
- Supports partial pipeline execution

Architecture:
- Input: Single PDF file (not directory)
- State: document, units, indexers stored in instance
- Output: Organized in output_root subdirectories
- Caching: Markdown, JSON for expensive operations
"""

import os
import json
import time
import pickle
from pathlib import Path
from typing import Optional, List, Union, Dict, Any
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tenacity import retry, stop_after_attempt, wait_exponential, wait_fixed, retry_if_exception_type
from pydantic import BaseModel
from functools import wraps

# Zag imports
from zag.readers import MinerUReader
from zag.splitters import MarkdownHeaderSplitter, TextSplitter, TableSplitter, RecursiveMergingSplitter
from zag.extractors import KeywordExtractor, TableEnricher, TableSummarizer
from zag.postprocessors.correctors import HeadingCorrector
from zag.parsers import TableParser
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.indexers import VectorIndexer, FullTextIndexer
from zag.schemas import DocumentMetadata, Page, UnitType
from zag.schemas.pdf import PDF
from zag.schemas.unit import TextUnit, TableUnit
from zag.utils.hash import calculate_file_hash

from ..constants import ProcessingMode
from ..config import LLM_PROVIDER, LLM_MODEL, LLM_API_KEY

console = Console()


def checkpoint_cache(func):
    """
    Decorator to cache function calls in checkpoint
    
    Like functools.lru_cache but persists to disk via processor checkpoint.
    Automatically generates cache key from function name + kwargs.
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Generate cache key from function name + kwargs
        cache_key = f"{func.__name__}({','.join(f'{k}={v}' for k, v in sorted(kwargs.items()))})" if kwargs else f"{func.__name__}()"
        
        # Check cache
        if cache_key in self._completed_calls:
            console.print(f"[dim]⏭️  {cache_key} (cached)[/dim]")
            # Return appropriate default based on method
            return getattr(self, 'units', None) if 'units' in func.__name__ or 'split' in func.__name__ or 'process' in func.__name__ or 'extract' in func.__name__ else None
        
        # Execute function
        result = await func(self, *args, **kwargs)
        
        # Save to cache
        self._completed_calls.add(cache_key)
        self._save_checkpoint()
        console.print(f"[dim]✓ {cache_key}[/dim]")
        
        return result
    
    return wrapper


class PagePart(BaseModel):
    """Represents a part of a large PDF file for chunked processing"""
    start_page: int  # 1-based inclusive
    end_page: int    # 1-based inclusive
    completed: bool = False
    part_doc: Optional[PDF] = None  # The processed document for this part (with heading correction)
    
    def __repr__(self):
        status = "✓" if self.completed else "○"
        return f"Part({self.start_page}-{self.end_page}) {status}"
    
    class Config:
        # Allow arbitrary types (for PDF objects)
        arbitrary_types_allowed = True


class ClassicDocumentProcessor:
    """Document processor for classic RAG (chunk-based) indexing."""

    def __init__(self, output_root: Path):
        """
        Initialize processor with output directory

        Args:
            output_root: Root directory for all outputs (subdirs auto-created)
        """
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        # State variables (populated by methods)
        self.pdf_path: Optional[Path] = None  # Current PDF being processed
        self.document: Optional[PDF] = None
        self.units: List[Union[TextUnit, TableUnit]] = []
        
        # Page range being processed (for large file processing)
        self._current_page_range: Optional[tuple] = None  # (start, end) 1-based inclusive
        
        # Track completed steps for checkpoint recovery
        self._completed_calls: set = set()  # Cache function calls: "method_name(arg1=val1,arg2=val2)"
        self._parts: List[PagePart] = []  # Track parts for large file processing

        # Lazy-initialized subdirectories
        self._split_dir: Optional[Path] = None
        self._cache_dir: Optional[Path] = None
    
    @classmethod
    def from_checkpoint(cls, output_root: Path) -> 'ClassicDocumentProcessor':
        """
        Restore processor from checkpoint if exists, otherwise create new one
        
        Args:
            output_root: Root directory for all outputs
            
        Returns:
            Restored or new processor instance
        """
        checkpoint_file = Path(output_root) / "checkpoints" / "processor.pkl"
        
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    old_processor = pickle.load(f)
                
                # Create a new processor with current __init__ logic
                processor = cls(output_root)
                
                # Copy state from old checkpoint
                processor.pdf_path = old_processor.pdf_path
                processor.document = old_processor.document
                processor.units = old_processor.units
                processor._current_page_range = getattr(old_processor, '_current_page_range', None)
                processor._completed_calls = getattr(old_processor, '_completed_calls', set())
                processor._parts = getattr(old_processor, '_parts', [])
                
                console.print(f"[bold cyan]🔄 Checkpoint restored![/bold cyan]")
                console.print(f"   Checkpoint: {checkpoint_file}")
                console.print(f"   Document: {len(processor.document.pages) if processor.document else 0} pages")
                console.print(f"   Units: {len(processor.units)}")
                console.print(f"   Completed calls: {len(processor._completed_calls)}")
                if processor._parts:
                    completed = sum(1 for p in processor._parts if p.completed)
                    console.print(f"   Parts: {completed}/{len(processor._parts)} completed - {processor._parts}")
                console.print(f"[yellow]⚠️  Will skip completed calls automatically[/yellow]\n")
                
                return processor
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to restore checkpoint: {e}[/yellow]")
                console.print(f"[yellow]   Creating new processor...[/yellow]\n")
        
        # No checkpoint or restore failed, create new
        return cls(output_root)

    @property
    def split_dir(self) -> Path:
        """Directory for split visualization"""
        if self._split_dir is None:
            self._split_dir = self.output_root / "split"
            self._split_dir.mkdir(exist_ok=True)
        return self._split_dir
    
    @property
    def cache_dir(self) -> Path:
        """Directory for document cache"""
        if self._cache_dir is None:
            self._cache_dir = self.output_root / "doc"
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        return self._cache_dir
    
    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoint files"""
        checkpoint_path = self.output_root / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True)
        return checkpoint_path
    
    def _save_checkpoint(self) -> None:
        """Save current processor state to checkpoint"""
        checkpoint_file = self.checkpoint_dir / "processor.pkl"
        
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(self, f)
            
            console.print(f"[green]💾 Checkpoint saved: {checkpoint_file}[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠️  Failed to save checkpoint: {e}[/yellow]")
    
    # ========== Document Processing ==========
    
    @staticmethod
    def _calculate_page_ranges(total_pages: int, pages_per_part: int) -> List[PagePart]:
        """
        Calculate page ranges for large file processing
        
        Args:
            total_pages: Total number of pages in the PDF
            pages_per_part: Maximum pages per part
            
        Returns:
            List of PagePart objects
            
        Example:
            >>> _calculate_page_ranges(250, 100)
            [PagePart(1-100) ○, PagePart(101-200) ○, PagePart(201-250) ○]
        """
        parts = []
        start = 1
        while start <= total_pages:
            end = min(start + pages_per_part - 1, total_pages)
            parts.append(PagePart(start_page=start, end_page=end))
            start = end + 1
        return parts

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(10),
        retry=retry_if_exception_type((TimeoutError, Exception)),
        reraise=True
    )
    async def _read_single_part(
        self,
        pdf_path: Path,
        page_range: Optional[tuple] = None
    ) -> PDF:
        """
        Read a single PDF part with MinerU + HeadingCorrection (with automatic retry)
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional page range tuple (start, end) 1-based inclusive
            
        Returns:
            PDF document object
            
        Note:
            - Automatically retries up to 3 times on TimeoutError
            - Waits 10 seconds between retries
            - This is a private method called by read_document()
        """
        # Parse PDF with MinerU
        range_str = f" (pages {page_range[0]}-{page_range[1]})" if page_range else ""
        console.print(f"📄 Parsing PDF with MinerU{range_str}: {pdf_path.name}")

        from zag.readers import MinerUReader
        reader = MinerUReader()
        doc = reader.read(str(pdf_path), page_range=page_range)

        # Apply heading correction
        console.print("  🔧 Correcting headings...")
        corrector = HeadingCorrector(
            llm_uri=f"{LLM_PROVIDER}/{LLM_MODEL}",
            api_key=LLM_API_KEY,
            llm_correction=True
        )
        doc = await corrector.acorrect_document(doc)
        console.print("  ✅ Headings corrected")

        console.print(f"  ✅ Parsed {len(doc.content):,} characters")
        console.print(f"  ✅ Pages: {len(doc.pages)}")
        if doc.metadata.custom:
            console.print(
                f"  ✅ Text items: {doc.metadata.custom.get('text_items_count', 0)}")
            console.print(
                f"  ✅ Table items: {doc.metadata.custom.get('table_items_count', 0)}")

        return doc
    
    async def read_document(
        self,
        pdf_path: Path,
        max_pages_per_part: int = 100
    ) -> PDF:
        """
        Read PDF document with automatic chunking for large files
        
        Automatically splits large files into parts, reads each part with retry,
        applies heading correction, and merges. Supports checkpoint recovery at
        part level for maximum resilience.
        
        Args:
            pdf_path: Path to PDF file
            max_pages_per_part: Maximum pages per part (default: 100)
                               Files with more pages will be split automatically
        
        Returns:
            PDF document object
            
        Note:
            - Small files (≤max_pages_per_part): Read directly
            - Large files: Split into parts, read each with retry, then merge
            - Part-level checkpoint: If part 3/5 fails, restarts from part 3
            - Each part gets HeadingCorrection before merging
            - Final merged document uses consistent doc_id
            - GPU memory: Single-threaded to avoid OOM (one part at a time)
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        # Get total pages
        from pypdf import PdfReader
        reader = PdfReader(str(self.pdf_path))
        total_pages = len(reader.pages)
        console.print(f"📖 Total pages: {total_pages}")
        
        # Initialize parts if not already exist (first run)
        if not self._parts:
            self._parts = self._calculate_page_ranges(total_pages, max_pages_per_part)
        
        console.print(f"✂️  Will read in {len(self._parts)} parts:")
        for i, part in enumerate(self._parts):
            console.print(f"     {i}: {part}")
        
        # Process each part (skip already completed ones)
        for part in self._parts:
            if part.completed:
                console.print(f"[dim]⏭️  Part {part.start_page}-{part.end_page} already completed, skipping[/dim]")
                continue
            
            console.print(f"\n[cyan]--- Reading part {part.start_page}-{part.end_page} ---[/cyan]")
            part_doc = await self._read_single_part(self.pdf_path, (part.start_page, part.end_page))
            
            # Store the processed part document
            part.completed = True
            part.part_doc = part_doc
            self._save_checkpoint()
        
        # All parts processed, now merge them
        console.print(f"\n[cyan]🔗 Merging all {len(self._parts)} parts...[/cyan]")
        merged_doc = None
        for part in self._parts:
            if part.part_doc is None:
                raise RuntimeError(f"Part {part.start_page}-{part.end_page} has no document!")
            
            if merged_doc is None:
                merged_doc = part.part_doc
            else:
                merged_doc = merged_doc + part.part_doc
            console.print(f"  ✅ Merged part {part.start_page}-{part.end_page}: {len(merged_doc.pages)} pages total")
        
        # All parts done
        console.print(f"\n[bold green]✅ All {len(self._parts)} parts completed and merged![/bold green]")
        console.print(f"   Final document: {len(merged_doc.pages)} pages, {len(merged_doc.content):,} characters")
        
        self.document = merged_doc
        self._parts = []  # Clear parts (no longer needed)
        self._save_checkpoint()
        
        # Dump to cache for reuse (best effort, failure is acceptable)
        try:
            console.print(f"\n💾 Caching document...")
            archive_path = self.document.dump(self.cache_dir)
            console.print(f"   ✅ Cached: {archive_path}")
        except Exception as e:
            console.print(f"   ⚠️  Cache failed (non-critical): {e}", style="yellow")
        
        return self.document
    
    def set_document(self, document: PDF) -> None:
        """
        Set the document directly (for merged documents)
        
        Used after merging multiple page range reads into a single document.
        
        Args:
            document: PDF document to set
        """
        self.document = document
        # Use source field for file path (source contains the file path for local files)
        self.pdf_path = Path(document.metadata.source) if document.metadata.source else None

    def set_business_context(
        self,
        custom_metadata: Dict[str, Any]
    ) -> None:
        """
        Set custom metadata for the document

        This method allows business layer to inject custom metadata before splitting.
        All units created from splitting will inherit these values in metadata.custom.

        IMPORTANT: This does NOT override framework IDs (doc_id, unit_id).
        Business IDs should be stored as regular fields in custom_metadata.

        Args:
            custom_metadata: Custom metadata dict (will be stored in unit.metadata.custom)

        Example:
            >>> processor.set_business_context(
            ...     custom_metadata={
            ...         # Business IDs (NOT framework IDs)
            ...         "lender_id": "LENDER_12345",
            ...         "business_doc_id": "MORT_2024_Q1_001",
            ...         "agent_id": "AGENT_789",
            ...         
            ...         # Business attributes
            ...         "department": "Mortgage Lending",
            ...         "product_type": "conventional",
            ...         "region": "US-West",
            ...         "version": "2024.1",
            ...         "status": "active",
            ...         "effective_date": "2024-01-01",
            ...         "owner": "John Doe"
            ...     }
            ... )

        Note:
            - Must be called AFTER read_document() and BEFORE split_document()
            - custom_metadata will be merged into unit.metadata.custom
            - Framework IDs (doc_id, unit_id) remain untouched
            - Framework metadata (page_numbers, keywords, document) are separate
        """
        if self.document is None:
            raise ValueError("No document loaded. Call read_document() first.")

        if not custom_metadata:
            console.print("\n⚠️  No custom metadata provided", style="yellow")
            return

        console.print(f"\n📋 Setting custom metadata:")
        for key, value in custom_metadata.items():
            console.print(f"   - {key}: {value}")

        # Store in document.metadata.custom (will be inherited by units)
        if not self.document.metadata.custom:
            self.document.metadata.custom = {}
        self.document.metadata.custom.update(custom_metadata)

        console.print(f"\n✅ Custom metadata set successfully")

    @checkpoint_cache
    async def split_document(
        self,
        max_chunk_tokens: int = 1200,
        table_max_tokens: int = 1500,
        target_token_size: int = 800,
        export_visualization: bool = True
    ) -> List[Union[TextUnit, TableUnit]]:
        """
        Split document using header-based + recursive merging pipeline

        Args:
            max_chunk_tokens: Max tokens for text chunks (TextSplitter)
            table_max_tokens: Max tokens for table chunks (TableSplitter)
            target_token_size: Target size for merged chunks (RecursiveMergingSplitter)
            export_visualization: Export split visualization to markdown

        Returns:
            List of text/table units

        Raises:
            ValueError: If document not loaded yet

        Note:
            Pipeline: MarkdownHeaderSplitter | TextSplitter | TableSplitter | RecursiveMergingSplitter
            Automatically skips if already completed (checkpoint recovery).
        """
        if self.document is None:
            raise ValueError("No document loaded. Call read_document() first.")

        console.print(f"\n🔪 Splitting document...")
        console.print(f"  Pipeline: MarkdownHeaderSplitter | TextSplitter({max_chunk_tokens}) | "
                      f"TableSplitter({table_max_tokens}) | RecursiveMergingSplitter({target_token_size})")

        # Build pipeline
        pipeline = (
            MarkdownHeaderSplitter()
            | TextSplitter(max_chunk_tokens=max_chunk_tokens)
            | TableSplitter(max_chunk_tokens=table_max_tokens)
            | RecursiveMergingSplitter(target_token_size=target_token_size)
        )

        # Split
        start_time = time.time()
        units = self.document.split(pipeline)
        elapsed = time.time() - start_time

        # Token statistics
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        token_counts = [len(tokenizer.encode(u.content)) for u in units]

        console.print(f"\n✅ Split complete ({elapsed:.2f}s):")
        console.print(f"   - Total units: {len(units)}")
        console.print(
            f"   - Token range: {min(token_counts)}-{max(token_counts)} (avg: {sum(token_counts)//len(token_counts)})")

        # Token distribution
        console.print(f"\n📊 Token distribution:")
        ranges = [
            ("Tiny (<200)", 0, 200),
            ("Small (200-500)", 200, 500),
            ("Medium (500-1000)", 500, 1000),
            ("Large (1000-1500)", 1000, 1500),
            ("Oversized (>1500)", 1500, float('inf')),
        ]

        for label, low, high in ranges:
            count = sum(1 for t in token_counts if low <= t < high)
            if count > 0:
                pct = (count / len(token_counts)) * 100
                bar = "█" * int(pct / 2)
                console.print(
                    f"   {label:<20} {count:>4} ({pct:>5.1f}%) {bar}")

        # Check oversized
        oversized = [(i, t) for i, t in enumerate(token_counts) if t > 1500]
        if oversized:
            console.print(
                f"\n⚠️  {len(oversized)} oversized units (>1500 tokens)", style="yellow")

        # Export visualization
        if export_visualization:
            self._export_split_visualization(units, token_counts, tokenizer)

        self.units = units
        
        return units

    def _export_split_visualization(
        self,
        units: List[Union[TextUnit, TableUnit]],
        token_counts: List[int],
        tokenizer
    ):
        """Export split visualization to markdown file"""
        viz_dir = self.split_dir / "visualization"
        viz_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = self.document.metadata.file_name.rsplit(
            '.', 1)[0] if self.document else "document"
        viz_file = viz_dir / f"{doc_name}_split_{timestamp}.md"

        with open(viz_file, 'w', encoding='utf-8') as f:
            f.write(f"# Document Splitting Visualization\n\n")
            f.write(f"**Total Units**: {len(units)}\n\n")
            f.write(
                f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(
                f"**Token Range**: {min(token_counts)}-{max(token_counts)} tokens\n\n")
            f.write(
                f"**Average**: {sum(token_counts)//len(token_counts)} tokens\n\n")
            f.write("---\n\n")

            for i, unit in enumerate(units):
                tokens = len(tokenizer.encode(unit.content))

                f.write(f"\n\n{'🔷' * 50}\n\n")
                f.write(f"## 📦 Unit {i} | {tokens} tokens\n\n")

                if hasattr(unit, 'metadata') and unit.metadata:
                    if hasattr(unit.metadata, 'context_path') and unit.metadata.context_path:
                        f.write(
                            f"**Context**: {unit.metadata.context_path}\n\n")

                preview = unit.content.strip()[:100].replace('\n', ' ')
                f.write(f"**Preview**: {preview}...\n\n")

                if tokens > 1500:
                    f.write(f"⚠️ **OVERSIZED** ({tokens} tokens)\n\n")
                elif tokens >= 1000:
                    f.write(f"📊 **LARGE** ({tokens} tokens)\n\n")

                f.write(f"---\n\n")
                f.write(unit.content)
                f.write("\n\n")

        console.print(f"\n💾 Visualization exported: {viz_file.name}")

    @checkpoint_cache
    async def process_tables(
        self,
        llm_uri: str,
        api_key: Optional[str] = None
    ) -> List[Union[TextUnit, TableUnit]]:
        """
        Process tables with three stages:
        
        Stage 1 (Original): Extract embedding_content for TextUnits
                           (replace markdown tables with natural language)
        Stage 2 (New):     Parse data-critical TableUnits from TextUnits
        Stage 3 (New):     Enrich TableUnits with caption and embedding_content
        
        Args:
            llm_uri: LLM URI (e.g., "ollama/qwen2.5:7b")
            api_key: API key if needed

        Returns:
            Updated units list (original TextUnits + new TableUnits)

        Raises:
            ValueError: If units not available

        Note:
            - Stage 1: Updates unit.embedding_content for TextUnits
            - Stage 2: Parses new TableUnits from TextUnits
            - Stage 3: Enriches TableUnits with caption and embedding_content
            - Final output: merged list of TextUnits + TableUnits
            - Automatically skips if already completed (checkpoint recovery).
        """
        if not self.units:
            raise ValueError(
                "No units available. Call split_document() first.")

        console.print(f"\n📊 Processing tables with LLM: {llm_uri}")
        
        # ========== Stage 1: Process TextUnit embedding_content ==========
        console.print(f"  Stage 1: Processing TextUnit embedding_content...")
        summarizer = TableSummarizer(llm_uri=llm_uri, api_key=api_key)
        results = await summarizer.aextract(self.units)
        
        # Apply results to units
        for unit, metadata in zip(self.units, results):
            if metadata.get("embedding_content"):
                unit.embedding_content = metadata["embedding_content"]
        
        units_with_embedding = sum(1 for u in self.units if hasattr(
            u, 'embedding_content') and u.embedding_content)
        console.print(f"  ✅ Processed {len(self.units)} TextUnits")
        console.print(
            f"  ✅ Units with embedding_content: {units_with_embedding}")
        
        # ========== Stage 2: Parse TableUnits from existing document ==========
        console.print(f"\n  Stage 2: Parsing tables from existing document...")
        
        from zag.parsers import TableParser
        from zag.schemas import UnitMetadata
        
        # Use existing document (already read by read_document)
        parser = TableParser()
        
        # Prepare unit metadata with business context (same as TextUnit)
        unit_metadata = UnitMetadata(
            document=self.document.metadata.model_dump_deep()
        )
        
        # Inject business custom metadata from original document (same as PDF.split() does for TextUnit)
        if self.document and self.document.metadata and self.document.metadata.custom:
            unit_metadata.custom.update(self.document.metadata.custom)
        
        table_units = parser.parse(
            text=self.document.content,
            metadata=unit_metadata,
            doc_id=self.document.doc_id,
        )
        console.print(f"  ✅ Total tables parsed: {len(table_units)}")
        
        # Infer page numbers for table units
        if table_units:
            from zag.utils.page_inference import infer_page_numbers
            infer_page_numbers(
                table_units,
                self.document.pages,
                full_content=self.document.content
            )
            tables_with_pages = sum(1 for t in table_units if t.metadata.page_numbers)
            console.print(f"  📄 Tables with page_numbers: {tables_with_pages}/{len(table_units)}")
        
        # ========== Stage 3: Enrich TableUnits (LLM-based) ==========
        if table_units:
            console.print(f"\n  Stage 3: Enriching TableUnits with LLM...")
            from zag.extractors import TableEnrichMode
            
            enricher = TableEnricher(llm_uri=llm_uri, api_key=api_key)
            enriched_tables = await enricher.aextract(
                table_units,
                mode=TableEnrichMode.CRITICAL_ONLY  # Judge all, enrich only critical tables
            )
            console.print(f"  ✅ Enriched {len(enriched_tables)} TableUnits")
            
            # Count critical tables
            critical_count = sum(
                1 for u in enriched_tables 
                if u.metadata.custom.get("table", {}).get("is_data_critical", False)
            )
            console.print(f"  📊 {critical_count}/{len(enriched_tables)} tables are data-critical")
            
            # Show sample
            console.print("\n  Sample enriched tables (first 3):")
            for i, table_unit in enumerate(enriched_tables[:3], 1):
                meta_table = (table_unit.metadata.custom or {}).get("table", {})
                is_critical = meta_table.get("is_data_critical", False)
                console.print(f"    {i}. shape: {table_unit.df.shape}, critical: {is_critical}")
                if is_critical and table_unit.caption:
                    console.print(f"       caption: {table_unit.caption[:60]}...")
            
            # Use enriched tables
            table_units = enriched_tables
        else:
            console.print(f"\n  ⚠️  No tables found in document")
        
        # ========== Merge units ==========
        original_count = len(self.units)
        self.units = self.units + table_units
        console.print(f"\n  📦 Total units: {original_count} TextUnit + {len(table_units)} TableUnit = {len(self.units)}")
        
        return self.units

    @checkpoint_cache
    async def extract_metadata(
        self,
        llm_uri: str,
        api_key: Optional[str] = None,
        num_keywords: int = 5
    ) -> List[Union[TextUnit, TableUnit]]:
        """
        Extract keywords metadata using LLM

        Args:
            llm_uri: LLM URI (e.g., "ollama/qwen2.5:7b")
            api_key: API key if needed
            num_keywords: Number of keywords to extract

        Returns:
            Updated units with keywords in metadata.keywords

        Raises:
            ValueError: If units not available

        Note:
            - Updates unit.metadata.keywords
            - For large files (>500 pages), skips extraction to save time
            - Automatically skips if already completed (checkpoint recovery).
        """
        if not self.units:
            raise ValueError(
                "No units available. Call split_document() first.")

        # Skip keyword extraction for large files
        total_pages = len(self.document.pages) if self.document else 0
        if total_pages > 500:
            console.print(f"\n⚠️  Large file ({total_pages} pages), skipping keyword extraction")
            console.print(f"  💡 Keywords can be added later via separate task")
            # Early return (decorator will still mark as completed)
            return self.units

        # Extract keywords
        console.print(f"\n🏷️  Extracting keywords with LLM: {llm_uri}")
        extractor = KeywordExtractor(
            llm_uri=llm_uri,
            api_key=api_key,
            num_keywords=num_keywords
        )

        await extractor.aextract(self.units)

        console.print(f"  ✅ Extracted keywords for {len(self.units)} units")

        # Show examples
        console.print("\n  Example keywords (first 3 units):")
        for i, unit in enumerate(self.units[:3], 1):
            keywords = unit.metadata.keywords or []
            console.print(f"    {i}. {keywords}")
        
        return self.units

    # ========== Indexing ==========

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    @checkpoint_cache
    async def build_vector_index(
        self,
        embedding_uri: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_grpc_port: int = 6334,
        collection_name: str = "debug_collection",
        api_key: Optional[str] = None
    ) -> VectorIndexer:
        """
        Build vector index with Qdrant (with automatic retry)

        Args:
            embedding_uri: Embedding model URI (e.g., "ollama/jina-embeddings-v2-base-en:latest")
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant HTTP port (for REST API)
            qdrant_grpc_port: Qdrant gRPC port (for high performance)
            collection_name: Collection name
            api_key: API key for embedding service (if needed)

        Returns:
            VectorIndexer instance

        Raises:
            ValueError: If units not available
            
        Note:
            - Retries up to 10 times with exponential backoff (2s, 4s, 8s, ..., max 60s)
            - Must succeed, will keep retrying until success
            - Automatically skips if already completed (checkpoint recovery).
        """
        if not self.units:
            raise ValueError(
                "No units available. Call split_document() first.")

        console.print(f"\n📚 Building vector index...")
        console.print(f"  Embedding: {embedding_uri}")
        console.print(
            f"  Storage: Qdrant at {qdrant_host}:{qdrant_port} (gRPC: {qdrant_grpc_port})")
        console.print(f"  Collection: {collection_name}")

        embedder = Embedder(embedding_uri, api_key=api_key)
        vector_store = QdrantVectorStore.server(
            host=qdrant_host,
            port=qdrant_port,
            grpc_port=qdrant_grpc_port,
            prefer_grpc=True,
            timeout=300,  # 5 minutes timeout for large batch operations
            collection_name=collection_name,
            embedder=embedder
        )

        vector_indexer = VectorIndexer(vector_store=vector_store)

        # Clear existing data for this document before indexing
        if self.document and self.document.doc_id:
            console.print(f"  🗑️  Clearing old classic data for doc_id: {self.document.doc_id}")
            await vector_store.aremove({
                "doc_id": self.document.doc_id,
                "metadata.custom.mode": ProcessingMode.CLASSIC
            })

        # Use upsert instead of add for safety (can retry without errors)
        await vector_indexer.aupsert(self.units)

        console.print(
            f"  ✅ Vector index built: {vector_indexer.count()} units")

        return vector_indexer

    @checkpoint_cache
    async def build_fulltext_index(
        self,
        meilisearch_url: str,
        index_name: str = "mortgage_guidelines",
        primary_key: str = "unit_id"
    ) -> Optional[FullTextIndexer]:
        """
        Build fulltext index with Meilisearch (best effort, failure is acceptable)

        Args:
            meilisearch_url: Meilisearch server URL
            index_name: Index name
            primary_key: Primary key field

        Returns:
            FullTextIndexer instance if successful, None if failed

        Raises:
            ValueError: If units not available
            
        Note:
            - This is a best-effort operation
            - If fails, logs warning and returns None
            - Does not block the pipeline
            - Automatically skips if already completed (checkpoint recovery).
        """
        if not self.units:
            raise ValueError(
                "No units available. Call split_document() first.")

        console.print(f"\n📚 Building fulltext index...")
        console.print(f"  Meilisearch: {meilisearch_url}")
        console.print(f"  Index: {index_name}")

        try:
            fulltext_indexer = FullTextIndexer(
                url=meilisearch_url,
                index_name=index_name,
                primary_key=primary_key
            )

            # Note: Meilisearch uses upsert by default, so we don't need to delete old data
            # The units with same unit_id will be automatically replaced

            fulltext_indexer.configure_settings(
                searchable_attributes=["content", "context_path"],
                filterable_attributes=["unit_type", "source_doc_id"],
                sortable_attributes=["created_at"],
            )

            fulltext_indexer.add(self.units)

            console.print(
                f"  ✅ Fulltext index built: {fulltext_indexer.count()} units")

            return fulltext_indexer
            
        except Exception as e:
            console.print(f"  ⚠️  Fulltext index failed (non-critical): {e}", style="yellow")
            console.print(f"  💡 You can rebuild fulltext index later", style="dim")
            # Return None (decorator will still mark as completed)
            return None
