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

# Zag imports
from zag.readers.docling import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
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
        self.vector_indexer: Optional[VectorIndexer] = None
        self.fulltext_indexer: Optional[FullTextIndexer] = None
        
        # Page range being processed (for large file processing)
        self._current_page_range: Optional[tuple] = None  # (start, end) 1-based inclusive

        # Lazy-initialized subdirectories
        self._raw_dir: Optional[Path] = None
        self._split_dir: Optional[Path] = None
        self._tables_dir: Optional[Path] = None
        self._metadata_dir: Optional[Path] = None
        self._indices_dir: Optional[Path] = None

    @property
    def raw_dir(self) -> Path:
        """Directory for raw markdown cache"""
        if self._raw_dir is None:
            self._raw_dir = self.output_root / "raw"
            self._raw_dir.mkdir(exist_ok=True)
        return self._raw_dir

    @property
    def split_dir(self) -> Path:
        """Directory for split visualization"""
        if self._split_dir is None:
            self._split_dir = self.output_root / "split"
            self._split_dir.mkdir(exist_ok=True)
        return self._split_dir

    @property
    def tables_dir(self) -> Path:
        """Directory for table processing cache"""
        if self._tables_dir is None:
            self._tables_dir = self.output_root / "tables"
            self._tables_dir.mkdir(exist_ok=True)
        return self._tables_dir

    @property
    def metadata_dir(self) -> Path:
        """Directory for metadata extraction cache"""
        if self._metadata_dir is None:
            self._metadata_dir = self.output_root / "metadata"
            self._metadata_dir.mkdir(exist_ok=True)
        return self._metadata_dir

    @property
    def indices_dir(self) -> Path:
        """Directory for index metadata"""
        if self._indices_dir is None:
            self._indices_dir = self.output_root / "indices"
            self._indices_dir.mkdir(exist_ok=True)
        return self._indices_dir
    
    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoint files"""
        checkpoint_path = self.output_root / "checkpoints"
        checkpoint_path.mkdir(exist_ok=True)
        return checkpoint_path
    
    def save_checkpoint(self, stage_name: str) -> None:
        """
        Save units to checkpoint file
        
        Args:
            stage_name: Name of the pipeline stage (e.g., 'split', 'tables', 'metadata')
        """
        if not self.units:
            console.print(f"[dim]No units to save for checkpoint: {stage_name}[/dim]")
            return
        
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{stage_name}.pkl"
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(self.units, f)
            console.print(f"[green]✓ Checkpoint saved: {checkpoint_file.name} ({len(self.units)} units)[/green]")
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to save checkpoint {stage_name}: {e}[/yellow]")
    
    def load_checkpoint(self, stage_name: str) -> bool:
        """
        Load units from checkpoint file
        
        Args:
            stage_name: Name of the pipeline stage
        
        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{stage_name}.pkl"
        if not checkpoint_file.exists():
            return False
        
        try:
            with open(checkpoint_file, 'rb') as f:
                self.units = pickle.load(f)
            console.print(f"[cyan]✓ Checkpoint loaded: {checkpoint_file.name} ({len(self.units)} units)[/cyan]")
            return True
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to load checkpoint {stage_name}: {e}[/yellow]")
            return False

    # ========== Document Processing ==========

    async def read_document(
        self,
        pdf_path: Path,
        use_gpu: bool = True,
        num_threads: int = 8,
        page_range: Optional[tuple] = None  # (start, end) 1-based inclusive
    ) -> PDF:
        """
        Read a single PDF document

        Args:
            pdf_path: Path to PDF file
            use_gpu: Enable GPU acceleration for Docling
            num_threads: Number of threads for parsing
            page_range: Optional page range tuple (start, end) 1-based inclusive.
                        If provided, only reads the specified page range.
                        Example: (1, 100) reads pages 1-100.

        Returns:
            PDF document object
            
        Note:
            When processing large files in chunks:
            1. Each chunk is read separately with page_range
            2. HeadingCorrection is applied to each chunk
            3. Chunks are merged using PDF.__add__()
            4. The merged document maintains consistent doc_id
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        # Store page range for later use (e.g., process_tables)
        self._current_page_range = page_range

        # Parse PDF
        range_str = f" (pages {page_range[0]}-{page_range[1]})" if page_range else ""
        console.print(f"📄 Parsing PDF{range_str}: {self.pdf_path.name}")

        # Configure Docling
        pdf_options = PdfPipelineOptions()
        pdf_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
        )

        reader = DoclingReader(pdf_pipeline_options=pdf_options)
        doc = reader.read(str(self.pdf_path), page_range=page_range)

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

        self.document = doc
        return doc
    
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
        
        # Save checkpoint
        self.save_checkpoint('split')
        
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
        
        # ========== Stage 2: Parse TableUnits from PDF (re-read with MinerU) ==========
        console.print(f"\n  Stage 2: Parsing tables with MinerU (high-quality table extraction)...")
        
        # Re-read PDF with MinerU for better table quality
        from zag.readers import MinerUReader
        from zag.parsers import TableParser
        from zag.schemas import UnitMetadata
        from ..config import MAX_PAGES_PER_PART
        
        # Get full page range from merged document
        if self.document and self.document.pages:
            page_numbers = [p.page_number for p in self.document.pages]
            total_pages = len(page_numbers)
            start_page = min(page_numbers)
            end_page = max(page_numbers)
        else:
            # Fallback to current page range
            if self._current_page_range:
                start_page, end_page = self._current_page_range
                total_pages = end_page - start_page + 1
            else:
                raise ValueError("No page range information available")
        
        # Calculate page ranges for chunked reading (same logic as classic.py)
        if total_pages > MAX_PAGES_PER_PART:
            page_ranges = []
            s = start_page
            while s <= end_page:
                e = min(s + MAX_PAGES_PER_PART - 1, end_page)
                page_ranges.append((s, e))
                s = e + 1
            console.print(f"  📄 Large document, reading in {len(page_ranges)} parts")
        else:
            page_ranges = [(start_page, end_page)]
        
        # Read in parts and collect all tables
        mineru_reader = MinerUReader()
        merged_pdf: Optional[PDF] = None
        
        for part_idx, (part_start, part_end) in enumerate(page_ranges, 1):
            console.print(f"  ⏳ Reading part {part_idx}/{len(page_ranges)}: pages {part_start}-{part_end}...")
            pdf_mineru = mineru_reader.read(self.pdf_path, page_range=(part_start, part_end))
            
            # Merge PDFs
            if merged_pdf is None:
                merged_pdf = pdf_mineru
            else:
                merged_pdf = merged_pdf + pdf_mineru
        
        # Dump merged PDF to cache for reuse
        if merged_pdf:
            cache_dir = Path.home() / ".zag" / "cache" / "readers" / "mineru"
            archive_path = merged_pdf.dump(cache_dir)
            console.print(f"  💾 Cached: {archive_path}")
        
        # Parse tables from merged PDF (so span is relative to merged content)
        console.print(f"  ⏳ Parsing tables from merged content...")
        parser = TableParser()
        unit_metadata = UnitMetadata(
            document=merged_pdf.metadata.model_dump_deep()
        )
        table_units = parser.parse(
            text=merged_pdf.content,
            metadata=unit_metadata,
            doc_id=merged_pdf.doc_id,
        )
        console.print(f"  ✅ Total tables parsed: {len(table_units)}")
        
        # Infer page numbers for table units
        if table_units:
            from zag.utils.page_inference import infer_page_numbers
            infer_page_numbers(
                table_units,
                merged_pdf.pages,
                full_content=merged_pdf.content
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
        
        # Save checkpoint
        self.save_checkpoint('tables')

        return self.units

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
        """
        if not self.units:
            raise ValueError(
                "No units available. Call split_document() first.")

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
        
        # Save checkpoint
        self.save_checkpoint('metadata')

        return self.units

    # ========== Indexing ==========

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
        Build vector index with Qdrant

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

        self.vector_indexer = vector_indexer
        return vector_indexer

    async def build_fulltext_index(
        self,
        meilisearch_url: str,
        index_name: str = "mortgage_guidelines",
        primary_key: str = "unit_id"
    ) -> FullTextIndexer:
        """
        Build fulltext index with Meilisearch

        Args:
            meilisearch_url: Meilisearch server URL
            index_name: Index name
            primary_key: Primary key field

        Returns:
            FullTextIndexer instance

        Raises:
            ValueError: If units not available
        """
        if not self.units:
            raise ValueError(
                "No units available. Call split_document() first.")

        console.print(f"\n📚 Building fulltext index...")
        console.print(f"  Meilisearch: {meilisearch_url}")
        console.print(f"  Index: {index_name}")

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

        self.fulltext_indexer = fulltext_indexer
        return fulltext_indexer
