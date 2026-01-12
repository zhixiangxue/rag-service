#!/usr/bin/env python3
"""
Mortgage Document Processor

A production-ready document processor for mortgage guidelines.
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
from zag.extractors import TableExtractor, KeywordExtractor
from zag.embedders import Embedder
from zag.storages.vector import ChromaVectorStore
from zag.indexers import VectorIndexer, FullTextIndexer
from zag.retrievers import VectorRetriever, FullTextRetriever, QueryFusionRetriever, FusionMode
from zag.postprocessors import SimilarityFilter, Deduplicator, ContextAugmentor, ChainPostprocessor
from zag.schemas.base import DocumentMetadata, Page, UnitType
from zag.schemas.pdf import PDF
from zag.schemas.unit import TextUnit, TableUnit
from zag.utils.hash import calculate_file_hash

console = Console()


class MortgageDocumentProcessor:
    """
    Document processor for mortgage guidelines
    
    Design Principles:
    - Single file processing (not directory batch)
    - Independent methods for Celery tasks
    - Explicit parameter passing
    - State persistence in instance
    - Full caching with validation
    
    Example Usage:
        # Full pipeline
        processor = MortgageDocumentProcessor(output_root=Path("output/doc1"))
        await processor.read_document(pdf_path=Path("doc.pdf"))
        await processor.split_document(max_chunk_tokens=1200)
        await processor.process_tables(llm_uri="ollama/qwen2.5:7b")
        await processor.extract_metadata(llm_uri="ollama/qwen2.5:7b")
        await processor.build_vector_index(
            embedding_uri="ollama/jina-embeddings-v2-base-en:latest",
            chroma_persist_dir=Path("/data/chroma")
        )
        
        # Search only (stateless)
        results = await processor.retrieve_fusion(
            query="FHA loan requirements",
            embedding_uri="ollama/jina-embeddings-v2-base-en:latest",
            chroma_persist_dir=Path("/data/chroma"),
            meilisearch_url="http://localhost:7700"
        )
    """
    
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
    
    # ========== Document Processing ==========
    
    async def read_document(
        self,
        pdf_path: Path,
        use_gpu: bool = True,
        num_threads: int = 8,
        force_reparse: bool = False,
        quality_threshold: float = 90.0
    ) -> PDF:
        """
        Read a single PDF document with caching and quality validation
        
        Args:
            pdf_path: Path to PDF file
            use_gpu: Enable GPU acceleration for Docling
            num_threads: Number of threads for parsing
            force_reparse: Force re-parsing even if cache exists
            quality_threshold: Minimum quality score for cache (0-100)
            
        Returns:
            PDF document object
            
        Note:
            - Cache stored as markdown in raw_dir
            - Quality validated using validate_cache_quality()
            - Low-quality cache automatically regenerated
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        markdown_path = self.raw_dir / f"{self.pdf_path.stem}.md"
        
        # Check cache quality
        use_cache = False
        if markdown_path.exists() and not force_reparse:
            console.print(f"🔍 Checking cache quality: {self.pdf_path.name}")
            
            # Import quality validator
            try:
                from validate_conversion import validate_cache_quality
                is_valid = validate_cache_quality(
                    self.pdf_path, 
                    markdown_path, 
                    threshold=quality_threshold, 
                    verbose=False
                )
                
                if is_valid:
                    console.print(f"  ✅ Cache quality OK (>= {quality_threshold}), using cache")
                    use_cache = True
                else:
                    console.print(f"  ⚠️  Cache quality low (< {quality_threshold}), re-parsing PDF")
                    markdown_path.unlink()
            except ImportError:
                console.print(f"  ⚠️  validate_conversion not found, skipping quality check")
                use_cache = True
        
        if use_cache:
            # Load from cache
            console.print(f"📄 Loading from cache: {markdown_path.name}")
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Calculate file hash (required for metadata)
            file_hash = calculate_file_hash(self.pdf_path)
            
            # Construct minimal metadata
            metadata = DocumentMetadata(
                source=str(self.pdf_path),
                source_type="local",
                file_type="pdf",
                file_name=self.pdf_path.name,
                file_size=self.pdf_path.stat().st_size,
                file_extension=".pdf",
                md5=file_hash,  # Required field
                content_length=len(content),
                reader_name="DoclingReader (cached)",
                custom={
                    'cached': True,
                    'cache_file': str(markdown_path),
                    'quality_validated': True
                }
            )
            
            # Create minimal Page object
            pages = [Page(
                page_number=1,
                content={'texts': [], 'tables': [], 'pictures': []},
                metadata={'cached': True}
            )]
            
            doc = PDF(
                content=content,
                metadata=metadata,
                pages=pages
            )
            
            console.print(f"  ✅ Loaded {len(content):,} characters from cache")
        else:
            # Parse PDF
            console.print(f"📄 Parsing PDF: {self.pdf_path.name}")
            
            # Configure Docling
            pdf_options = PdfPipelineOptions()
            pdf_options.accelerator_options = AcceleratorOptions(
                num_threads=num_threads,
                device=AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
            )
            
            reader = DoclingReader(pdf_pipeline_options=pdf_options)
            doc = reader.read(str(self.pdf_path))
            
            console.print(f"  ✅ Parsed {len(doc.content):,} characters")
            console.print(f"  ✅ Pages: {len(doc.pages)}")
            if doc.metadata.custom:
                console.print(f"  ✅ Text items: {doc.metadata.custom.get('text_items_count', 0)}")
                console.print(f"  ✅ Table items: {doc.metadata.custom.get('table_items_count', 0)}")
            
            # Save cache
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(doc.content)
            console.print(f"  ✅ Cache saved: {markdown_path.name}")
        
        self.document = doc
        return doc
    
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
        console.print(f"   - Token range: {min(token_counts)}-{max(token_counts)} (avg: {sum(token_counts)//len(token_counts)})")
        
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
                console.print(f"   {label:<20} {count:>4} ({pct:>5.1f}%) {bar}")
        
        # Check oversized
        oversized = [(i, t) for i, t in enumerate(token_counts) if t > 1500]
        if oversized:
            console.print(f"\n⚠️  {len(oversized)} oversized units (>1500 tokens)", style="yellow")
        
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
        doc_name = self.document.metadata.file_name.rsplit('.', 1)[0] if self.document else "document"
        viz_file = viz_dir / f"{doc_name}_split_{timestamp}.md"
        
        with open(viz_file, 'w', encoding='utf-8') as f:
            f.write(f"# Document Splitting Visualization\n\n")
            f.write(f"**Total Units**: {len(units)}\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Token Range**: {min(token_counts)}-{max(token_counts)} tokens\n\n")
            f.write(f"**Average**: {sum(token_counts)//len(token_counts)} tokens\n\n")
            f.write("---\n\n")
            
            for i, unit in enumerate(units):
                tokens = len(tokenizer.encode(unit.content))
                
                f.write(f"\n\n{'🔷' * 50}\n\n")
                f.write(f"## 📦 Unit {i} | {tokens} tokens\n\n")
                
                if hasattr(unit, 'metadata') and unit.metadata:
                    if hasattr(unit.metadata, 'context_path') and unit.metadata.context_path:
                        f.write(f"**Context**: {unit.metadata.context_path}\n\n")
                
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
        api_key: Optional[str] = None,
        use_cache: bool = True
    ) -> List[Union[TextUnit, TableUnit]]:
        """
        Extract table summaries using LLM
        
        Args:
            llm_uri: LLM URI (e.g., "ollama/qwen2.5:7b")
            api_key: API key if needed
            use_cache: Use cached results if available
            
        Returns:
            Updated units with table metadata (embedding_content)
            
        Raises:
            ValueError: If units not available
            
        Note:
            - Cache stored as JSON in tables_dir
            - Updates unit.embedding_content for tables
        """
        if not self.units:
            raise ValueError("No units available. Call split_document() first.")
        
        units_json_path = self.tables_dir / "units_after_table_processing.json"
        
        # Check cache
        if use_cache and units_json_path.exists():
            console.print(f"\n📊 Loading table processing cache: {units_json_path.name}")
            
            with open(units_json_path, 'r', encoding='utf-8') as f:
                units_data = json.load(f)
            
            # Rebuild units
            units = []
            for data in units_data:
                unit_type = data.get('unit_type', 'TEXT')
                if unit_type == 'TABLE' or unit_type == UnitType.TABLE.value:
                    units.append(TableUnit(**data))
                else:
                    units.append(TextUnit(**data))
            
            console.print(f"  ✅ Loaded {len(units)} units from cache")
            units_with_embedding = sum(1 for u in units if hasattr(u, 'embedding_content') and u.embedding_content)
            console.print(f"  ✅ Units with embedding_content: {units_with_embedding}")
            
            self.units = units
            return units
        
        # Process tables
        console.print(f"\n📊 Processing tables with LLM: {llm_uri}")
        extractor = TableExtractor(llm_uri=llm_uri, api_key=api_key)
        
        results = await extractor.aextract(self.units)
        
        # Update embedding_content
        for unit, metadata in zip(self.units, results):
            if metadata.get("embedding_content"):
                unit.embedding_content = metadata["embedding_content"]
        
        console.print(f"  ✅ Processed {len(self.units)} units")
        
        # Save cache
        units_data = [unit.model_dump(mode='json') for unit in self.units]
        with open(units_json_path, 'w', encoding='utf-8') as f:
            json.dump(units_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"  ✅ Cache saved: {units_json_path.name}")
        
        units_with_embedding = sum(1 for u in self.units if hasattr(u, 'embedding_content') and u.embedding_content)
        console.print(f"  ✅ Units with embedding_content: {units_with_embedding}")
        
        return self.units
    
    async def extract_metadata(
        self,
        llm_uri: str,
        api_key: Optional[str] = None,
        num_keywords: int = 5,
        use_cache: bool = True
    ) -> List[Union[TextUnit, TableUnit]]:
        """
        Extract keywords metadata using LLM
        
        Args:
            llm_uri: LLM URI (e.g., "ollama/qwen2.5:7b")
            api_key: API key if needed
            num_keywords: Number of keywords to extract
            use_cache: Use cached results if available
            
        Returns:
            Updated units with keywords in metadata.custom
            
        Raises:
            ValueError: If units not available
            
        Note:
            - Cache stored as JSON in metadata_dir
            - Updates unit.metadata.custom['excerpt_keywords']
        """
        if not self.units:
            raise ValueError("No units available. Call split_document() first.")
        
        units_json_path = self.metadata_dir / "units_after_keyword_extraction.json"
        
        # Check cache
        if use_cache and units_json_path.exists():
            console.print(f"\n🏷️  Loading keyword extraction cache: {units_json_path.name}")
            
            with open(units_json_path, 'r', encoding='utf-8') as f:
                units_data = json.load(f)
            
            # Rebuild units
            units = []
            for data in units_data:
                unit_type = data.get('unit_type', 'TEXT')
                if unit_type == 'TABLE' or unit_type == UnitType.TABLE.value:
                    units.append(TableUnit(**data))
                else:
                    units.append(TextUnit(**data))
            
            console.print(f"  ✅ Loaded {len(units)} units from cache")
            units_with_keywords = sum(1 for u in units if u.metadata.custom.get('excerpt_keywords'))
            console.print(f"  ✅ Units with keywords: {units_with_keywords}")
            
            self.units = units
            return units
        
        # Extract keywords
        console.print(f"\n🏷️  Extracting keywords with LLM: {llm_uri}")
        extractor = KeywordExtractor(
            llm_uri=llm_uri,
            api_key=api_key,
            num_keywords=num_keywords
        )
        
        results = await extractor.aextract(self.units)
        
        # Update metadata
        for unit, metadata in zip(self.units, results):
            unit.metadata.custom.update(metadata)
        
        console.print(f"  ✅ Extracted keywords for {len(self.units)} units")
        
        # Show examples
        console.print("\n  Example keywords (first 3 units):")
        for i, unit in enumerate(self.units[:3], 1):
            keywords = unit.metadata.custom.get("excerpt_keywords", [])
            console.print(f"    {i}. {keywords}")
        
        # Save cache
        units_data = [unit.model_dump(mode='json') for unit in self.units]
        with open(units_json_path, 'w', encoding='utf-8') as f:
            json.dump(units_data, f, ensure_ascii=False, indent=2)
        
        console.print(f"\n  ✅ Cache saved: {units_json_path.name}")
        
        return self.units
    
    # ========== Indexing ==========
    
    async def build_vector_index(
        self,
        embedding_uri: str,
        chroma_persist_dir: Path,
        collection_name: str = "mortgage_guidelines",
        clear_existing: bool = True
    ) -> VectorIndexer:
        """
        Build vector index with ChromaDB
        
        Args:
            embedding_uri: Embedding model URI (e.g., "ollama/jina-embeddings-v2-base-en:latest")
            chroma_persist_dir: ChromaDB persistence directory
            collection_name: Collection name
            clear_existing: Clear existing index before building
            
        Returns:
            VectorIndexer instance
            
        Raises:
            ValueError: If units not available
        """
        if not self.units:
            raise ValueError("No units available. Call split_document() first.")
        
        console.print(f"\n📚 Building vector index...")
        console.print(f"  Embedding: {embedding_uri}")
        console.print(f"  Storage: {chroma_persist_dir}")
        console.print(f"  Collection: {collection_name}")
        
        embedder = Embedder(embedding_uri)
        vector_store = ChromaVectorStore.local(
            path=str(chroma_persist_dir),
            collection_name=collection_name,
            embedder=embedder
        )
        
        vector_indexer = VectorIndexer(vector_store=vector_store)
        
        if clear_existing:
            await vector_indexer.aclear()
        
        await vector_indexer.aadd(self.units)
        
        console.print(f"  ✅ Vector index built: {vector_indexer.count()} units")
        
        self.vector_indexer = vector_indexer
        return vector_indexer
    
    async def build_fulltext_index(
        self,
        meilisearch_url: str,
        index_name: str = "mortgage_guidelines",
        primary_key: str = "unit_id",
        clear_existing: bool = True
    ) -> FullTextIndexer:
        """
        Build fulltext index with Meilisearch
        
        Args:
            meilisearch_url: Meilisearch server URL
            index_name: Index name
            primary_key: Primary key field
            clear_existing: Clear existing index before building
            
        Returns:
            FullTextIndexer instance
            
        Raises:
            ValueError: If units not available
        """
        if not self.units:
            raise ValueError("No units available. Call split_document() first.")
        
        console.print(f"\n📚 Building fulltext index...")
        console.print(f"  Meilisearch: {meilisearch_url}")
        console.print(f"  Index: {index_name}")
        
        fulltext_indexer = FullTextIndexer(
            url=meilisearch_url,
            index_name=index_name,
            primary_key=primary_key
        )
        
        if clear_existing:
            fulltext_indexer.clear()
        
        fulltext_indexer.configure_settings(
            searchable_attributes=["content", "context_path"],
            filterable_attributes=["unit_type", "source_doc_id"],
            sortable_attributes=["created_at"],
        )
        
        fulltext_indexer.add(self.units)
        
        console.print(f"  ✅ Fulltext index built: {fulltext_indexer.count()} units")
        
        self.fulltext_indexer = fulltext_indexer
        return fulltext_indexer
    
    # ========== Retrieval (Stateless) ==========
    
    async def retrieve_vector(
        self,
        query: str,
        embedding_uri: str,
        chroma_persist_dir: Path,
        collection_name: str = "mortgage_guidelines",
        top_k: int = 5
    ) -> List[TextUnit]:
        """
        Standalone vector retrieval (loads index from disk)
        
        Args:
            query: Search query
            embedding_uri: Embedding model URI
            chroma_persist_dir: ChromaDB persistence directory
            collection_name: Collection name
            top_k: Number of results
            
        Returns:
            List of retrieved units
            
        Note:
            This method is stateless - doesn't require prior indexing via this instance
        """
        embedder = Embedder(embedding_uri)
        vector_store = ChromaVectorStore.local(
            path=str(chroma_persist_dir),
            collection_name=collection_name,
            embedder=embedder
        )
        
        retriever = VectorRetriever(vector_store=vector_store, top_k=top_k)
        results = retriever.retrieve(query)
        
        return results
    
    async def retrieve_fulltext(
        self,
        query: str,
        meilisearch_url: str,
        index_name: str = "mortgage_guidelines",
        top_k: int = 5
    ) -> List[TextUnit]:
        """
        Standalone fulltext retrieval
        
        Args:
            query: Search query
            meilisearch_url: Meilisearch server URL
            index_name: Index name
            top_k: Number of results
            
        Returns:
            List of retrieved units
        """
        retriever = FullTextRetriever(
            url=meilisearch_url,
            index_name=index_name,
            top_k=top_k
        )
        
        results = retriever.retrieve(query)
        return results
    
    async def retrieve_fusion(
        self,
        query: str,
        embedding_uri: str,
        chroma_persist_dir: Path,
        meilisearch_url: str,
        collection_name: str = "mortgage_guidelines",
        index_name: str = "mortgage_guidelines",
        fusion_mode: FusionMode = FusionMode.RECIPROCAL_RANK,
        top_k: int = 3
    ) -> List[TextUnit]:
        """
        Standalone fusion retrieval (combines vector + fulltext)
        
        Args:
            query: Search query
            embedding_uri: Embedding model URI
            chroma_persist_dir: ChromaDB persistence directory
            meilisearch_url: Meilisearch server URL
            collection_name: ChromaDB collection name
            index_name: Meilisearch index name
            fusion_mode: Fusion strategy
            top_k: Number of final results
            
        Returns:
            List of retrieved units
        """
        # Create retrievers
        embedder = Embedder(embedding_uri)
        vector_store = ChromaVectorStore.local(
            path=str(chroma_persist_dir),
            collection_name=collection_name,
            embedder=embedder
        )
        
        vector_retriever = VectorRetriever(vector_store=vector_store, top_k=top_k * 2)
        fulltext_retriever = FullTextRetriever(
            url=meilisearch_url,
            index_name=index_name,
            top_k=top_k * 2
        )
        
        # Fusion retrieval
        fusion_retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=fusion_mode,
            top_k=top_k
        )
        
        results = fusion_retriever.retrieve(query)
        return results
    
    # ========== Post-processing ==========
    
    async def postprocess(
        self,
        query: str,
        results: List[TextUnit],
        similarity_threshold: float = 0.6,
        dedup_strategy: str = "exact",
        context_window: int = 1
    ) -> List[TextUnit]:
        """
        Apply postprocessing chain to retrieval results
        
        Args:
            query: Original query
            results: Retrieved units
            similarity_threshold: Minimum similarity score
            dedup_strategy: Deduplication strategy ("exact", "fuzzy")
            context_window: Context window size for augmentation
            
        Returns:
            Processed units
        """
        postprocessor = ChainPostprocessor([
            SimilarityFilter(threshold=similarity_threshold),
            Deduplicator(strategy=dedup_strategy),
            ContextAugmentor(window_size=context_window),
        ])
        
        processed_results = postprocessor.process(query, results)
        return processed_results
