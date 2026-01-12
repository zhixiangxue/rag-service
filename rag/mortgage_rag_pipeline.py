#!/usr/bin/env python3
"""
Mortgage Guidelines RAG Pipeline

处理房贷指南文档的完整 RAG 流程：
1. 文档读取 (PDF)
2. 文档切分 (基于 Markdown 标题 + 递归合并)
3. 表格处理 (解析 + 摘要)
4. 元数据提取 (关键词)
5. 索引构建 (向量 + 全文)
6. 检索测试 (融合检索)
7. 后处理 (过滤 + 去重 + 上下文增强)

运行前准备：
1. 启动 Meilisearch: ./meilisearch
2. 设置环境变量 .env:
   BAILIAN_API_KEY=your-api-key
3. PDF 文件在 rag/files/ 目录下
4. GPU 加速 (可选):
   # 卸载 CPU 版 PyTorch
   pip uninstall torch torchvision torchaudio
   
   # 安装 CUDA 版 PyTorch (以 CUDA 12.1 为例)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # 验证 CUDA 可用性
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from zag.readers.docling import DoclingReader
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from zag.splitters import MarkdownHeaderSplitter, TextSplitter, TableSplitter, RecursiveMergingSplitter
from zag.extractors import TableExtractor, KeywordExtractor
from zag.embedders import Embedder
from zag.storages.vector import ChromaVectorStore
from zag.indexers import VectorIndexer, FullTextIndexer
from zag.retrievers import VectorRetriever, FullTextRetriever, QueryFusionRetriever, FusionMode
from zag.postprocessors import (
    SimilarityFilter,
    Deduplicator,
    ContextAugmentor,
    ChainPostprocessor,
)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel

console = Console()

# 加载环境变量
load_dotenv()

# 配置参数
API_KEY = os.getenv("BAILIAN_API_KEY")  # 保留用于可能的远程模型
EMBEDDING_MODEL = "jina/jina-embeddings-v2-base-en:latest"  # Ollama 本地模型
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 服务地址
LLM_MODEL = "qwen2.5:7b"  # 使用 Ollama 本地千问模型（调试用）
EMBEDDING_URI = f"ollama/{EMBEDDING_MODEL}"  # 使用 Ollama embedder
LLM_URI = f"ollama/{LLM_MODEL}"  # 使用 Ollama 本地 LLM（不需要 API Key）
MEILISEARCH_URL = "http://127.0.0.1:7700"
FILES_DIR = Path(__file__).parent / "files"
OUTPUT_ROOT = Path(__file__).parent / "output"  # 根输出目录
CHROMA_PERSIST_DIR = OUTPUT_ROOT / "chroma_db"  # 共享的向量数据库

# 流程控制配置
RUN_UNTIL_STEP = 6  # 运行到第几步就停止 (1-7)，设置为 7 表示运行完整流程

# 创建根输出目录
OUTPUT_ROOT.mkdir(exist_ok=True)

# 全局变量：当前处理文档的输出目录（在 main() 中设置）
CURRENT_DOC_OUTPUT_DIR = None


def print_section(title: str, char: str = "="):
    """打印分节标题"""
    console.print(f"\n{char * 70}")
    console.print(f"  {title}", style="bold cyan")
    console.print(f"{char * 70}\n")


# 全局变量用于存储开始时间
_pipeline_start_time = None


def should_stop_after_step(step_num: int):
    """检查是否应该在当前步骤后停止"""
    if step_num >= RUN_UNTIL_STEP:
        total_time = time.time() - _pipeline_start_time
        console.print(f"\n✅ 执行到 Step {step_num}，完成！(耗时 {total_time:.2f}s)", style="bold green")
        console.print(f"\n💡 提示: 修改 RUN_UNTIL_STEP 配置可以运行更多步骤", style="yellow")
        sys.exit(0)


def check_prerequisites():
    """检查前置条件"""
    print_section("🔍 检查前置条件")
    
    issues = []
    
    # 检查 API Key (可选，仅当使用远程 LLM 时需要)
    if LLM_URI.startswith("bailian/"):
        if not API_KEY:
            issues.append("❌ .env 文件中未找到 BAILIAN_API_KEY (Bailian LLM 需要)")
        else:
            console.print(f"✅ API Key 已找到: {API_KEY[:10]}...")
    else:
        console.print(f"📦 使用本地 LLM: {LLM_URI}（无需 API Key）")
    
    # 检查 Ollama 服务
    try:
        import httpx
        response = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            console.print(f"✅ Ollama 服务运行中: {OLLAMA_BASE_URL}")
        else:
            issues.append(f"❌ Ollama 服务异常: {response.status_code}")
    except Exception as e:
        issues.append(f"❌ 无法连接到 Ollama: {e}")
        console.print(f"   💡 提示: 请确保 Ollama 已启动 (ollama serve)")
    
    # 检查 GPU/CUDA 支持
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"✅ GPU 可用: {gpu_name}")
            console.print(f"   CUDA 版本: {torch.version.cuda}")
        else:
            console.print(f"⚠️  GPU 不可用，将使用 CPU（性能较低）", style="yellow")
            console.print(f"   💡 提示: 安装 CUDA 版 PyTorch 以启用 GPU 加速")
            console.print(f"   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        issues.append("❌ PyTorch 未安装")
    
    # 检查 PDF 文件
    pdf_files = list(FILES_DIR.glob("*.pdf"))
    if not pdf_files:
        issues.append(f"❌ 未找到 PDF 文件: {FILES_DIR}")
    else:
        console.print(f"✅ 找到 {len(pdf_files)} 个 PDF 文件:")
        for pdf in pdf_files:
            size_mb = pdf.stat().st_size / (1024 * 1024)
            console.print(f"   - {pdf.name} ({size_mb:.1f} MB)")
    
    # 检查 Meilisearch
    try:
        import meilisearch
        client = meilisearch.Client(MEILISEARCH_URL)
        health = client.health()
        if health.get("status") == "available":
            console.print(f"✅ Meilisearch 运行中: {MEILISEARCH_URL}")
        else:
            issues.append("❌ Meilisearch 不可用")
    except Exception as e:
        issues.append(f"❌ 无法连接到 Meilisearch: {e}")
    
    if issues:
        console.print("\n⚠️  发现问题:", style="bold yellow")
        for issue in issues:
            console.print(f"  {issue}")
        return False
    
    console.print("\n✅ 所有前置条件满足!", style="bold green")
    return True


async def step1_read_documents():
    """步骤 1: 读取所有 PDF 文档（支持缓存 + 质量验证）"""
    print_section("📄 步骤 1: 读取文档", "-")
    
    pdf_files = sorted(FILES_DIR.glob("*.pdf"))
    console.print(f"准备读取 {len(pdf_files)} 个 PDF 文件...")
    
    # 导入质量验证工具
    from validate_conversion import validate_cache_quality
    
    # 配置 DoclingReader
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )
    
    reader = DoclingReader(pdf_pipeline_options=pdf_options)
    documents = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]读取文档...", total=len(pdf_files))
        
        for pdf_path in pdf_files:
            # 使用文档专属的 raw/ 子目录
            raw_dir = CURRENT_DOC_OUTPUT_DIR / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            markdown_path = raw_dir / f"{pdf_path.stem}.md"
            
            # 检查缓存是否存在且质量合格
            use_cache = False
            if markdown_path.exists():
                console.print(f"\n🔍 检查缓存质量: {pdf_path.name}")
                is_valid = validate_cache_quality(pdf_path, markdown_path, threshold=90.0, verbose=False)
                
                if is_valid:
                    console.print(f"  ✅ 缓存质量合格 (>= 90分)，使用缓存")
                    use_cache = True
                else:
                    console.print(f"  ⚠️  缓存质量不足 (< 90分)，重新解析 PDF")
                    # 删除低质量缓存
                    markdown_path.unlink()
            
            if use_cache:
                # 从缓存加载
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 构造简化的元数据
                from zag.schemas.base import DocumentMetadata, Page
                from zag.schemas.pdf import PDF
                
                metadata = DocumentMetadata(
                    source=str(pdf_path),
                    source_type="local",
                    file_type="pdf",
                    file_name=pdf_path.name,
                    file_size=pdf_path.stat().st_size,
                    file_extension=".pdf",
                    content_length=len(content),
                    reader_name="DoclingReader (cached)",
                    custom={
                        'cached': True,
                        'cache_file': str(markdown_path),
                        'quality_validated': True
                    }
                )
                
                # 创建简单的 Page 对象
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
                
                documents.append(doc)
                console.print(f"  ✅ 内容长度: {len(content):,} 字符 (从缓存)")
                
            else:
                # 没有缓存或质量不合格，从 PDF 读取
                console.print(f"\n📄 解析 PDF: {pdf_path.name}")
                doc = reader.read(str(pdf_path))
                documents.append(doc)
                
                console.print(f"  ✅ 内容长度: {len(doc.content):,} 字符")
                console.print(f"  ✅ 页数: {len(doc.pages)}")
                if doc.metadata.custom:
                    console.print(f"  ✅ 文本项: {doc.metadata.custom.get('text_items_count', 0)}")
                    console.print(f"  ✅ 表格项: {doc.metadata.custom.get('table_items_count', 0)}")
                
                # 保存 Markdown 内容作为缓存
                with open(markdown_path, 'w', encoding='utf-8') as f:
                    f.write(doc.content)
                console.print(f"  ✅ Markdown 已保存: {markdown_path.name}")
            
            progress.update(task, advance=1)
    
    console.print(f"\n✅ 共读取 {len(documents)} 个文档", style="bold green")
    return documents


async def step2_split_documents(documents):
    """步骤 2: 切分所有文档"""
    print_section("🔪 步骤 2: 切分文档", "-")
    
    console.print("使用完整 Pipeline: MarkdownHeaderSplitter | TextSplitter | TableSplitter | RecursiveMergingSplitter")
    console.print("  - MarkdownHeaderSplitter: 按标题切分")
    console.print("  - TextSplitter(1200 tokens): 打断超大块")
    console.print("  - TableSplitter(1500 tokens): 切分超大表格")
    console.print("  - RecursiveMergingSplitter(800 tokens): 合并小块到目标大小\n")
    
    # 构建完整的切分 pipeline
    pipeline = (
        MarkdownHeaderSplitter()
        | TextSplitter(max_chunk_tokens=1200)
        | TableSplitter(max_chunk_tokens=1500)
        | RecursiveMergingSplitter(target_token_size=800)
    )
    
    all_units = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]切分文档...", total=len(documents))
        
        for doc in documents:
            units = doc.split(pipeline)
            all_units.extend(units)
            console.print(f"  {doc.metadata.file_name}: {len(units)} 个单元")
            progress.update(task, advance=1)
    
    # 计算 token 统计
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_counts = [len(tokenizer.encode(u.content)) for u in all_units]
    
    console.print(f"\n✅ 切分完成:", style="bold green")
    console.print(f"   - 总单元数: {len(all_units)}")
    console.print(f"   - Token 范围: {min(token_counts)}-{max(token_counts)} (平均: {sum(token_counts)//len(token_counts)})")
    
    # Token 分布统计
    console.print(f"\n📊 Token 分布:")
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
    
    # 检查超大块
    oversized = [(i, t) for i, t in enumerate(token_counts) if t > 1500]
    if oversized:
        console.print(f"\n⚠️  发现 {len(oversized)} 个超大单元 (>1500 tokens):", style="yellow")
        for idx, tokens in oversized[:5]:  # 只显示前5个
            context = all_units[idx].metadata.context_path if all_units[idx].metadata else "N/A"
            console.print(f"   Unit {idx}: {tokens:,} tokens | {context[:50]}...")
        if len(oversized) > 5:
            console.print(f"   ... 还有 {len(oversized) - 5} 个")
    
    # 导出可视化文件
    from datetime import datetime
    console.print(f"\n💾 导出可视化文件...")
    
    # 使用文档专属的 split/ 子目录
    split_dir = CURRENT_DOC_OUTPUT_DIR / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = split_dir / "visualization"
    viz_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 从文档元数据中获取文档名
    doc_name = documents[0].metadata.file_name.rsplit('.', 1)[0] if documents else "document"
    viz_file = viz_dir / f"{doc_name}_split_{timestamp}.md"
    
    with open(viz_file, 'w', encoding='utf-8') as f:
        # 写入头部
        f.write(f"# Mortgage Guidelines - Document Splitting Visualization\n\n")
        f.write(f"**Total Units**: {len(all_units)}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Token Range**: {min(token_counts)}-{max(token_counts)} tokens\n\n")
        f.write(f"**Average**: {sum(token_counts)//len(token_counts)} tokens\n\n")
        f.write("---\n\n")
        
        # 写入每个 unit
        for i, unit in enumerate(all_units):
            tokens = len(tokenizer.encode(unit.content))
            
            # 视觉分隔符
            f.write(f"\n\n")
            f.write(f"{'🔷' * 50}\n\n")
            
            # Unit 头部
            f.write(f"## 📦 Unit {i} | {tokens} tokens\n\n")
            
            # 元数据信息
            if hasattr(unit, 'metadata') and unit.metadata:
                if hasattr(unit.metadata, 'context_path') and unit.metadata.context_path:
                    f.write(f"**Context**: {unit.metadata.context_path}\n\n")
            
            # 内容预览
            preview = unit.content.strip()[:100].replace('\n', ' ')
            f.write(f"**Preview**: {preview}...\n\n")
            
            # Token 大小指示
            if tokens > 1500:
                f.write(f"⚠️ **OVERSIZED** ({tokens} tokens)\n\n")
            elif tokens >= 1000:
                f.write(f"📊 **LARGE** ({tokens} tokens)\n\n")
            
            f.write(f"---\n\n")
            
            # 实际内容
            f.write(unit.content)
            f.write("\n\n")
    
    console.print(f"   ✅ 已保存到: {viz_file.name}", style="green")
    console.print(f"   📁 位置: {viz_dir}")
    
    return all_units


async def step3_process_tables(units):
    """步骤 3: 处理表格 (解析 + 摘要)"""
    print_section("📊 步骤 3: 处理表格", "-")
    
    # 检查缓存
    tables_dir = CURRENT_DOC_OUTPUT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    units_json_path = tables_dir / "units_after_table_processing.json"
    
    if units_json_path.exists():
        console.print(f"🔍 发现缓存文件: {units_json_path.name}")
        console.print(f"   跳过表格处理，直接加载缓存...")
        
        import json
        from zag.schemas.unit import TextUnit, TableUnit
        from zag.schemas.base import UnitType
        
        with open(units_json_path, 'r', encoding='utf-8') as f:
            units_data = json.load(f)
        
        # 重建 Unit 对象（根据 unit_type 选择类）
        units = []
        for data in units_data:
            unit_type = data.get('unit_type', 'TEXT')
            if unit_type == 'TABLE' or unit_type == UnitType.TABLE.value:
                units.append(TableUnit(**data))
            else:
                units.append(TextUnit(**data))
        
        console.print(f"✅ 已从缓存加载 {len(units)} 个单元", style="bold green")
        units_with_embedding = sum(1 for u in units if hasattr(u, 'embedding_content') and u.embedding_content)
        console.print(f"   包含 embedding_content 的单元: {units_with_embedding}")
        
        return units
    
    # 没有缓存，执行处理
    console.print(f"使用 LLM 提取表格信息: {LLM_URI}")
    extractor = TableExtractor(llm_uri=LLM_URI, api_key=API_KEY)
    
    # 批量提取
    results = await extractor.aextract(units)
    
    # 更新 embedding_content
    for unit, metadata in zip(units, results):
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    console.print(f"✅ 已处理 {len(units)} 个单元", style="bold green")
    
    # 保存处理后的 units 到 JSON（使用文档专属的 tables/ 子目录）
    import json
    units_data = [unit.model_dump(mode='json') for unit in units]
    
    with open(units_json_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, ensure_ascii=False, indent=2)
    
    console.print(f"💾 已保存处理后的 units: {units_json_path.name}")
    console.print(f"   总单元数: {len(units)}")
    
    # 显示一些统计信息
    units_with_embedding = sum(1 for u in units if hasattr(u, 'embedding_content') and u.embedding_content)
    console.print(f"   包含 embedding_content 的单元: {units_with_embedding}")
    
    return units


async def step4_extract_metadata(units):
    """步骤 4: 提取元数据 (关键词)"""
    print_section("🏷️  步骤 4: 提取元数据", "-")
    
    # 检查缓存
    metadata_dir = CURRENT_DOC_OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    units_json_path = metadata_dir / "units_after_keyword_extraction.json"
    
    if units_json_path.exists():
        console.print(f"🔍 发现缓存文件: {units_json_path.name}")
        console.print(f"   跳过关键词提取，直接加载缓存...")
        
        import json
        from zag.schemas.unit import TextUnit, TableUnit
        from zag.schemas.base import UnitType
        
        with open(units_json_path, 'r', encoding='utf-8') as f:
            units_data = json.load(f)
        
        # 重建 Unit 对象（根据 unit_type 选择类）
        units = []
        for data in units_data:
            unit_type = data.get('unit_type', 'TEXT')
            if unit_type == 'TABLE' or unit_type == UnitType.TABLE.value:
                units.append(TableUnit(**data))
            else:
                units.append(TextUnit(**data))
        
        console.print(f"✅ 已从缓存加载 {len(units)} 个单元", style="bold green")
        units_with_keywords = sum(1 for u in units if u.metadata.custom.get('excerpt_keywords'))
        console.print(f"   包含关键词的单元: {units_with_keywords}")
        
        return units
    
    # 没有缓存，执行提取
    console.print(f"为所有单元提取关键词: {LLM_URI}")
    extractor = KeywordExtractor(
        llm_uri=LLM_URI,
        api_key=API_KEY,
        num_keywords=5
    )
    
    # 批量提取
    results = await extractor.aextract(units)
    
    # 更新元数据
    for unit, metadata in zip(units, results):
        unit.metadata.custom.update(metadata)
    
    console.print(f"✅ 已为 {len(units)} 个单元提取关键词", style="bold green")
    console.print("\n示例关键词 (前 3 个单元):")
    for i, unit in enumerate(units[:3], 1):
        keywords = unit.metadata.custom.get("excerpt_keywords", [])
        console.print(f"   {i}. {keywords}")
    
    # 保存提取关键词后的 units 到 JSON（使用文档专属的 metadata/ 子目录）
    import json
    units_data = [unit.model_dump(mode='json') for unit in units]
    
    with open(units_json_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n💾 已保存处理后的 units: {units_json_path.name}")
    console.print(f"   总单元数: {len(units)}")
    
    # 显示统计信息
    units_with_keywords = sum(1 for u in units if u.metadata.custom.get('excerpt_keywords'))
    console.print(f"   包含关键词的单元: {units_with_keywords}")
    
    return units


async def step5_build_indices(units):
    """步骤 5: 构建索引 (向量 + 全文)"""
    print_section("📚 步骤 5: 构建索引", "-")
    
    # 保存 units 到 JSON 以供检查（使用文档专属的 metadata/ 子目录）
    import json
    metadata_dir = CURRENT_DOC_OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    units_json_path = metadata_dir / "units_data.json"
    units_data = [unit.model_dump(mode='json') for unit in units]
    
    with open(units_json_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, ensure_ascii=False, indent=2)
    
    console.print(f"Units 数据已保存到: {units_json_path}")
    console.print(f"总单元数: {len(units)}\n")
    
    # 5.1 向量索引
    console.print("构建向量索引...")
    console.print(f"   使用本地 Ollama 模型: {EMBEDDING_MODEL}")
    embedder = Embedder(EMBEDDING_URI)
    
    vector_store = ChromaVectorStore.local(
        path=str(CHROMA_PERSIST_DIR),
        collection_name="mortgage_guidelines",
        embedder=embedder
    )
    console.print(f"   持久化目录: {CHROMA_PERSIST_DIR}")
    
    vector_indexer = VectorIndexer(vector_store=vector_store)
    # 清空现有数据
    await vector_indexer.aclear()
    await vector_indexer.aadd(units)
    console.print(f"   ✅ 向量索引已构建: {vector_indexer.count()} 个单元", style="bold green")
    
    # 5.2 全文索引
    console.print("\n构建全文索引...")
    fulltext_indexer = FullTextIndexer(
        url=MEILISEARCH_URL,
        index_name="mortgage_guidelines",
        primary_key="unit_id"
    )
    
    # 清空现有数据
    fulltext_indexer.clear()
    fulltext_indexer.configure_settings(
        searchable_attributes=["content", "context_path"],
        filterable_attributes=["unit_type", "source_doc_id"],
        sortable_attributes=["created_at"],
    )
    fulltext_indexer.add(units)
    console.print(f"   ✅ 全文索引已构建: {fulltext_indexer.count()} 个单元", style="bold green")
    
    return vector_indexer, fulltext_indexer


async def step6_test_retrieval(retriever_type: str = "fusion"):
    """
    步骤 6: 测试检索功能（可独立运行）
    
    Args:
        retriever_type: 检索器类型，可选 "vector", "fulltext", "fusion"
    """
    print_section(f"🔍 步骤 6: 测试检索 ({retriever_type.upper()})", "-")
    
    # 自己构建 vector_store 和 indexer（从持久化数据加载）
    console.print("初始化检索器...")
    console.print(f"   向量数据库: {CHROMA_PERSIST_DIR}")
    console.print(f"   全文索引: {MEILISEARCH_URL}")
    console.print(f"   检索类型: {retriever_type.upper()}\n")
    
    embedder = Embedder(EMBEDDING_URI)
    vector_store = ChromaVectorStore.local(
        path=str(CHROMA_PERSIST_DIR),
        collection_name="mortgage_guidelines",
        embedder=embedder
    )
    vector_indexer = VectorIndexer(vector_store=vector_store)
    console.print(f"✅ 向量索引已加载: {vector_indexer.count()} 个单元")
    
    fulltext_indexer = FullTextIndexer(
        url=MEILISEARCH_URL,
        index_name="mortgage_guidelines",
        primary_key="unit_id"
    )
    console.print(f"✅ 全文索引已加载: {fulltext_indexer.count()} 个单元\n")
    
    # 测试查询
    test_queries = [
        "FHA 贷款的首付要求是什么？",
        "VA 贷款的资格条件有哪些？",
        "Fannie Mae 的 LTV 要求",
        "USDA 贷款适用的地区",
        "Freddie Mac 的利率政策",
    ]
    
    # 创建检索器
    vector_retriever = VectorRetriever(vector_store=vector_indexer.vector_store, top_k=5)
    fulltext_retriever = FullTextRetriever(url=MEILISEARCH_URL, index_name="mortgage_guidelines", top_k=5)
    
    # 根据类型选择检索器
    if retriever_type == "vector":
        retriever = vector_retriever
    elif retriever_type == "fulltext":
        retriever = fulltext_retriever
    elif retriever_type == "fusion":
        # 创建融合检索器
        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, fulltext_retriever],
            mode=FusionMode.RECIPROCAL_RANK,
            top_k=3
        )
    else:
        raise ValueError(f"Unknown retriever_type: {retriever_type}")
    
    console.print("测试查询示例:\n")
    for i, query in enumerate(test_queries, 1):
        console.print(f"[bold cyan]{i}. 查询:[/bold cyan] {query}")
        
        start = time.time()
        results = retriever.retrieve(query)
        elapsed = time.time() - start
        
        console.print(f"   ✅ 找到 {len(results)} 个结果 ({elapsed*1000:.0f}ms)")
        
        if results:
            # 显示第一个结果
            top_result = results[0]
            preview = top_result.content[:100].replace("\n", " ")
            
            # 获取来源信息：优先使用 context_path，其次使用 source检索来源
            source_info = "N/A"
            if top_result.metadata and top_result.metadata.context_path:
                source_info = top_result.metadata.context_path.split('/')[0]  # 取第一级路径
            if top_result.source:
                source_info = f"{source_info} ({top_result.source.value})"
            
            console.print(f"   📄 来源: {source_info}")
            console.print(f"   💯 得分: {top_result.score:.4f}")
            console.print(f"   📝 预览: {preview}...")
        console.print()
    
    return vector_retriever, fulltext_retriever


async def step7_test_postprocessing():
    """步骤 7: 测试后处理（可独立运行）"""
    print_section("🔄 步骤 7: 测试后处理", "-")
    
    # 自己构建 retriever（从持久化数据加载）
    console.print("初始化检索器...")
    embedder = Embedder(EMBEDDING_URI)
    vector_store = ChromaVectorStore.local(
        path=str(CHROMA_PERSIST_DIR),
        collection_name="mortgage_guidelines",
        embedder=embedder
    )
    vector_retriever = VectorRetriever(vector_store=vector_store, top_k=5)
    fulltext_retriever = FullTextRetriever(url=MEILISEARCH_URL, index_name="mortgage_guidelines", top_k=5)
    console.print("✅ 检索器已初始化\n")
    
    query = "不同贷款产品的利率和首付要求比较"
    
    console.print(f"查询: [bold]{query}[/bold]\n")
    console.print("使用融合检索 (RRF)...")
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, fulltext_retriever],
        mode=FusionMode.RECIPROCAL_RANK,
        top_k=10
    )
    raw_results = fusion_retriever.retrieve(query)
    console.print(f"   原始结果: {len(raw_results)} 个单元")
    
    # 创建后处理链
    postprocessor = ChainPostprocessor([
        SimilarityFilter(threshold=0.6),
        Deduplicator(strategy="exact"),
        ContextAugmentor(window_size=1),
    ])
    
    console.print("\n应用后处理链:")
    console.print("   1. SimilarityFilter(threshold=0.6)")
    console.print("   2. Deduplicator(strategy='exact')")
    console.print("   3. ContextAugmentor(window_size=1)")
    
    processed_results = postprocessor.process(query, raw_results)
    console.print(f"\n   ✅ 处理后结果: {len(processed_results)} 个单元", style="bold green")
    
    # 显示结果
    if processed_results:
        table = Table(title="后处理结果", show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", width=4)
        table.add_column("得分", style="green", width=10)
        table.add_column("来源", style="yellow", width=30)
        table.add_column("内容预览", style="white", width=50)
        
        for i, unit in enumerate(processed_results[:5], 1):
            score = f"{unit.score:.4f}" if hasattr(unit, 'score') and unit.score else "N/A"
            
            # 获取来源信息
            source = "N/A"
            if unit.metadata and unit.metadata.context_path:
                source = unit.metadata.context_path.split('/')[0]  # 取第一级路径
            if hasattr(unit, 'source') and unit.source:
                source = f"{source} ({unit.source.value})"
            
            preview = unit.content[:47].replace("\n", " ") + "..."
            
            table.add_row(str(i), score, source, preview)
        
        console.print(table)
    
    return processed_results


async def main():
    """主流程"""
    global _pipeline_start_time, CURRENT_DOC_OUTPUT_DIR
    
    console.print("\n" + "=" * 70)
    console.print("  🚀 房贷指南 RAG Pipeline", style="bold cyan")
    console.print("=" * 70)
    console.print(f"\n📌 配置: 运行到 Step {RUN_UNTIL_STEP}\n", style="yellow")
    
    # 检查前置条件
    if not check_prerequisites():
        console.print("\n❌ 前置条件不满足，请修复上述问题。", style="bold red")
        sys.exit(1)
    
    # 检测 PDF 文件，创建文档专属输出目录
    pdf_files = sorted(FILES_DIR.glob("*.pdf"))
    if not pdf_files:
        console.print("\n❌ 未找到 PDF 文件", style="bold red")
        sys.exit(1)
    
    # 使用第一个 PDF 的文件名作为输出目录名（去除扩展名）
    doc_name = pdf_files[0].stem
    CURRENT_DOC_OUTPUT_DIR = OUTPUT_ROOT / doc_name
    
    # 创建文档专属子目录结构
    subdirs = ["raw", "split", "tables", "metadata", "indices"]
    for subdir in subdirs:
        (CURRENT_DOC_OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n📁 文档输出目录: {CURRENT_DOC_OUTPUT_DIR}")
    console.print(f"   子目录: {', '.join(subdirs)}\n", style="dim")
    
    try:
        _pipeline_start_time = time.time()
        
        # 执行流程
        documents = await step1_read_documents()
        should_stop_after_step(1)
        
        units = await step2_split_documents(documents)
        should_stop_after_step(2)
        
        units = await step3_process_tables(units)
        should_stop_after_step(3)
        
        units = await step4_extract_metadata(units)
        should_stop_after_step(4)
        
        vector_indexer, fulltext_indexer = await step5_build_indices(units)
        should_stop_after_step(5)
        
        # 分别测试三种检索方式
        console.print("\n" + "=" * 70)
        console.print("  🎯 开始检索测试", style="bold cyan")
        console.print("=" * 70)
        
        await step6_test_retrieval(retriever_type="vector")
        console.print("\n" + "-" * 70 + "\n")
        
        await step6_test_retrieval(retriever_type="fulltext")
        console.print("\n" + "-" * 70 + "\n")
        
        await step6_test_retrieval(retriever_type="fusion")
        should_stop_after_step(6)
                
        final_results = await step7_test_postprocessing()
        
        total_time = time.time() - _pipeline_start_time
        
        # 总结
        print_section("📊 Pipeline 总结")
        console.print(f"✅ E2E pipeline 完成，耗时 {total_time:.2f}s", style="bold green")
        console.print(f"\nPipeline 阶段:")
        console.print(f"   1. ✅ 文档读取: PDF → Markdown")
        console.print(f"   2. ✅ 文档切分: 基于标题 + 递归合并")
        console.print(f"   3. ✅ 表格处理: 解析 + 摘要")
        console.print(f"   4. ✅ 元数据提取: 关键词")
        console.print(f"   5. ✅ 索引构建: 向量 + 全文")
        console.print(f"   6. ✅ 检索测试: 融合检索策略")
        console.print(f"   7. ✅ 后处理: 过滤 + 去重 + 增强")
        
        console.print(f"\n💡 关键指标:")
        console.print(f"   - 总文档数: {len(documents)}")
        console.print(f"   - 总单元数: {len(units)}")
        console.print(f"   - 索引方式: 向量 + 全文双索引")
        console.print(f"   - 检索策略: 融合检索 (RRF)")
        console.print(f"   - 输出目录: {CURRENT_DOC_OUTPUT_DIR}")
        
        console.print("\n" + "=" * 70)
        console.print("✅ 测试成功完成!", style="bold green")
        console.print("=" * 70)
        
    except Exception as e:
        console.print(f"\n❌ Pipeline 失败: {e}", style="bold red")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
