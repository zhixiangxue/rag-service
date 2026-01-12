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
from zag.splitters import MarkdownHeaderSplitter, RecursiveMergingSplitter
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
API_KEY = os.getenv("BAILIAN_API_KEY")
EMBEDDING_MODEL = "text-embedding-v3"
LLM_MODEL = "qwen-plus"
EMBEDDING_URI = f"bailian/{EMBEDDING_MODEL}"
LLM_URI = f"bailian/{LLM_MODEL}"
MEILISEARCH_URL = "http://127.0.0.1:7700"
FILES_DIR = Path(__file__).parent / "files"
OUTPUT_DIR = Path(__file__).parent / "output"
CHROMA_PERSIST_DIR = OUTPUT_DIR / "chroma_db"

# 流程控制配置
RUN_UNTIL_STEP = 1  # 运行到第几步就停止 (1-7)，设置为 7 表示运行完整流程

# 创建输出目录
OUTPUT_DIR.mkdir(exist_ok=True)


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
    
    # 检查 API Key
    if not API_KEY:
        issues.append("❌ .env 文件中未找到 BAILIAN_API_KEY")
    else:
        console.print(f"✅ API Key 已找到: {API_KEY[:10]}...")
    
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
    """步骤 1: 读取所有 PDF 文档"""
    print_section("📄 步骤 1: 读取文档", "-")
    
    pdf_files = sorted(FILES_DIR.glob("*.pdf"))
    console.print(f"准备读取 {len(pdf_files)} 个 PDF 文件...")
    
    # 配置 DoclingReader
    pdf_options = PdfPipelineOptions()
    pdf_options.accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CPU
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
            console.print(f"\n正在读取: {pdf_path.name}")
            doc = reader.read(str(pdf_path))
            documents.append(doc)
            
            console.print(f"  ✅ 内容长度: {len(doc.content):,} 字符")
            console.print(f"  ✅ 页数: {len(doc.pages)}")
            if doc.metadata.custom:
                console.print(f"  ✅ 文本项: {doc.metadata.custom.get('text_items_count', 0)}")
                console.print(f"  ✅ 表格项: {doc.metadata.custom.get('table_items_count', 0)}")
            
            # 保存 Markdown 内容
            markdown_path = OUTPUT_DIR / f"{pdf_path.stem}_content.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(doc.content)
            console.print(f"  ✅ Markdown 已保存: {markdown_path.name}")
            
            progress.update(task, advance=1)
    
    console.print(f"\n✅ 共读取 {len(documents)} 个文档", style="bold green")
    return documents


async def step2_split_documents(documents):
    """步骤 2: 切分所有文档"""
    print_section("🔪 步骤 2: 切分文档", "-")
    
    console.print("使用 RecursiveMergingSplitter (目标: 800 tokens)...")
    base_splitter = MarkdownHeaderSplitter()
    merger = RecursiveMergingSplitter(
        base_splitter=base_splitter,
        target_token_size=800
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
            units = doc.split(merger)
            all_units.extend(units)
            console.print(f"  {doc.metadata.filename}: {len(units)} 个单元")
            progress.update(task, advance=1)
    
    # 计算 token 统计
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = [len(tokenizer.encode(u.content)) for u in all_units]
    
    console.print(f"\n✅ 切分完成:", style="bold green")
    console.print(f"   - 总单元数: {len(all_units)}")
    console.print(f"   - Token 范围: {min(tokens)}-{max(tokens)} (平均: {sum(tokens)//len(tokens)})")
    
    return all_units


async def step3_process_tables(units):
    """步骤 3: 处理表格 (解析 + 摘要)"""
    print_section("📊 步骤 3: 处理表格", "-")
    
    console.print("使用 LLM 提取表格信息...")
    extractor = TableExtractor(
        llm_uri=LLM_URI,
        api_key=API_KEY
    )
    
    # 批量提取
    results = await extractor.aextract(units)
    
    # 更新 embedding_content
    for unit, metadata in zip(units, results):
        if metadata.get("embedding_content"):
            unit.embedding_content = metadata["embedding_content"]
    
    console.print(f"✅ 已处理 {len(units)} 个单元", style="bold green")
    return units


async def step4_extract_metadata(units):
    """步骤 4: 提取元数据 (关键词)"""
    print_section("🏷️  步骤 4: 提取元数据", "-")
    
    console.print("为所有单元提取关键词...")
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
    
    return units


async def step5_build_indices(units):
    """步骤 5: 构建索引 (向量 + 全文)"""
    print_section("📚 步骤 5: 构建索引", "-")
    
    # 保存 units 到 JSON 以供检查
    import json
    units_json_path = OUTPUT_DIR / "units_data.json"
    units_data = [unit.model_dump(mode='json') for unit in units]
    
    with open(units_json_path, 'w', encoding='utf-8') as f:
        json.dump(units_data, f, ensure_ascii=False, indent=2)
    
    console.print(f"Units 数据已保存到: {units_json_path}")
    console.print(f"总单元数: {len(units)}\n")
    
    # 5.1 向量索引
    console.print("构建向量索引...")
    embedder = Embedder(
        EMBEDDING_URI,
        api_key=API_KEY
    )
    
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


async def step6_test_retrieval(vector_indexer, fulltext_indexer):
    """步骤 6: 测试检索功能"""
    print_section("🔍 步骤 6: 测试检索", "-")
    
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
    
    # 创建融合检索器
    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, fulltext_retriever],
        mode=FusionMode.RECIPROCAL_RANK,
        top_k=3
    )
    
    console.print("测试查询示例:\n")
    for i, query in enumerate(test_queries[:3], 1):
        console.print(f"[bold cyan]{i}. 查询:[/bold cyan] {query}")
        
        start = time.time()
        results = fusion_retriever.retrieve(query)
        elapsed = time.time() - start
        
        console.print(f"   ✅ 找到 {len(results)} 个结果 ({elapsed*1000:.0f}ms)")
        
        if results:
            # 显示第一个结果
            top_result = results[0]
            preview = top_result.content[:100].replace("\n", " ")
            console.print(f"   📄 来源: {top_result.metadata.filename}")
            console.print(f"   💯 得分: {top_result.score:.4f}")
            console.print(f"   📝 预览: {preview}...")
        console.print()
    
    return vector_retriever, fulltext_retriever


async def step7_test_postprocessing(vector_retriever, fulltext_retriever):
    """步骤 7: 测试后处理"""
    print_section("🔄 步骤 7: 测试后处理", "-")
    
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
            source = unit.metadata.filename if hasattr(unit.metadata, 'filename') else "N/A"
            preview = unit.content[:47].replace("\n", " ") + "..."
            
            table.add_row(str(i), score, source, preview)
        
        console.print(table)
    
    return processed_results


async def main():
    """主流程"""
    global _pipeline_start_time
    
    console.print("\n" + "=" * 70)
    console.print("  🚀 房贷指南 RAG Pipeline", style="bold cyan")
    console.print("=" * 70)
    console.print(f"\n📌 配置: 运行到 Step {RUN_UNTIL_STEP}\n", style="yellow")
    
    # 检查前置条件
    if not check_prerequisites():
        console.print("\n❌ 前置条件不满足，请修复上述问题。", style="bold red")
        sys.exit(1)
    
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
        
        vector_retriever, fulltext_retriever = await step6_test_retrieval(vector_indexer, fulltext_indexer)
        should_stop_after_step(6)
        
        final_results = await step7_test_postprocessing(vector_retriever, fulltext_retriever)
        
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
        console.print(f"   - 输出目录: {OUTPUT_DIR}")
        
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
