#!/usr/bin/env python3
"""
Simple RAG Query Client for Demo
Provides clean, easy-to-understand output for demonstrations
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from rich.spinner import Spinner
from rich.live import Live

import chak
from pydantic import BaseModel, Field
from typing import List, Optional

from zag.postprocessors import Reranker, LLMSelector
from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers import VectorRetriever

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For OpenAI embeddings and LLMs
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # For Cohere reranker
QDRANT_HOST = "localhost"
QDRANT_PORT = 16333
QDRANT_GRPC_PORT = 16334
COLLECTION_NAME = "mortgage_guidelines"
EMBEDDING_URI = "openai/text-embedding-3-small"
RERANKER_MODEL = "cohere/rerank-english-v3.0"  # Cohere production-grade reranker
TOP_K = 20
FINAL_TOP_K = 5

# LLM Selector for passage extraction (OpenAI GPT-4o-mini - fast and cost-effective)
LLM_SELECTOR_URI = "openai/gpt-4o-mini"

# LLM for relevance analysis (OpenAI GPT-4o - better quality)
LLM_URI = "openai/gpt-4o"

console = Console()


class RelevanceAnalysis(BaseModel):
    """Structured output for relevance analysis"""
    is_relevant: bool = Field(description="Whether the content is relevant to the query")
    confidence: str = Field(description="Confidence level: high/medium/low")
    reason: str = Field(description="Brief explanation of why it's relevant or not")
    relevant_excerpts: List[str] = Field(
        description="List of exact excerpts from original text that are relevant"
    )


async def analyze_relevance(query: str, content: str):
    """Analyze relevance between query and content"""
    try:
        conv = chak.Conversation(LLM_URI, api_key=OPENAI_API_KEY)
        
        analysis_prompt = f"""分析以下查询和检索内容的相关性。

用户查询：{query}

检索到的内容：
{content}

请分析：
1. 这段内容是否与查询相关？
2. 相关度如何（高/中/低）？
3. 为什么相关或不相关？
4. 如果相关，请一字不改地摘录出相关的部分（可以多段）

注意：摘录时必须完全按照原文，不要修改任何字词。
"""
        
        analysis = await conv.asend(analysis_prompt, returns=RelevanceAnalysis)
        return analysis
        
    except Exception as e:
        # Return a default analysis if failed
        return RelevanceAnalysis(
            is_relevant=True,
            confidence="unknown",
            reason=f"分析失败: {str(e)}",
            relevant_excerpts=[]
        )


def display_result(result, result_num: int, analysis: RelevanceAnalysis):
    """Display a single result with analysis (excludes internal 'document' metadata)"""
    
    # Build content for panel
    content_lines = []
    
    # Header
    content_lines.append(f"[bold cyan]ID:[/bold cyan] {result.unit_id}")
    content_lines.append(f"[bold cyan]分数:[/bold cyan] {result.score:.4f}")
    content_lines.append("")
    
    # Metadata (if available)
    if hasattr(result, 'metadata') and result.metadata:
        content_lines.append("[bold white]--- 元数据 ---[/bold white]")
        metadata = result.metadata
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                # Skip 'document' field - not relevant for business users
                if key == 'document':
                    continue
                content_lines.append(f"[cyan]{key}:[/cyan] {value}")
        else:
            # If it's a BaseModel
            for field_name, field_value in metadata.__dict__.items():
                # Skip private fields and 'document' field
                if field_name.startswith('_') or field_name == 'document':
                    continue
                content_lines.append(f"[cyan]{field_name}:[/cyan] {field_value}")
        content_lines.append("")
    
    # Content - display full content without truncation
    content_lines.append("[bold white]--- 内容 ---[/bold white]")
    content_lines.append(result.content)
    content_lines.append("")
    
    # Quality analysis
    content_lines.append("[bold white]--- 质量分析 ---[/bold white]")
    
    # Relevance status with color
    if analysis.is_relevant:
        relevance_text = "[green]✓ 相关[/green]"
    else:
        relevance_text = "[red]✗ 不相关[/red]"
    content_lines.append(f"[bold]相关性:[/bold] {relevance_text}")
    
    # Confidence with color
    confidence_colors = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
        "unknown": "dim"
    }
    conf_key = analysis.confidence.lower() if hasattr(analysis, 'confidence') else "unknown"
    confidence_color = confidence_colors.get(conf_key, "white")
    confidence_label = {"high": "高", "medium": "中", "low": "低", "unknown": "未知"}.get(conf_key, conf_key)
    content_lines.append(f"[bold]置信度:[/bold] [{confidence_color}]{confidence_label}[/{confidence_color}]")
    
    # Reason
    content_lines.append(f"[bold]原因:[/bold] {analysis.reason}")
    
    # Relevant excerpts
    if analysis.is_relevant and analysis.relevant_excerpts:
        content_lines.append("")
        content_lines.append("[bold]关键摘录:[/bold]")
        for excerpt in analysis.relevant_excerpts:
            # Truncate long excerpts
            excerpt_display = excerpt if len(excerpt) <= 150 else excerpt[:150] + "..."
            content_lines.append(f"  [dim]•[/dim] {excerpt_display}")
    
    # Create panel
    panel = Panel(
        "\n".join(content_lines),
        title=f"[bold]结果 {result_num}[/bold]",
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(panel)


async def query_loop():
    """Main query loop"""
    
    # Initialize components
    console.print("\n[dim]正在初始化...[/dim]")
    reranker = Reranker(RERANKER_MODEL, api_key=COHERE_API_KEY)
    selector = LLMSelector(llm_uri=LLM_SELECTOR_URI, api_key=OPENAI_API_KEY)
    
    embedder = Embedder(EMBEDDING_URI, api_key=OPENAI_API_KEY)
    vector_store = QdrantVectorStore.server(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=True,
        collection_name=COLLECTION_NAME,
        embedder=embedder
    )
    vector_retriever = VectorRetriever(vector_store=vector_store, top_k=TOP_K)
    console.print("[green]✓ 初始化完成[/green]\n")
    
    while True:
        # Get user input
        console.print("=" * 80)
        try:
            query = console.input("[bold green]问题:[/bold green] ").strip()
        except KeyboardInterrupt:
            console.print("\n\n[yellow]检测到 Ctrl+C，正在退出...[/yellow]")
            console.print("👋 再见!\n")
            break
        except EOFError:
            console.print("\n\n👋 再见!")
            break
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            console.print("\n👋 再见!")
            break
        
        console.print("=" * 80)
        console.print()
        
        # Execute query
        try:
            # Stage 1: Retrieval
            console.print("[bold][ 检索阶段 ][/bold]")
            
            start_retrieval = time.time()
            with console.status("[cyan]向量检索中...[/cyan]", spinner="dots"):
                results = vector_retriever.retrieve(query)
            time_retrieval = time.time() - start_retrieval
            
            if not results:
                console.print("[yellow]✗ 未找到结果[/yellow]\n")
                continue
            
            console.print(f"[green]✓ 检索完成: 找到 {len(results)} 条候选结果 (耗时 {time_retrieval:.2f}s)[/green]")
            console.print()
            
            # Stage 2: Postprocessing
            console.print("[bold][ 后处理阶段 ][/bold]")
            
            # Step 1: Reranking
            start_rerank = time.time()
            with console.status("[cyan]步骤1: Reranker 重排序...[/cyan]", spinner="dots"):
                results_reranked = reranker.rerank(query, results[:TOP_K], top_k=None)
            time_rerank = time.time() - start_rerank
            console.print(f"[green]✓ 重排序完成: 保留 {len(results_reranked)} 条结果 (耗时 {time_rerank:.2f}s)[/green]")
            console.print()
            
            # Step 2: LLM Selector
            start_selector = time.time()
            try:
                with console.status("[cyan]步骤2: LLM Selector 数据剪枝...[/cyan]", spinner="dots"):
                    results = await selector.aprocess(query, results_reranked)
                time_selector = time.time() - start_selector
                console.print(f"[green]✓ 段落提取完成: 筛选出 {len(results)} 条相关结果 (耗时 {time_selector:.2f}s)[/green]")
            except Exception as e:
                time_selector = time.time() - start_selector
                console.print(f"[yellow]⚠ LLM Selector 失败，跳过此步骤: {e}[/yellow]")
                console.print(f"[yellow]→ 使用重排序结果继续流程 ({len(results_reranked)} 条结果)[/yellow]")
                results = results_reranked
            console.print()
            
            # Total time
            time_total = time_retrieval + time_rerank + time_selector
            console.print(f"[bold green]总耗时: {time_total:.2f}s[/bold green]")
            console.print()
            
            # Limit to final top k
            results = results[:FINAL_TOP_K]
            
            # Display results
            console.print("=" * 80)
            console.print(f"[bold]最终结果: {len(results)} 条[/bold]")
            console.print("=" * 80)
            console.print()
            
            for i, result in enumerate(results, 1):
                # Analyze relevance
                try:
                    analysis = await analyze_relevance(query, result.content)
                except Exception as e:
                    console.print(f"[yellow]⚠ 结果 {i} 质量分析失败: {e}[/yellow]")
                    analysis = RelevanceAnalysis(
                        is_relevant=True,
                        confidence="unknown",
                        reason="质量分析失败，无法评估相关性",
                        relevant_excerpts=[]
                    )
                
                # Display result with analysis
                display_result(result, i, analysis)
                console.print()
        
        except KeyboardInterrupt:
            console.print("\n\n[yellow]检测到 Ctrl+C，正在取消当前查询...[/yellow]")
            console.print("提示: 再次按 Ctrl+C 可退出程序\n")
            continue
        except Exception as e:
            console.print(f"\n[bold red]❌ 错误: {e}[/bold red]")
            console.print("\n[yellow]故障排查:[/yellow]")
            console.print("  1. Qdrant 是否运行? 检查: curl http://localhost:16333/healthz")
            console.print("  2. Ollama 是否运行? 检查: curl http://localhost:11434/api/tags")
            console.print("  3. Collection 是否已索引? 先运行主流水线")
            console.print()


async def main():
    """Entry point"""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]           Mortgage RAG query demo            [/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    
    try:
        await query_loop()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]程序被中断[/yellow]")
        console.print("👋 Bye!\n")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]❌ 致命错误: {e}[/bold red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
