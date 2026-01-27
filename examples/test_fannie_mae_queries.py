"""
Fannie Mae Eligibility Matrix 查询测试

根据上传的 Fannie_Mae_Eligibility Matrix_12-10-25.pdf 文档进行查询测试。

使用方法：
    python examples/test_fannie_mae_queries.py
"""
import requests
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

BASE_URL = "http://13.56.109.233:8000"
DATASET_ID = "mortgage_guidelines"


def query_and_display(question: str, top_k: int = 3):
    """
    查询并显示结果
    
    Args:
        question: 查询问题
        top_k: 返回结果数量
    """
    console.print("\n" + "=" * 80)
    console.print(f"[bold cyan]问题: {question}[/bold cyan]")
    console.print("=" * 80 + "\n")
    
    try:
        response = requests.post(
            f"{BASE_URL}/datasets/{DATASET_ID}/query",
            json={
                "query": question,
                "top_k": top_k,
                "filters": {}
            },
            timeout=30
        )
        
        if response.status_code != 200:
            console.print(f"[red]❌ 请求失败: {response.status_code}[/red]")
            console.print(f"[red]{response.text}[/red]")
            return
        
        data = response.json()
        results = data.get("data", [])
        
        if not results:
            console.print("[yellow]⚠️  没有找到相关结果[/yellow]")
            return
        
        console.print(f"[green]✅ 找到 {len(results)} 条相关结果\n[/green]")
        
        # 显示每条结果
        for idx, result in enumerate(results, 1):
            score = result.get("score", 0)
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            custom = metadata.get("custom", {})
            
            # 构建标题
            title = f"结果 #{idx} (相似度: {score:.4f})"
            
            # 构建元数据信息
            meta_info = []
            if custom.get("lender"):
                meta_info.append(f"贷款机构: {custom['lender']}")
            if custom.get("doc_name"):
                meta_info.append(f"文档: {custom['doc_name']}")
            if custom.get("tags"):
                meta_info.append(f"标签: {custom['tags']}")
            
            meta_text = " | ".join(meta_info) if meta_info else "无元数据"
            
            # 显示结果面板
            panel_content = f"[dim]{meta_text}[/dim]\n\n{content}"
            
            console.print(Panel(
                panel_content,
                title=f"[bold blue]{title}[/bold blue]",
                title_align="left",
                border_style="blue",
                padding=(1, 2)
            ))
            console.print()  # 空行分隔
        
    except requests.exceptions.Timeout:
        console.print("[red]❌ 请求超时（30秒）[/red]")
    except requests.exceptions.ConnectionError:
        console.print(f"[red]❌ 无法连接到服务器: {BASE_URL}[/red]")
    except Exception as e:
        console.print(f"[red]❌ 错误: {e}[/red]")


def main():
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]Fannie Mae Eligibility Matrix 查询测试[/bold cyan]\n"
        f"数据集: {DATASET_ID}\n"
        f"服务器: {BASE_URL}",
        border_style="cyan"
    ))
    
    # 定义查询问题
    questions = [
        "What is the maximum LTV ratio for a primary residence purchase with a single unit?",
        "What are the credit score requirements for manually underwritten loans?",
        "What is the maximum debt-to-income ratio for HomeReady loans?",
        "What are the reserve requirements for investment properties?",
        "What are the LTV requirements for cash-out refinance transactions?",
    ]
    
    # 执行查询
    for question in questions:
        query_and_display(question, top_k=3)
        console.print("\n")  # 问题之间多一个空行
    
    console.print("=" * 80)
    console.print("[bold green]✅ 查询完成[/bold green]")
    console.print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
