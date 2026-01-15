#!/usr/bin/env python3
"""
Check Qdrant Data - 查看 Qdrant 中的数据

快速检查 Qdrant collection 中的数据量和样例
"""

from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

console = Console()

# 配置
QDRANT_HOST = "localhost"
QDRANT_PORT = 16333
QDRANT_GRPC_PORT = 16334
COLLECTION_NAME = "mortgage_guidelines"


def main():
    console.print("\n" + "=" * 70)
    console.print("  🔍 Qdrant Data Inspection", style="bold cyan")
    console.print("=" * 70 + "\n")
    
    # 连接 Qdrant
    console.print(f"📡 Connecting to Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    try:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=True
        )
        console.print("   ✅ Connected successfully\n")
    except Exception as e:
        console.print(f"   ❌ Connection failed: {e}\n", style="bold red")
        return
    
    # 获取所有 collections
    try:
        collections = client.get_collections()
        console.print(f"📚 Available Collections: {len(collections.collections)}")
        
        table = Table(show_header=True, title="Collections")
        table.add_column("Name", style="cyan")
        table.add_column("Status", style="yellow")
        
        for col in collections.collections:
            table.add_row(col.name, "Active")
        
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"❌ Failed to get collections: {e}\n", style="bold red")
        return
    
    # 检查目标 collection
    console.print(f"🎯 Inspecting Collection: [cyan]{COLLECTION_NAME}[/cyan]\n")
    
    try:
        # 获取 collection 信息
        collection_info = client.get_collection(COLLECTION_NAME)
        
        console.print(f"📊 Collection Info:")
        console.print(f"   - Vectors count: {collection_info.points_count}")
        console.print(f"   - Vector size: {collection_info.config.params.vectors.size}")
        console.print(f"   - Distance: {collection_info.config.params.vectors.distance}")
        console.print()
        
        if collection_info.points_count == 0:
            console.print("⚠️  Collection is empty", style="yellow")
            return
        
        # 滚动获取前 5 个点
        console.print("📄 Sample Data (first 5 points):\n")
        
        points = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        for i, point in enumerate(points[0], 1):
            console.print(f"{'─' * 70}")
            console.print(f"[bold cyan]Point {i}[/bold cyan]")
            console.print(f"{'─' * 70}")
            console.print(f"ID: {point.id}")
            
            # 显示 payload 大小
            if point.payload:
                import json
                import sys
                payload_json = json.dumps(point.payload)
                payload_size = len(payload_json.encode('utf-8'))
                console.print(f"\n[yellow]💾 Payload Size: {payload_size / 1024:.2f} KB[/yellow]")
                
                # 分析各个字段的大小
                if isinstance(point.payload, dict):
                    console.print(f"\n[dim]Field sizes:[/dim]")
                    for key, value in point.payload.items():
                        try:
                            field_size = len(json.dumps(value).encode('utf-8'))
                            console.print(f"  - {key}: {field_size / 1024:.2f} KB")
                        except:
                            console.print(f"  - {key}: (cannot serialize)")
            
            if point.payload:
                console.print("\nPayload:")
                
                # 显示关键字段
                key_fields = [
                    'content', 'unit_type', 'doc_id', 
                    'context_path', 'page_numbers',
                    'lender', 'pdf_name', 'tags'
                ]
                
                # 先显示所有可用的字段
                console.print(f"\n  Available fields: {list(point.payload.keys())}")
                
                for key in key_fields:
                    if key in point.payload:
                        value = point.payload[key]
                        
                        # 内容截断显示
                        if key == 'content' and isinstance(value, str) and len(value) > 200:
                            value = value[:200] + "..."
                        
                        console.print(f"  • {key}: {value}")
                
                # 显示 custom metadata（如果有）
                if 'metadata' in point.payload:
                    console.print("\n  Metadata Object:")
                    metadata_obj = point.payload['metadata']
                    
                    # 显示 metadata 的关键字段
                    if isinstance(metadata_obj, dict):
                        for k, v in list(metadata_obj.items())[:10]:  # 只显示前10个
                            if k != 'document':  # document 太长，跳过
                                console.print(f"    - {k}: {v}")
                    
                    # 特别显示 custom 字段
                    if isinstance(metadata_obj, dict) and 'custom' in metadata_obj:
                        console.print("\n  Custom Metadata (from metadata.custom):")
                        for k, v in metadata_obj['custom'].items():
                            console.print(f"    - {k}: {v}")
            
            console.print()
        
        # 统计信息
        console.print(f"\n{'=' * 70}")
        console.print("  📈 Statistics", style="bold green")
        console.print(f"{'=' * 70}\n")
        
        # 按 lender 统计
        console.print("📊 Count by Lender:")
        
        # 使用 scroll 获取所有点的 lender 和 doc_id 信息
        all_lenders = {}
        all_doc_ids = {}
        offset = None
        
        while True:
            result = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=True,  # 获取完整 payload
                with_vectors=False
            )
            
            points_batch, offset = result
            
            if not points_batch:
                break
            
            for point in points_batch:
                # 从 metadata.custom.lender 读取
                lender = 'Unknown'
                if 'metadata' in point.payload:
                    metadata_obj = point.payload['metadata']
                    if isinstance(metadata_obj, dict) and 'custom' in metadata_obj:
                        custom = metadata_obj['custom']
                        if isinstance(custom, dict):
                            lender = custom.get('lender', 'Unknown')
                
                all_lenders[lender] = all_lenders.get(lender, 0) + 1
                
                # 统计 doc_id
                doc_id = point.payload.get('doc_id', 'Unknown')
                all_doc_ids[doc_id] = all_doc_ids.get(doc_id, 0) + 1
            
            if offset is None:
                break
        
        # 显示统计
        lender_table = Table(show_header=True)
        lender_table.add_column("Lender", style="cyan")
        lender_table.add_column("Units Count", justify="right", style="green")
        lender_table.add_column("Percentage", justify="right", style="yellow")
        
        total = sum(all_lenders.values())
        for lender, count in sorted(all_lenders.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            lender_table.add_row(lender, str(count), f"{pct:.1f}%")
        
        console.print(lender_table)
        console.print(f"\n✅ Total Units: {total}")
        
        # 按 doc_id 统计
        console.print("\n\n📊 Count by Doc ID:")
        
        doc_table = Table(show_header=True)
        doc_table.add_column("Doc ID", style="cyan")
        doc_table.add_column("Units Count", justify="right", style="green")
        doc_table.add_column("Percentage", justify="right", style="yellow")
        
        for doc_id, count in sorted(all_doc_ids.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            doc_table.add_row(doc_id, str(count), f"{pct:.1f}%")
        
        console.print(doc_table)
        console.print(f"\n✅ Total Doc IDs: {len(all_doc_ids)}")
        
    except Exception as e:
        console.print(f"❌ Error inspecting collection: {e}", style="bold red")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
