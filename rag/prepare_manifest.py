#!/usr/bin/env python3
"""
Prepare Manifest - Preprocess documents and create processing manifest

Scans a directory for PDF files and their metadata JSON files,
checks page counts, splits large PDFs if needed, and generates
a manifest.json file for batch processing.

Features:
- Scans directory for PDF + JSON pairs
- Calculates source file hash for ID consistency
- Splits PDFs >100 pages into parts
- Generates manifest.json with all processing tasks
"""

import json
import sys
from pathlib import Path
from PyPDF2 import PdfReader
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Import from local modules
from extract_pdf_pages import split_pdf_with_overlap
from zag.utils.hash import calculate_file_hash

console = Console()


def scan_directory(
    input_dir: Path,
    max_pages_per_part: int = 100
) -> dict:
    """
    扫描目录，准备处理清单
    
    Args:
        input_dir: 输入目录（包含 PDF 和 JSON 文件）
        max_pages_per_part: 每个 part 最大页数
    
    Returns:
        Manifest dict with processing tasks
    """
    console.print("\n" + "=" * 70)
    console.print("  📋 Preparing Processing Manifest", style="bold cyan")
    console.print("=" * 70 + "\n")
    
    input_dir = Path(input_dir)
    
    # temp_dir 放在输入目录下
    temp_dir = input_dir / "temp_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # 扫描 PDF 文件
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]❌ No PDF files found in {input_dir}[/red]")
        return {"tasks": []}
    
    console.print(f"📂 Scanning directory: {input_dir}")
    console.print(f"   Found {len(pdf_files)} PDF files\n")
    
    manifest = {"tasks": []}
    stats = {
        "total_files": 0,
        "small_files": 0,
        "large_files": 0,
        "total_parts": 0,
        "missing_json": 0
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(pdf_files))
        
        for pdf_path in pdf_files:
            json_path = pdf_path.with_suffix('.json')
            
            # 计算源文件 hash
            console.print(f"\n📄 {pdf_path.name}")
            source_hash = calculate_file_hash(pdf_path)
            console.print(f"   🔑 Hash: {source_hash}")
            
            # 检查页数
            try:
                reader = PdfReader(pdf_path)
                total_pages = len(reader.pages)
                console.print(f"   📖 Pages: {total_pages}")
            except Exception as e:
                console.print(f"   [red]❌ Failed to read PDF: {e}[/red]")
                progress.update(task, advance=1)
                continue
            
            # 检查 JSON 文件
            if not json_path.exists():
                console.print(f"   [yellow]⚠️  No metadata JSON found[/yellow]")
                stats["missing_json"] += 1
            
            # 创建任务
            processing_task = {
                "source_file": str(pdf_path.absolute()),
                "source_hash": source_hash,
                "metadata_file": str(json_path.absolute()) if json_path.exists() else None,
                "total_pages": total_pages,
                "parts": []
            }
            
            # 判断是否需要切分
            if total_pages > max_pages_per_part:
                # 需要切分
                console.print(f"   ✂️  Large file, splitting into parts...")
                stats["large_files"] += 1
                
                try:
                    part_files = split_pdf_with_overlap(
                        input_pdf=str(pdf_path),
                        output_dir=str(temp_dir),
                        pages_per_part=max_pages_per_part,
                        overlap_pages=0
                    )
                    
                    for part_file in part_files:
                        # 从文件名提取页码范围 (e.g., "file_pages_1-100.pdf")
                        part_name = Path(part_file).stem
                        page_range = part_name.split('_pages_')[-1] if '_pages_' in part_name else "unknown"
                        
                        processing_task["parts"].append({
                            "file": str(part_file),
                            "page_range": page_range
                        })
                        stats["total_parts"] += 1
                    
                    console.print(f"   ✅ Split into {len(part_files)} parts")
                    
                except Exception as e:
                    console.print(f"   [red]❌ Failed to split: {e}[/red]")
                    # Fallback: 使用原文件
                    processing_task["parts"].append({
                        "file": str(pdf_path.absolute()),
                        "page_range": f"1-{total_pages}"
                    })
                    stats["total_parts"] += 1
            else:
                # 直接处理
                console.print(f"   ✅ Small file, no splitting needed")
                stats["small_files"] += 1
                processing_task["parts"].append({
                    "file": str(pdf_path.absolute()),
                    "page_range": f"1-{total_pages}"
                })
                stats["total_parts"] += 1
            
            manifest["tasks"].append(processing_task)
            stats["total_files"] += 1
            progress.update(task, advance=1)
    
    # 打印统计
    console.print("\n" + "=" * 70)
    console.print("  📊 Scanning Summary", style="bold green")
    console.print("=" * 70 + "\n")
    
    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="green", justify="right")
    
    table.add_row("Total PDF files", str(stats["total_files"]))
    table.add_row("Small files (≤100 pages)", str(stats["small_files"]))
    table.add_row("Large files (>100 pages)", str(stats["large_files"]))
    table.add_row("Total parts to process", str(stats["total_parts"]))
    table.add_row("Missing JSON files", str(stats["missing_json"]))
    
    console.print(table)
    
    return manifest


def save_manifest(manifest: dict, output_path: Path) -> None:
    """保存 manifest 到文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n✅ Manifest saved: {output_path}")
    console.print(f"   Tasks: {len(manifest['tasks'])}")
    total_parts = sum(len(task['parts']) for task in manifest['tasks'])
    console.print(f"   Total parts: {total_parts}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare processing manifest for PDF documents"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Input directory containing PDF and JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output manifest file path (default: <input_dir>/manifest.json)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=100,
        help="Maximum pages per part (default: 100)"
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    # 如果没有指定 output，默认放在 input_dir 里
    if args.output is None:
        output_path = input_dir / "manifest.json"
    else:
        output_path = Path(args.output)
    
    # 扫描目录
    manifest = scan_directory(
        input_dir=input_dir,
        max_pages_per_part=args.max_pages
    )
    
    if not manifest["tasks"]:
        console.print("\n[red]❌ No tasks generated[/red]")
        sys.exit(1)
    
    # 保存 manifest
    save_manifest(manifest, output_path)
    
    console.print("\n🎉 Preparation complete!", style="bold green")
    console.print(f"\n💡 Next step: Run import.py with the manifest file")
    console.print(f"   python import.py {output_path}")


if __name__ == "__main__":
    main()
