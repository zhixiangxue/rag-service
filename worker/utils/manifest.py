"""Manifest preparation for document processing."""
import json
from pathlib import Path
from PyPDF2 import PdfReader
from rich.console import Console
from typing import Dict, Any, Optional

from ..config import MAX_PAGES_PER_PART
from .pdf_splitter import split_pdf_with_overlap
from zag.utils.hash import calculate_file_hash

console = Console()


async def prepare_single_file_manifest(
    file_path: Path,
    workspace_dir: Path,
    custom_metadata: Optional[Dict[str, Any]] = None
) -> dict:
    """
    Prepare manifest for a single file.
    
    Args:
        file_path: Path to the PDF file
        workspace_dir: Working directory for this file
        custom_metadata: Optional business metadata
    
    Returns:
        Manifest dict with processing tasks
    """
    console.print(f"\n[cyan]📋 Preparing manifest for: {file_path.name}[/cyan]")
    
    # Calculate source file hash
    source_hash = calculate_file_hash(file_path)
    console.print(f"   🔑 Hash: {source_hash}")
    
    # Check page count
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        console.print(f"   📖 Pages: {total_pages}")
    except Exception as e:
        raise Exception(f"Failed to read PDF: {e}")
    
    # Create temp_parts directory inside workspace
    temp_dir = workspace_dir / "temp_parts"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create task
    task = {
        "source_file": str(file_path.absolute()),
        "source_hash": source_hash,
        "metadata": custom_metadata or {},
        "total_pages": total_pages,
        "parts": []
    }
    
    # Split if needed
    if total_pages > MAX_PAGES_PER_PART:
        console.print(f"   ✂️  Large file, splitting into parts...")
        
        try:
            part_files = split_pdf_with_overlap(
                input_pdf=str(file_path),
                output_dir=str(temp_dir),
                pages_per_part=MAX_PAGES_PER_PART,
                overlap_pages=0
            )
            
            for part_file in part_files:
                part_name = Path(part_file).stem
                page_range = part_name.split('_pages_')[-1] if '_pages_' in part_name else "unknown"
                
                task["parts"].append({
                    "file": str(part_file),
                    "page_range": page_range
                })
            
            console.print(f"   ✅ Split into {len(part_files)} parts")
            
        except Exception as e:
            console.print(f"   [yellow]⚠️  Failed to split, using original: {e}[/yellow]")
            # Fallback: use original file
            task["parts"].append({
                "file": str(file_path.absolute()),
                "page_range": f"1-{total_pages}"
            })
    else:
        console.print(f"   ✅ Small file, no splitting needed")
        task["parts"].append({
            "file": str(file_path.absolute()),
            "page_range": f"1-{total_pages}"
        })
    
    # Save manifest to workspace
    manifest = {"tasks": [task]}
    manifest_path = workspace_dir / "manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    console.print(f"   💾 Manifest saved: {manifest_path}")
    console.print(f"   📦 Parts to process: {len(task['parts'])}\n")
    
    return manifest
