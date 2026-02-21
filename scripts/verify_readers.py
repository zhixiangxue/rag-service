#!/usr/bin/env python3
"""
Post-deployment verification script for document readers
Tests MinerU and Docling readers to ensure they work correctly
"""
import sys
import tempfile
import urllib.request
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress

console = Console()

# Test PDF URL (small file for quick test)
TEST_PDF_URL = "https://wcbpub.oss-cn-hangzhou.aliyuncs.com/xue/zeitro/guidelines/usda.pdf"
# You can replace with your own PDF URL

def download_test_pdf() -> Path:
    """Download test PDF to temp directory"""
    console.print("\n[cyan]📥 Downloading test PDF...[/cyan]")
    
    temp_dir = Path(tempfile.gettempdir()) / "rag_reader_test"
    temp_dir.mkdir(exist_ok=True)
    
    pdf_path = temp_dir / "test.pdf"
    
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Downloading...", total=100)
            
            def report_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, (downloaded / total_size) * 100)
                    progress.update(task, completed=percent)
            
            urllib.request.urlretrieve(TEST_PDF_URL, pdf_path, reporthook=report_hook)
        
        console.print(f"[green]✓ Downloaded to: {pdf_path}[/green]")
        console.print(f"[dim]  Size: {pdf_path.stat().st_size / 1024:.1f} KB[/dim]\n")
        return pdf_path
        
    except Exception as e:
        console.print(f"[red]✗ Download failed: {e}[/red]")
        return None


def test_mineru_reader(pdf_path: Path) -> tuple[bool, str]:
    """Test MinerU reader"""
    try:
        from zag.readers.mineru import MinerUReader
        
        console.print("[cyan]Testing MinerU...[/cyan]")
        reader = MinerUReader()
        pdf = reader.read(str(pdf_path), page_range=(1, 1))  # Read first page only
        
        if len(pdf.pages) > 0 and len(pdf.content) > 0:
            return True, f"OK ({len(pdf.pages)} pages, {len(pdf.content)} chars)"
        else:
            return False, "Read succeeded but no content extracted"
            
    except Exception as e:
        error_msg = str(e)
        # Shorten long error messages
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."
        return False, error_msg


def test_docling_reader(pdf_path: Path) -> tuple[bool, str]:
    """Test Docling reader"""
    try:
        from zag.readers.docling import DoclingReader
        
        console.print("[cyan]Testing Docling...[/cyan]")
        reader = DoclingReader()
        pdf = reader.read(str(pdf_path), page_range=(1, 1))  # Read first page only
        
        if len(pdf.pages) > 0 and len(pdf.content) > 0:
            return True, f"OK ({len(pdf.pages)} pages, {len(pdf.content)} chars)"
        else:
            return False, "Read succeeded but no content extracted"
            
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 100:
            error_msg = error_msg[:97] + "..."
        return False, error_msg


def main():
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]📋 Post-Deployment Reader Verification[/bold cyan]")
    console.print("=" * 60 + "\n")
    
    # Download test PDF
    pdf_path = download_test_pdf()
    if pdf_path is None:
        console.print("\n[red]❌ Cannot proceed without test PDF[/red]")
        console.print("[yellow]Please check your internet connection or provide a local PDF[/yellow]\n")
        sys.exit(1)
    
    # Test readers
    console.print("[bold yellow]🧪 Testing Readers...[/bold yellow]\n")
    
    results = []
    
    # Test MinerU
    mineru_ok, mineru_msg = test_mineru_reader(pdf_path)
    results.append(("MinerU", mineru_ok, mineru_msg))
    
    # Test Docling
    docling_ok, docling_msg = test_docling_reader(pdf_path)
    results.append(("Docling", docling_ok, docling_msg))
    
    # Display results table
    console.print("\n[bold cyan]📊 Verification Results[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Reader", style="cyan", width=15)
    table.add_column("Status", width=10)
    table.add_column("Details", width=50)
    
    all_ok = True
    for reader_name, success, message in results:
        if success:
            status = "[green]✓ PASS[/green]"
        else:
            status = "[red]✗ FAIL[/red]"
            all_ok = False
        
        table.add_row(reader_name, status, message)
    
    console.print(table)
    
    # Cleanup
    if pdf_path and pdf_path.exists():
        pdf_path.unlink()
        console.print(f"\n[dim]Cleaned up test file: {pdf_path}[/dim]")
    
    # Final verdict
    console.print("\n" + "=" * 60)
    if all_ok:
        console.print("[bold green]✅ All readers verified successfully![/bold green]")
        console.print("[dim]Your deployment is ready for production use.[/dim]")
        console.print("=" * 60 + "\n")
        sys.exit(0)
    else:
        console.print("[bold red]❌ Some readers failed verification[/bold red]")
        console.print("[yellow]⚠️  Please check the error messages above and fix the issues.[/yellow]")
        console.print("[yellow]⚠️  Common issues:[/yellow]")
        console.print("[yellow]  - Missing CUDA Toolkit (for MinerU)[/yellow]")
        console.print("[yellow]  - vLLM/xformers version conflicts[/yellow]")
        console.print("[yellow]  - Missing system dependencies[/yellow]")
        console.print("=" * 60 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
