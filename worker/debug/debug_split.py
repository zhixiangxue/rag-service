#!/usr/bin/env python3
"""
Debug: read + split a document locally.

Usage:
    python debug_split.py <file_path>                    # basic (no LLM)
    python debug_split.py <file_path> --enhance          # with Claude table enhancement
    python debug_split.py <file_path> --pages 1-50       # read pages 1-50 only
    python debug_split.py <file_path> --enhance --pages 1-50
    python debug_split.py                                # interactive prompt
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parents[3] / ".env")

import tiktoken
from rich.console import Console

from zag.splitters import (
    MarkdownHeaderSplitter,
    TextSplitter,
    TableSplitter,
    RecursiveMergingSplitter,
)

console = Console()
OUTPUT_DIR = Path(__file__).parent / "output"


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #

def read_document(file_path: Path, enhance: bool = False, page_range=None):
    """Read document with no LLM involvement."""
    ext = file_path.suffix.lower()

    if ext in (".docx", ".doc"):
        from zag.readers.docling import DoclingReader
        console.print(f"Reading Word document: {file_path.name}")
        reader = DoclingReader()
        doc = reader.read(str(file_path))
        console.print(f"  {len(doc.content):,} chars, {len(doc.pages)} pages")
        return doc

    elif ext in (".md", ".txt", ".markdown"):
        from zag.readers.markitdown import MarkItDownReader
        console.print(f"Reading plain text: {file_path.name}")
        reader = MarkItDownReader()
        doc = reader.read(str(file_path))
        console.print(f"  {len(doc.content):,} chars")
        return doc

    elif ext == ".pdf":
        from zag.readers.pymupdf4llm import PyMuPDF4LLMReader
        if enhance:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            console.print(f"Reading PDF with PyMuPDF4LLM + Claude table enhancement: {file_path.name}")
            reader = PyMuPDF4LLMReader(enhance_tables=True, anthropic_api_key=api_key)
        else:
            console.print(f"Reading PDF with PyMuPDF4LLM: {file_path.name}")
            reader = PyMuPDF4LLMReader()
        doc = reader.read(str(file_path), page_range=page_range)
        console.print(f"  {len(doc.content):,} chars, {len(doc.pages)} pages")
        return doc

    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: .pdf .docx .doc .md .txt")


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #

def split_document(doc, max_chunk_tokens=1200, table_max_tokens=1500, target_token_size=800):
    """Run the standard splitter pipeline synchronously."""
    pipeline = (
        MarkdownHeaderSplitter()
        | TextSplitter(max_chunk_tokens=max_chunk_tokens)
        | TableSplitter(max_chunk_tokens=table_max_tokens)
        | RecursiveMergingSplitter(target_token_size=target_token_size)
    )
    t0 = time.time()
    units = doc.split(pipeline)
    console.print(f"Split in {time.time() - t0:.2f}s  →  {len(units)} units")
    return units


# --------------------------------------------------------------------------- #
# Token distribution
# --------------------------------------------------------------------------- #

def print_token_distribution(units, tokenizer):
    """Print token size distribution to console and return counts list."""
    token_counts = [len(tokenizer.encode(u.content)) for u in units]

    avg = sum(token_counts) // len(token_counts)
    console.print(
        f"\nToken range: {min(token_counts)} – {max(token_counts)}  "
        f"(avg {avg})"
    )

    ranges = [
        ("Tiny    < 200",        0,    200),
        ("Small   200–500",    200,    500),
        ("Medium  500–1000",   500,   1000),
        ("Large   1000–1500", 1000,   1500),
        ("Oversized > 1500",  1500, float("inf")),
    ]

    console.print("\nDistribution:")
    for label, lo, hi in ranges:
        count = sum(1 for t in token_counts if lo <= t < hi)
        if count == 0:
            continue
        pct = count / len(token_counts) * 100
        bar = "█" * int(pct / 2)
        console.print(f"  {label:<22}  {count:>4}  ({pct:>5.1f}%)  {bar}")

    oversized = [(i, t) for i, t in enumerate(token_counts) if t > 1500]
    if oversized:
        console.print(f"\n  [!] {len(oversized)} oversized units:", style="yellow")
        for idx, t in oversized[:10]:
            console.print(f"      unit {idx:>4}  {t} tokens", style="yellow")

    return token_counts


# --------------------------------------------------------------------------- #
# Visualization file
# --------------------------------------------------------------------------- #

def export_visualization(units, token_counts, doc_name: str, enhanced: bool = False) -> Path:
    """Write each unit separated by a clear divider into a markdown file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_enhanced" if enhanced else ""
    out_file = OUTPUT_DIR / f"{doc_name}_split{suffix}.md"

    with open(out_file, "w", encoding="utf-8") as f:
        avg = sum(token_counts) // len(token_counts)
        f.write(f"# Split Visualization — {doc_name}\n\n")
        f.write(f"**Total units**: {len(units)}\n\n")
        f.write(
            f"**Token range**: {min(token_counts)} – {max(token_counts)}"
            f"  (avg {avg})\n\n"
        )
        f.write("---\n")

        for i, (unit, tokens) in enumerate(zip(units, token_counts)):
            tag = ""
            if tokens > 1500:
                tag = "  [OVERSIZED]"
            elif tokens >= 1000:
                tag = "  [LARGE]"

            f.write(f"\n\n{'=' * 80}\n\n")
            f.write(f"## Unit {i}  |  {tokens} tokens{tag}\n\n")

            if hasattr(unit, "metadata") and unit.metadata:
                ctx = getattr(unit.metadata, "context_path", None)
                pages = getattr(unit.metadata, "page_numbers", None)
                if ctx:
                    f.write(f"**Context**: {ctx}\n\n")
                if pages:
                    f.write(f"**Pages**: {pages}\n\n")

            f.write(unit.content)
            f.write("\n")

    console.print(f"\nVisualization -> {out_file}")
    return out_file


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main():
    # Parse args: optional --enhance flag and --pages start-end
    args = sys.argv[1:]
    enhance = "--enhance" in args

    # Parse --pages N-M
    page_range = None
    for a in args:
        if a.startswith("--pages"):
            val = a.split("=", 1)[-1] if "=" in a else None
            if val is None:
                idx = args.index(a)
                val = args[idx + 1] if idx + 1 < len(args) else None
            if val and "-" in val:
                try:
                    s, e = val.split("-", 1)
                    page_range = (int(s), int(e))
                except ValueError:
                    pass

    paths = [a for a in args if not a.startswith("--") and not (page_range and a == f"{page_range[0]}-{page_range[1]}")]

    if paths:
        file_path = Path(paths[0])
    else:
        raw = input("File path: ").strip()
        # Handle Windows PowerShell drag-and-drop: & 'C:\path with spaces\file.pdf'
        raw = raw.lstrip("&").strip().strip('"').strip("'")
        file_path = Path(raw)

    if not file_path.exists():
        console.print(f"[red]Not found: {file_path}[/red]")
        sys.exit(1)

    console.print(f"\nFile    : {file_path}")
    console.print(f"Size    : {file_path.stat().st_size / 1024:.1f} KB")
    console.print(f"Enhance : {enhance}")
    if page_range:
        console.print(f"Pages   : {page_range[0]}-{page_range[1]}")
    console.print()

    doc = read_document(file_path, enhance=enhance, page_range=page_range)

    # Dump raw extraction for inspection
    dump_dir = doc.dump(OUTPUT_DIR / "dumps")
    console.print(f"Dump    -> {dump_dir}")

    units = split_document(doc)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    token_counts = print_token_distribution(units, tokenizer)

    export_visualization(units, token_counts, file_path.stem, enhanced=enhance)


if __name__ == "__main__":
    main()
