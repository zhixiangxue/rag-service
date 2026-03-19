"""Scan-to-text PDF conversion using Claude Vision.

Core processing logic shared between the CLI script and the API router.
"""
import base64
import json
import logging
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger("rag.utils.scan2text")

# ---------------------------------------------------------------------------
# PDF rendering helpers
# ---------------------------------------------------------------------------

def wkhtmltopdf_path() -> str:
    """Locate wkhtmltopdf binary or raise RuntimeError."""
    for candidate in [
        shutil.which("wkhtmltopdf"),
        r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe",
        r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltopdf.exe",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "wkhtmltopdf not found. Download: https://wkhtmltopdf.org/downloads.html"
    )


def _single_page_html(page_num: int, md_content: str, font_size: float) -> str:
    """Render markdown content to a standalone HTML page at the given font size."""
    import markdown as md_lib

    html_body = md_lib.markdown(
        md_content,
        extensions=["tables", "fenced_code", "nl2br"],
    )
    table_font = max(font_size - 1.0, 5.0)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
body {{ font-family: Arial, sans-serif; font-size: {font_size:.1f}pt; margin: 0; line-height: 1.4; }}
h1, h2, h3 {{ color: #222; margin: 4px 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0; font-size: {table_font:.1f}pt; }}
th, td {{ border: 1px solid #555; padding: 3px 6px; text-align: left; }}
th {{ background-color: #e0e0e0; font-weight: bold; }}
tr:nth-child(even) td {{ background-color: #f9f9f9; }}
.page-label {{ color: #999; font-size: 7pt; margin-bottom: 4px; }}
</style></head><body>
<p class="page-label">— Page {page_num} —</p>
{html_body}
</body></html>"""


def render_single_page(
    wkhtmltopdf: str,
    page_num: int,
    md_content: str,
    temp_dir: Path,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """
    Render one source page to PDF, auto-shrinking font until it fits in 1 page.

    Returns the path to the rendered single-page PDF.
    """
    import fitz

    font_size = 10.0
    out_path = temp_dir / f"_page_{page_num:04d}.pdf"

    while font_size >= 5.0:
        html = _single_page_html(page_num, md_content, font_size)
        html_path = temp_dir / f"_page_{page_num:04d}.html"
        html_path.write_text(html, encoding="utf-8")

        subprocess.run(
            [
                wkhtmltopdf,
                "--page-size", "A4",
                "--margin-top", "15mm",
                "--margin-bottom", "15mm",
                "--margin-left", "15mm",
                "--margin-right", "15mm",
                "--encoding", "utf-8",
                "--quiet",
                str(html_path),
                str(out_path),
            ],
            check=True,
        )
        html_path.unlink(missing_ok=True)

        doc = fitz.open(str(out_path))
        page_count = len(doc)
        doc.close()

        if page_count <= 1:
            break

        new_size = round(font_size * (0.92 if page_count == 2 else 0.85), 1)
        msg = f"page {page_num}: {page_count} pages at {font_size:.1f}pt, retrying at {new_size:.1f}pt"
        logger.debug(msg)
        if on_progress:
            on_progress(msg)
        font_size = new_size

    return out_path


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_pdf(
    input_path: Path,
    work_dir: Path,
    api_key: str,
    original_stem: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """
    OCR a scanned PDF and produce a searchable text PDF.

    - Resumes from checkpoint.json in work_dir if present.
    - Raises RuntimeError if any page fails after retries.
    - Returns the Path of the output PDF on success.

    Args:
        input_path: Path to the source (scanned) PDF.
        work_dir:   Directory for checkpoint, temp files, and output.
        api_key:    Anthropic API key.
        on_progress: Optional callback for progress messages.
    """
    import fitz
    import anthropic
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    from zag.readers.claude_vision import PARSE_PROMPT, _clean_content

    def _log(msg: str) -> None:
        logger.info(msg)
        if on_progress:
            on_progress(msg)

    checkpoint_path = work_dir / "checkpoint.json"
    stem = original_stem or input_path.stem
    output_path = work_dir / (stem + "_text.pdf")

    # Load checkpoint
    checkpoint: dict[str, str] = {}
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            _log(f"Resuming from checkpoint — {len(checkpoint)} pages already done.")
        except Exception as exc:
            _log(f"Warning: could not read checkpoint ({exc}), starting fresh.")

    # Open PDF and validate it's a scanned document
    pdf_doc = fitz.open(str(input_path))
    total_pages = len(pdf_doc)

    if total_pages > 0:
        sample = [pdf_doc[i] for i in range(min(5, total_pages))]
        avg_chars = sum(len(p.get_text().strip()) for p in sample) / len(sample)
        if avg_chars >= 100:
            pdf_doc.close()
            raise ValueError(
                f"Not a scanned PDF (avg {avg_chars:.0f} chars/page). "
                "This tool is only for image-only scanned documents."
            )

    # Rasterize all pages upfront (no API calls)
    scale = 250 / 72.0
    pages_data: list[tuple[int, str]] = []
    for i in range(total_pages):
        page = pdf_doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        pages_data.append((i + 1, b64))
    pdf_doc.close()

    # OCR pending pages
    client = anthropic.Anthropic(api_key=api_key)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _parse_page(img_b64: str) -> str:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": img_b64,
                    }},
                    {"type": "text", "text": PARSE_PROMPT},
                ],
            }],
        )
        return _clean_content(resp.content[0].text)

    pending = [(n, b) for n, b in pages_data if str(n) not in checkpoint]
    _log(f"Total pages: {total_pages}  |  To process: {len(pending)}  |  Skipped: {total_pages - len(pending)}")

    failed_pages: list[int] = []
    for idx, (page_num, img_b64) in enumerate(pending, 1):
        _log(f"[{idx}/{len(pending)}] OCR page {page_num}/{total_pages}...")
        try:
            content = _parse_page(img_b64)
            checkpoint[str(page_num)] = content
            checkpoint_path.write_text(
                json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _log(f"  page {page_num} OK ({len(content)} chars)")
        except Exception as exc:
            _log(f"  page {page_num} FAILED: {exc}")
            failed_pages.append(page_num)

    if failed_pages:
        raise RuntimeError(
            f"{len(failed_pages)} page(s) failed: {failed_pages}. "
            "Checkpoint kept — re-run to retry."
        )

    # Render pages to PDF
    wkhtmltopdf = wkhtmltopdf_path()
    pages = [(page_num, checkpoint[str(page_num)]) for page_num, _ in pages_data]
    temp_pdfs: list[Path] = []

    for page_num, content in pages:
        _log(f"rendering page {page_num}...")
        p = render_single_page(wkhtmltopdf, page_num, content, work_dir, on_progress)
        temp_pdfs.append(p)

    # Merge
    merged = fitz.open()
    for tp in temp_pdfs:
        src = fitz.open(str(tp))
        merged.insert_pdf(src)
        src.close()

    tmp_output = work_dir / "_merged_tmp.pdf"
    merged.save(str(tmp_output))
    merged.close()

    # Move to final path; fall back to timestamped name if target is locked
    try:
        shutil.move(str(tmp_output), str(output_path))
        final_output = output_path
    except (PermissionError, OSError):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output = work_dir / f"{output_path.stem}_{ts}.pdf"
        shutil.move(str(tmp_output), str(final_output))
        _log(f"Note: {output_path.name} is locked — saved as {final_output.name}")

    # Cleanup
    for tp in temp_pdfs:
        tp.unlink(missing_ok=True)
    checkpoint_path.unlink(missing_ok=True)

    total_chars = sum(len(c) for _, c in pages)
    _log(f"Done. {total_pages}/{total_pages} pages, {total_chars} chars. Saved: {final_output}")

    return final_output
