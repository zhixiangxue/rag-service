"""Utility endpoints — miscellaneous tools (scan2text, etc.)."""
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

router = APIRouter(prefix="/utility", tags=["utility"])

_SCAN2TEXT_BASE = Path.home() / ".zag" / "scan2text"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _work_dir(file_hash: str) -> Path:
    return _SCAN2TEXT_BASE / file_hash


def _latest_result(work_dir: Path) -> Path | None:
    """Return the most recently modified *_text*.pdf in work_dir, or None."""
    candidates = sorted(
        work_dir.glob("*_text*.pdf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _checkpoint_pages(checkpoint: Path) -> int:
    try:
        return len(json.loads(checkpoint.read_text()))
    except Exception:
        return 0


def _task_status(file_hash: str) -> dict[str, Any]:
    """
    Derive task status from filesystem state only.

    States: not_found | running | interrupted | done
    """
    wd = _work_dir(file_hash)
    if not wd.exists():
        return {"status": "not_found"}

    lock_file = wd / "running.lock"
    checkpoint = wd / "checkpoint.json"
    result = _latest_result(wd)

    if lock_file.exists():
        try:
            pid = int(lock_file.read_text().strip())
            if _is_pid_alive(pid):
                return {"status": "running", "pages_done": _checkpoint_pages(checkpoint)}
        except Exception:
            pass
        return {"status": "interrupted", "pages_done": _checkpoint_pages(checkpoint)}

    if result and not checkpoint.exists():
        return {"status": "done", "result_file": result.name}

    if checkpoint.exists():
        return {"status": "interrupted", "pages_done": _checkpoint_pages(checkpoint)}

    return {"status": "pending"}


def _run_scan2text_sync(file_hash: str) -> None:
    """Blocking wrapper around process_pdf; manages lock file lifecycle."""
    from .scan2text import process_pdf

    wd = _work_dir(file_hash)
    lock_file = wd / "running.lock"

    stem_file = wd / "original_stem.txt"
    original_stem = stem_file.read_text(encoding="utf-8").strip() if stem_file.exists() else None

    # Find the stored PDF (named after original filename)
    pdf_candidates = [p for p in wd.glob("*.pdf") if not p.name.endswith("_text.pdf") and not p.name.startswith("_")]
    if not pdf_candidates:
        return
    original_pdf = pdf_candidates[0]

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return

    import os as _os
    lock_file.write_text(str(_os.getpid()))

    try:
        process_pdf(
            input_path=original_pdf,
            work_dir=wd,
            api_key=api_key,
            original_stem=original_stem,
        )
    except Exception:
        pass
    finally:
        lock_file.unlink(missing_ok=True)


async def _run_scan2text(file_hash: str) -> None:
    """Run blocking scan2text in a thread pool so the event loop stays free."""
    await asyncio.to_thread(_run_scan2text_sync, file_hash)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/scan2text/submit", summary="Submit a scanned PDF for text extraction")
async def scan2text_submit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Scanned PDF file"),
) -> dict[str, Any]:
    """
    Upload a scanned PDF. Returns `task_id` (file hash) for status/result queries.

    - Already done: returns status=done, no reprocessing.
    - Already running: returns status=running.
    - Otherwise: saves file to work_dir and starts background extraction.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    import tempfile
    from zag.utils.hash import calculate_file_hash

    raw = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)

    try:
        file_hash = calculate_file_hash(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    wd = _work_dir(file_hash)
    wd.mkdir(parents=True, exist_ok=True)

    status = _task_status(file_hash)

    if status["status"] in ("done", "running"):
        return {"task_id": file_hash, **status}

    # Save (overwrite if interrupted/pending)
    stored_pdf = wd / file.filename
    stored_pdf.write_bytes(raw)
    # Persist original filename stem for output naming
    (wd / "original_stem.txt").write_text(Path(file.filename).stem, encoding="utf-8")

    background_tasks.add_task(_run_scan2text, file_hash)
    return {"task_id": file_hash, "status": "processing_started"}


@router.get("/scan2text/{task_id}/status", summary="Query scan2text task status")
async def scan2text_status(task_id: str) -> dict[str, Any]:
    """
    Return current task status.

    Possible values: not_found | running | interrupted | done
    """
    status = _task_status(task_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")
    return {"task_id": task_id, **status}


@router.get("/scan2text/{task_id}/result", summary="Download the extracted text PDF")
async def scan2text_result(task_id: str) -> FileResponse:
    """
    Stream the latest result PDF for the given task.

    Returns 404 if the task does not exist or is not yet complete.
    """
    wd = _work_dir(task_id)
    if not wd.exists():
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")

    result = _latest_result(wd)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Result not ready yet. Check /status first.",
        )

    return FileResponse(
        path=str(result),
        media_type="application/pdf",
        filename=result.name,
    )


@router.post("/scan2text/{task_id}/retry", summary="Retry a failed or interrupted task")
async def scan2text_retry(task_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """
    Re-trigger extraction for an interrupted or failed task.
    Resumes from checkpoint — already-processed pages are skipped.

    Returns 404 if task was never submitted.
    Returns 409 if task is already running.
    """
    wd = _work_dir(task_id)
    pdf_candidates = [p for p in wd.glob("*.pdf") if not p.name.endswith("_text.pdf") and not p.name.startswith("_")] if wd.exists() else []
    if not wd.exists() or not pdf_candidates:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found.")

    status = _task_status(task_id)

    if status["status"] == "running":
        raise HTTPException(status_code=409, detail="Task is already running.")

    if status["status"] == "done":
        return {"task_id": task_id, "message": "Already completed.", **status}

    # Clear stale lock
    (wd / "running.lock").unlink(missing_ok=True)

    background_tasks.add_task(_run_scan2text, task_id)
    return {"task_id": task_id, "status": "retry_started"}
