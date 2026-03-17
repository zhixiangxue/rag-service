"""Document API endpoints."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import FileResponse, StreamingResponse
from typing import List, Optional, Dict, Any
import os
import json
import sqlite3
import tempfile
from pathlib import Path

from zag.utils.hash import calculate_file_hash
from ..utils.s3 import get_s3_object_info, download_file_from_s3_async
from ..database import get_connection, now, generate_id
from ..domain.deps import Rule, DependencySource
from ..schemas import (
    DocumentResponse,
    ApiResponse,
    MessageResponse,
    TaskResponse,
    ProcessingMode,
    ReaderType,
    _extract_tags,
    LocatePageRequest,
    LocatePageResult,
    LocatePageResponse,
    CacheUploadResponse,
)
from ..storage import get_storage
from ..constants import TaskStatus, DocumentStatus
from .. import config

router = APIRouter(prefix="/datasets/{dataset_id}/documents", tags=["documents"])


def _build_file_url(stored_path: str, base_dir_str: Optional[str] = None) -> Optional[str]:
    """Build file_url from stored file path (s3:// or local)."""
    if not stored_path:
        return None
    if stored_path.startswith("s3://"):
        return stored_path
    # Local file: pure string processing, no filesystem calls
    norm_path = stored_path.replace("\\", "/")
    if base_dir_str is None:
        base_dir_str = Path(get_storage().base_dir).resolve().as_posix()
    # Make absolute if relative
    if not norm_path.startswith("/"):
        norm_path = Path.cwd().as_posix() + "/" + norm_path
    # Strip base_dir prefix to get relative path
    base = base_dir_str.rstrip("/") + "/"
    if norm_path.startswith(base):
        rel_path = norm_path[len(base):]
        return f"http://{config.API_PUBLIC_HOST}:{config.API_PORT}/files/{rel_path}"
    return None


def _validate_metadata(metadata: dict) -> dict:
    """
    Validate metadata dict structure.

    Args:
        metadata: Metadata dict

    Returns:
        The validated metadata dict

    Raises:
        ValueError: If metadata is invalid or guideline/overlays values are not allowed
    """
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")

    # Validate guideline and overlays fields
    VALID_GUIDELINES = {"FannieMae", "FreddieMac", "VA", "USDA", "FHA"}

    if "guideline" in metadata:
        guideline_val = metadata["guideline"]
        if guideline_val not in VALID_GUIDELINES:
            raise ValueError(
                f'Invalid guideline value: "{guideline_val}". '
                f"Must be one of: {sorted(VALID_GUIDELINES)}"
            )

    if "overlays" in metadata:
        overlays = metadata["overlays"]
        if not isinstance(overlays, list):
            raise ValueError('overlays must be an array')
        for item in overlays:
            if item not in VALID_GUIDELINES:
                raise ValueError(
                    f'Invalid overlays value: "{item}". '
                    f"Must be one of: {sorted(VALID_GUIDELINES)}"
                )

    return metadata


def _create_document_record(
    dataset_id: str,
    file_path: str,
    filename: str,
    metadata: Optional[dict] = None,
    s3_url: Optional[str] = None,
) -> dict:
    """
    Create document record after a local file is saved.
    Handles dataset validation, hash calculation, duplicate check, and database insert.

    Args:
        dataset_id: Dataset ID
        file_path: Local path to the saved file (used for hash/size calculation)
        filename: Original filename
        metadata: Validated metadata dict
        s3_url: If provided, stored as file_path in DB instead of local path

    Returns:
        dict with doc_id, file_name, file_path, file_hash, and status info

    Raises:
        ValueError: If metadata is invalid
        HTTPException: If dataset not found (404)
    """
    # The path actually stored in DB
    db_path = s3_url if s3_url else file_path
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Check if dataset exists
        cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Calculate file hash using zag's utility on saved file
        file_hash = calculate_file_hash(file_path)

        # Check for duplicate: same dataset + same file_hash
        cursor.execute(
            "SELECT id, file_name, file_path FROM documents WHERE dataset_id = ? AND file_hash = ?",
            (dataset_id, file_hash)
        )
        existing_doc = cursor.fetchone()

        if existing_doc:
            # Duplicate: update file_path to latest and return
            cursor.execute(
                "UPDATE documents SET file_path = ?, updated_at = ? WHERE id = ?",
                (db_path, now(), existing_doc['id'])
            )
            conn.commit()
            return {
                "dataset_id": str(dataset_id),
                "doc_id": str(existing_doc['id']),
                "file_name": existing_doc['file_name'],
                "file_hash": file_hash,
                "is_duplicate": True
            }

        # Extract workspace directory (parent of file)
        workspace_dir = os.path.dirname(db_path)

        file_size = os.path.getsize(file_path)
        file_type = filename.split(".")[-1] if "." in filename else "unknown"

        # Serialize metadata to JSON for storage
        if metadata is None:
            raise ValueError(
                "metadata is required. Please provide a JSON object. "
                'Recommended fields: {"lender": "xxx", "guideline": "FannieMae|...", '
                '"overlays": ["xxx"], "tags": ["xxx"]}'
            )
        metadata_json = json.dumps(metadata)

        timestamp = now()
        # Use file_hash as doc_id to ensure same content gets same ID
        doc_id = file_hash

        # Create Document record with file_hash and metadata
        # Use INSERT OR IGNORE to handle race conditions
        try:
            cursor.execute(
                """
                INSERT INTO documents (id, dataset_id, file_name, file_path, workspace_dir, file_size, file_type, file_hash, metadata, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, dataset_id, filename, db_path, workspace_dir, file_size, file_type, file_hash, metadata_json, DocumentStatus.PROCESSING, timestamp, timestamp)
            )
            conn.commit()

            return {
                "dataset_id": str(dataset_id),
                "doc_id": str(doc_id),
                "file_name": filename,
                "file_path": db_path,
                "file_hash": file_hash,
                "is_duplicate": False
            }
        except sqlite3.IntegrityError:
            # Race condition: another request inserted the same file first, just return duplicate
            conn.rollback()
            cursor.execute(
                "SELECT id, file_name FROM documents WHERE dataset_id = ? AND file_hash = ?",
                (dataset_id, file_hash)
            )
            existing_doc = cursor.fetchone()
            return {
                "dataset_id": str(dataset_id),
                "doc_id": str(existing_doc['id']),
                "file_name": existing_doc['file_name'],
                "file_hash": file_hash,
                "is_duplicate": True
            }
    finally:
        conn.close()


def _cache_pdf_to_files_dir(src_path: str, doc_id: str) -> None:
    """Copy uploaded PDF to PDF_FILES_DIR/{doc_id}.pdf for locate_pages lookup.

    Only copies .pdf files. Skips silently if file already exists or on any error
    so the main upload flow is never affected.

    Args:
        src_path: Local path to the source PDF file
        doc_id: Document ID used as the target filename
    """
    import shutil
    if not src_path.lower().endswith(".pdf"):
        return
    try:
        dest_dir = Path(config.PDF_FILES_DIR)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{doc_id}.pdf"
        if not dest.exists():
            shutil.copy2(src_path, dest)
    except Exception as e:
        print(f"[WARN] Failed to cache PDF to PDF_FILES_DIR for doc_id {doc_id}: {e}")


async def upload_file(
    dataset_id: str,
    file: UploadFile = File(...),
    metadata: str = Form(...)
):
    """Upload file to dataset."""
    # Parse metadata string (form data is always str)
    try:
        metadata_dict = json.loads(metadata)
        if not isinstance(metadata_dict, dict):
            raise ValueError("Metadata must be a JSON object")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Metadata must be valid JSON")
    try:
        _validate_metadata(metadata_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Save file using storage abstraction
    storage = get_storage()
    file_path = storage.save(file.file, file.filename, dataset_id)

    # Create document record (handles dataset validation, hash, duplicate check, insert)
    try:
        result = _create_document_record(dataset_id, file_path, file.filename, metadata_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Cache PDF to PDF_FILES_DIR for locate_pages fast lookup
    _cache_pdf_to_files_dir(file_path, result["doc_id"])

    message = "File already exists, reusing existing document" if result["is_duplicate"] else "File uploaded successfully"
    return ApiResponse(success=True, code=200, message=message, data=result)


@router.post("/from-s3", response_model=ApiResponse[dict])
async def upload_from_s3(
    dataset_id: str,
    s3_url: str = Body(..., embed=True),
    metadata: Dict[str, Any] = Body(..., embed=True),
):
    """
    Download a file from S3 and register it in the dataset.

    Identical flow to local upload: download -> compute content hash -> insert record.
    doc_id = content hash, same dedup logic applies.
    S3 existence is verified upfront via a HEAD request (400 if not found).
    """
    if not s3_url.startswith("s3://"):
        raise HTTPException(status_code=400, detail="s3_url must start with 's3://'")

    # Validate metadata structure
    try:
        _validate_metadata(metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    filename = Path(s3_url).name
    if not filename:
        raise HTTPException(status_code=400, detail="Cannot determine filename from S3 URL")

    # Verify the file exists on S3 before downloading (cheap HEAD request)
    try:
        get_s3_object_info(
            s3_url,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_KEY,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"File not found in S3: {s3_url}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to verify S3 file: {str(e)}")

    # Download to a temp location for hash calculation only
    storage = get_storage()
    temp_dir = Path(tempfile.gettempdir()) / f"s3_{dataset_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / filename

    try:
        await download_file_from_s3_async(
            s3_url,
            temp_file,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_KEY,
        )
    except Exception as e:
        temp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to download file from S3: {str(e)}")

    # Create document record using temp file for hash/size, but store s3_url as file_path
    result = None
    try:
        result = _create_document_record(
            dataset_id, str(temp_file), filename, metadata, s3_url=s3_url
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Document record creation failed for {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create document record: {str(e)}")
    finally:
        # Cache PDF to PDF_FILES_DIR before removing temp file
        if result:
            _cache_pdf_to_files_dir(str(temp_file), result["doc_id"])
        # Temp file is no longer needed; worker fetches directly from S3
        temp_file.unlink(missing_ok=True)

    message = (
        "File already exists, reusing existing document"
        if result["is_duplicate"]
        else "File downloaded from S3 and registered successfully"
    )
    return ApiResponse(success=True, code=200, message=message, data=result)


@router.post("/{doc_id}/tasks", response_model=ApiResponse[TaskResponse])
async def create_task(
    dataset_id: str,
    doc_id: str,
    mode: ProcessingMode = ProcessingMode.CLASSIC,
    reader: ReaderType = ReaderType.MINERU
):
    """Create a processing task for an existing document."""
    conn = get_connection()
    cursor = conn.cursor()

    # Check if document exists
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    doc = cursor.fetchone()

    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    # Parse document metadata
    doc_metadata = json.loads(doc["metadata"]) if "metadata" in doc.keys() and doc["metadata"] else None
    
    timestamp = now()
    task_id = generate_id()
    
    # Create Task record
    cursor.execute(
        """
        INSERT INTO tasks (id, dataset_id, doc_id, mode, reader, status, progress, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (task_id, dataset_id, doc_id, mode.value, reader.value, TaskStatus.PENDING, 0, timestamp, timestamp)
    )
    conn.commit()
    
    # Update document status and task_id
    cursor.execute(
        "UPDATE documents SET status = ?, task_id = ?, updated_at = ? WHERE id = ?",
        (DocumentStatus.PROCESSING, task_id, timestamp, doc_id)
    )
    conn.commit()
    conn.close()
    
    return ApiResponse(
        success=True,
        code=200,
        message="Task created successfully",
        data=TaskResponse(
            task_id=str(task_id),
            dataset_id=dataset_id,
            doc_id=doc_id,
            mode=mode.value,
            reader=reader.value,
            status=TaskStatus.PENDING,
            progress=0,
            metadata=doc_metadata,
            created_at=timestamp,
            updated_at=timestamp
        )
    )


@router.get("/{doc_id}/units/export")
async def export_units_excel(dataset_id: str, doc_id: str):
    """Export all units for a document as an Excel file."""
    import io
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        raise HTTPException(status_code=500, detail="openpyxl is required: pip install openpyxl")

    # Verify document exists
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_name FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    row = cursor.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    # Fetch units via internal units router logic
    from .units import _list_units_for_doc
    units = await _list_units_for_doc(dataset_id, doc_id)

    # Build Excel in memory
    COLUMNS = [
        ("unit_id",           "Unit ID",           38),
        ("unit_type",         "Type",              10),
        ("has_views",         "LOD?",               8),
        ("content",           "Content",           80),
        ("embedding_content", "Embedding Content", 80),
    ]
    HEADER_FILL = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    CELL_FILL   = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    WRAP        = Alignment(wrap_text=True, vertical="top")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Units"

    for col_idx, (key, label, width) in enumerate(COLUMNS, 1):
        cell = ws.cell(row=1, column=col_idx, value=label)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = WRAP
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    for row_idx, unit in enumerate(units, 2):
        row_data = {
            "unit_id":           unit.get("unit_id", ""),
            "unit_type":         unit.get("unit_type", ""),
            "has_views":         "YES" if unit.get("has_views") else "",
            "content":           unit.get("content", ""),
            "embedding_content": unit.get("embedding_content", "") or "",
        }
        for col_idx, (key, label, width) in enumerate(COLUMNS, 1):
            cell = ws.cell(row=row_idx, column=col_idx, value=row_data.get(key, ""))
            cell.alignment = WRAP
            cell.fill = CELL_FILL

    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 20

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    filename = f"units_{doc_id[:8]}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{doc_id}/download")
async def download_document(dataset_id: str, doc_id: str):
    """Download the original file for a document.

    Works for both local and S3-backed files.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_name, file_path FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    file_name = row["file_name"]
    file_path = row["file_path"]

    if file_path.startswith("s3://"):
        # Stream from S3
        import io
        from ..utils.s3 import download_file_from_s3
        buf = io.BytesIO()
        try:
            import boto3
            import re
            m = re.match(r"s3://([^/]+)/(.+)", file_path)
            if not m:
                raise HTTPException(status_code=500, detail="Invalid S3 URL")
            bucket, key = m.group(1), m.group(2)
            s3 = boto3.client(
                "s3",
                aws_access_key_id=config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=config.AWS_SECRET_KEY,
            )
            s3.download_fileobj(bucket, key, buf)
            buf.seek(0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download from S3: {e}")
        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{file_name}"'},
        )
    else:
        # Local file
        p = Path(file_path)
        if not p.is_absolute():
            p = (config.RAG_SERVICE_DIR / p).resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        return FileResponse(
            path=str(p),
            filename=file_name,
            media_type="application/octet-stream",
        )


@router.get("/{doc_id}/views", response_model=ApiResponse[list])
def get_document_views(
    dataset_id: str,
    doc_id: str,
    level: Optional[str] = None,
):
    """Get LOD views for a document.

    Args:
        level: Filter by level - low | medium | high | full. Returns all if omitted.

    Response items:
        - LOW/MEDIUM/FULL: {level, content, token_count}
        - HIGH (DocTree):  {level, tree, token_count}
    """
    from zag.storages.vector import QdrantVectorStore
    from zag.embedders import Embedder
    from zag.schemas import LODLevel

    # Validate level param
    valid_levels = {lv.value for lv in LODLevel}
    if level and level not in valid_levels:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid level '{level}'. Must be one of: {sorted(valid_levels)}"
        )

    # Resolve collection_name
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM datasets WHERE id = ?", (dataset_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    else:
        collection_name = row["name"]

    # Fetch LOD unit via QdrantVectorStore.fetch
    try:
        embedder = Embedder(config.EMBEDDING_URI, api_key=config.OPENAI_API_KEY)
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            grpc_port=config.VECTOR_STORE_GRPC_PORT,
            prefer_grpc=True,
            collection_name=collection_name,
            embedder=embedder,
        )
        units = vector_store.fetch({
            "doc_id": doc_id,
            "metadata.custom.mode": "lod",
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch views: {str(e)}")

    if not units:
        raise HTTPException(status_code=404, detail="No LOD views found for this document")

    # Use the first (and only) LOD unit
    lod_unit = units[0]
    if not lod_unit.views:
        raise HTTPException(status_code=404, detail="LOD unit has no views")

    # Build response items, optionally filtered by level
    result = []
    tags = _extract_tags(lod_unit)
    for view in lod_unit.views:
        if level and view.level != level:
            continue

        if view.level == LODLevel.HIGH:
            result.append({
                "level": view.level if isinstance(view.level, str) else view.level.value,
                "tree": view.content,
                "token_count": view.token_count,
                "tags": tags,
            })
        else:
            result.append({
                "level": view.level if isinstance(view.level, str) else view.level.value,
                "content": view.content,
                "token_count": view.token_count,
                "tags": tags,
            })

    return ApiResponse(success=True, code=200, data=result)


@router.get("/{doc_id}/tasks", response_model=ApiResponse[List[TaskResponse]])
def list_document_tasks(dataset_id: str, doc_id: str):
    """Get all tasks for a specific document."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if document exists
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Query tasks and JOIN documents to get metadata
    cursor.execute(
        """
        SELECT t.*, d.metadata as doc_metadata
        FROM tasks t
        LEFT JOIN documents d ON t.doc_id = d.id
        WHERE t.dataset_id = ? AND t.doc_id = ?
        ORDER BY t.created_at DESC
        """,
        (dataset_id, doc_id)
    )
    rows = cursor.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        error_message = json.loads(row["error_message"]) if row["error_message"] else None
        doc_metadata = json.loads(row["doc_metadata"]) if "doc_metadata" in row.keys() and row["doc_metadata"] else None
        results.append(TaskResponse(
            task_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            doc_id=str(row["doc_id"]),
            mode=row["mode"] if "mode" in row.keys() else "classic",
            reader=row["reader"] if "reader" in row.keys() else "mineru",
            status=row["status"],
            progress=row["progress"],
            metadata=doc_metadata,
            error_message=error_message,
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("", response_model=ApiResponse[List[DocumentResponse]])
def list_documents(
    dataset_id: str,
    status: Optional[str] = None,
    limit: Optional[int] = None,
):
    """List documents in a dataset."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if dataset exists
    cursor.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Query documents
    if status:
        cursor.execute(
            "SELECT * FROM documents WHERE dataset_id = ? AND status = ? ORDER BY created_at DESC",
            (dataset_id, status)
        )
    else:
        cursor.execute(
            "SELECT * FROM documents WHERE dataset_id = ? ORDER BY created_at DESC",
            (dataset_id,)
        )
    
    rows = cursor.fetchall()
    if limit is not None:
        rows = rows[:limit]
    conn.close()

    # Pre-compute base_dir_str once for all file_url builds (pure string, no I/O per row)
    base_dir_str = Path(get_storage().base_dir).resolve().as_posix()

    results = []
    for row in rows:
        doc_metadata = json.loads(row["metadata"]) if "metadata" in row.keys() and row["metadata"] else None
        results.append(DocumentResponse(
            doc_id=str(row["id"]),
            dataset_id=str(row["dataset_id"]),
            file_name=row["file_name"],
            file_path=row["file_path"],
            file_url=_build_file_url(row["file_path"], base_dir_str),
            workspace_dir=row["workspace_dir"],
            file_size=row["file_size"],
            file_type=row["file_type"],
            file_hash=row["file_hash"] if "file_hash" in row.keys() else None,
            metadata=doc_metadata,
            status=row["status"],
            task_id=str(row["task_id"]) if row["task_id"] else None,
            unit_count=row["unit_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        ))
    
    return ApiResponse(success=True, code=200, data=results)


@router.get("/{doc_id}", response_model=ApiResponse[DocumentResponse])
def get_document(dataset_id: str, doc_id: str):
    """Get document by ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Parse document metadata
    doc_metadata = json.loads(row["metadata"]) if "metadata" in row.keys() and row["metadata"] else None
    
    # Build file_url for the worker to fetch the file
    file_url = _build_file_url(row["file_path"])
    
    data = DocumentResponse(
        doc_id=str(row["id"]),
        dataset_id=str(row["dataset_id"]),
        file_name=row["file_name"],
        file_path=row["file_path"],
        file_url=file_url,
        workspace_dir=row["workspace_dir"],
        file_size=row["file_size"],
        file_type=row["file_type"],
        file_hash=row["file_hash"] if "file_hash" in row.keys() else None,
        metadata=doc_metadata,
        status=row["status"],
        task_id=str(row["task_id"]) if row["task_id"] else None,
        unit_count=row["unit_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"]
    )
    
    return ApiResponse(success=True, code=200, data=data)


@router.patch("/{doc_id}", response_model=ApiResponse[DocumentResponse])
async def update_document(
    dataset_id: str,
    doc_id: str,
    metadata: Dict[str, Any] = Body(..., embed=True)
):
    """Update document metadata."""
    # Validate metadata structure
    try:
        _validate_metadata(metadata)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    conn = get_connection()
    cursor = conn.cursor()

    # Check if document exists
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    # Update metadata
    timestamp = now()
    metadata_json = json.dumps(metadata)

    cursor.execute(
        "UPDATE documents SET metadata = ?, updated_at = ? WHERE id = ?",
        (metadata_json, timestamp, doc_id)
    )
    conn.commit()

    # Fetch updated record
    cursor.execute(
        "SELECT * FROM documents WHERE id = ?",
        (doc_id,)
    )
    updated = cursor.fetchone()
    conn.close()

    doc_metadata = json.loads(updated["metadata"]) if updated["metadata"] else None

    return ApiResponse(
        success=True,
        code=200,
        message="Document updated successfully",
        data=DocumentResponse(
            doc_id=str(updated["id"]),
            dataset_id=str(updated["dataset_id"]),
            file_name=updated["file_name"],
            file_path=updated["file_path"],
            workspace_dir=updated["workspace_dir"],
            file_size=updated["file_size"],
            file_type=updated["file_type"],
            file_hash=updated["file_hash"],
            metadata=doc_metadata,
            status=updated["status"],
            task_id=str(updated["task_id"]) if updated["task_id"] else None,
            unit_count=updated["unit_count"],
            created_at=updated["created_at"],
            updated_at=updated["updated_at"]
        )
    )


@router.delete("/{doc_id}", response_model=ApiResponse[MessageResponse])
def delete_document(dataset_id: str, doc_id: str):
    """Delete document and cleanup vector store.
    
    Returns 409 Conflict if the document has active dependencies.
    Callers should resolve dependencies before retrying.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if document exists and get dataset info
    cursor.execute(
        "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    doc = cursor.fetchone()
    if not doc:
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check for active dependencies (documents that depend on this one)
    cursor.execute(
        "SELECT rule, target_doc_id FROM dependencies WHERE target_doc_id = ? AND dataset_id = ?",
        (doc_id, dataset_id)
    )
    dependencies = cursor.fetchall()
    if dependencies:
        conn.close()
        dep_list = [{"rule": d["rule"], "target_doc_id": d["target_doc_id"]} for d in dependencies]
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Document has active dependencies",
                "dependencies": dep_list
            }
        )
    
    # Get dataset name as collection_name
    cursor.execute("SELECT name FROM datasets WHERE id = ?", (dataset_id,))
    dataset = cursor.fetchone()
    if not dataset:
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    collection_name = dataset["name"]
    
    # Delete from database
    # Clean up dependencies where this doc is the source (doc:<doc_id>)
    cursor.execute(
        "DELETE FROM dependencies WHERE rule = ? AND dataset_id = ?",
        (str(Rule.build(DependencySource.DOC, doc_id)), dataset_id)
    )
    cursor.execute("DELETE FROM tasks WHERE doc_id = ?", (doc_id,))
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()
    
    # Cleanup vector store
    try:
        from zag.storages.vector import QdrantVectorStore
        from zag.embedders import Embedder
        
        embedder = Embedder(
            config.EMBEDDING_URI,
            api_key=config.OPENAI_API_KEY
        )
        vector_store = QdrantVectorStore.server(
            host=config.VECTOR_STORE_HOST,
            port=config.VECTOR_STORE_PORT,
            grpc_port=config.VECTOR_STORE_GRPC_PORT,
            prefer_grpc=True,
            collection_name=collection_name,
            embedder=embedder
        )
        vector_store.remove({"doc_id": doc_id})
    except Exception as e:
        # Log but don't fail the API call
        print(f"Warning: Failed to cleanup vector store for doc_id {doc_id}: {e}")

    # Cleanup fulltext store (Meilisearch)
    try:
        from zag.indexers import FullTextIndexer

        fulltext_indexer = FullTextIndexer(
            url=config.MEILISEARCH_HOST,
            index_name=collection_name,
            api_key=config.MEILISEARCH_API_KEY,
            auto_create_index=False,
        )
        deleted = fulltext_indexer.delete_by_doc_id(doc_id)
        if deleted:
            print(f"Cleaned up {deleted} units from Meilisearch for doc_id {doc_id}")
    except Exception as e:
        print(f"Warning: Failed to cleanup Meilisearch for doc_id {doc_id}: {e}")
    
    return ApiResponse(
        success=True,
        code=200,
        message="Document deleted successfully",
        data=MessageResponse(message="Document deleted successfully")
    )


@router.post("/{doc_id}/cache", response_model=ApiResponse[CacheUploadResponse])
async def upload_document_cache(
    dataset_id: str,
    doc_id: str,
    file: UploadFile = File(...),
):
    """
    Upload PDF parsing cache from worker.
    
    Worker calls this after processing a PDF to upload the cache directory.
    The cache is extracted to ARCHIVES_DIR/{doc_id}/ for later use by /locate API.
    
    Args:
        file: tar.gz archive of the cache directory
        
    Returns:
        Cache upload confirmation with path and size
    """
    import tarfile
    import shutil
    
    # Validate file type
    if not file.filename.endswith('.tar.gz'):
        raise HTTPException(status_code=400, detail="File must be a .tar.gz archive")
    
    # Prepare cache directory
    cache_dir = Path(config.ARCHIVES_DIR) / doc_id
    
    # Remove existing cache if present
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    
    # Save uploaded file to temp location
    temp_file = Path(tempfile.gettempdir()) / f"cache_{doc_id}.tar.gz"
    try:
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract archive
        cache_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(temp_file, "r:gz") as tar:
            tar.extractall(path=Path(config.ARCHIVES_DIR))
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        
        return ApiResponse(
            success=True,
            code=200,
            message="Cache uploaded successfully",
            data=CacheUploadResponse(
                doc_id=doc_id,
                cache_path=str(cache_dir),
                size_bytes=total_size,
                message=f"Cache extracted to {cache_dir}"
            )
        )
    except Exception as e:
        # Cleanup on error
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        raise HTTPException(status_code=500, detail=f"Failed to process cache: {str(e)}")
    finally:
        # Cleanup temp file (best-effort: on Windows the OS may still hold the lock)
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass  # Windows: file lock not yet released, OS will clean up temp dir


@router.post("/locate", response_model=ApiResponse[LocatePageResponse])
async def locate_pages(
    dataset_id: str,
    request: LocatePageRequest = Body(...),
):
    """
    Locate page numbers for text snippets.
    
    Batch API that accepts multiple items, each with doc_id + text_start (+ optional text_end).
    Returns page numbers for each item.
    
    Performance:
    - Concurrent PDF loading for different docs
    - Concurrent text search within each PDF
    """
    import asyncio
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor
    
    from zag.schemas.document import PageableDocument
    from zag.utils.page_inference import fuzzy_find_start
    
    # Cache directory for PDF parsing results
    cache_dir = Path(config.ARCHIVES_DIR)
    
    # Group items by doc_id for efficient processing
    doc_groups: dict[str, list] = defaultdict(list)
    for item in request.items:
        doc_groups[item.doc_id].append(item)
    
    def find_pages_for_text(doc: PageableDocument, text_start: str, text_end: str = None) -> tuple[list[int] | None, bool]:
        """
        Find page numbers for a text snippet.
        
        Returns:
            (page_numbers, found)
        """
        # Find start position
        # Search entire document (no range limit for correctness)
        start_pos = fuzzy_find_start(text_start, doc.content, start_from=0, threshold=0.80)
        
        if start_pos is None:
            return None, False
        
        # Determine end position
        if text_end:
            end_pos = fuzzy_find_start(text_end, doc.content, start_from=start_pos + len(text_start), threshold=0.80, max_search_range=100000)
            if end_pos is None:
                # Fallback: estimate length from text_start
                end_pos = start_pos + len(text_start)
            else:
                end_pos = end_pos + len(text_end)
        else:
            end_pos = start_pos + len(text_start)
        
        # Build page positions
        page_positions = []
        current_pos = 0
        
        for page in doc.pages:
            page_content = page.content or ""
            if not page_content.strip():
                page_positions.append((current_pos, current_pos, page.page_number))
                continue
            
            # Find page in full content
            page_sig = page_content[:50] if len(page_content) > 50 else page_content
            pos = doc.content.find(page_sig, current_pos)
            
            if pos != -1:
                page_positions.append((pos, pos + len(page_content), page.page_number))
                current_pos = pos + 1
            else:
                page_positions.append((current_pos, current_pos + len(page_content), page.page_number))
                current_pos += len(page_content) + 1
        
        # Find overlapping pages
        overlapping = []
        for page_start, page_end, page_num in page_positions:
            if not (end_pos <= page_start or start_pos >= page_end):
                overlapping.append(page_num)
        
        return sorted(overlapping) if overlapping else None, True
    
    def find_pages_in_document(file_path: str, text_start: str, text_end: str = None) -> tuple[list[int] | None, bool]:
        """
        Search directly in the document file for accurate page numbers.
        Only PDF files are searched via fitz; all other formats fall back to the
        archive cache (handled by the caller).
        """
        ext = Path(file_path).suffix.lower()
        if ext != '.pdf':
            # Non-PDF (DOCX, DOC, MD, TXT, etc.): fitz cannot open these formats.
            # Page location still works for these files via the archive-cache path:
            # DoclingReader / MarkItDownReader writes a PageableDocument with real
            # page structure (or a synthetic single page) into ARCHIVES_DIR when
            # the document is first processed.  The caller will load that cache
            # and call find_pages_for_text(), which works on any PageableDocument.
            return None, False

        import fitz
        from zag.utils.page_inference import normalize_text

        norm_start = normalize_text(text_start)
        norm_end = normalize_text(text_end) if text_end else None

        full_text = ""
        page_positions = []
        current = 0
        start_pos = None
        end_pos = None
        search_from = 0  # incremental search, avoid re-scanning old pages

        doc = fitz.open(file_path)
        try:
            for page in doc:
                page_num = page.number + 1
                norm = normalize_text(page.get_text())

                page_start = current
                page_end = current + len(norm)
                page_positions.append((page_start, page_end, page_num))
                full_text += ("\n" if full_text else "") + norm
                current = page_end + 1

                # Early scanned-PDF detection after 3 pages
                if page_num == 3 and len(full_text) / 3 < 50:
                    return None, False  # fallback to MinerU cache

                # Search only the newly added portion
                if start_pos is None:
                    found = fuzzy_find_start(norm_start, full_text, start_from=search_from, threshold=0.85)
                    if found is not None:
                        start_pos = found
                    else:
                        search_from = max(0, len(full_text) - len(norm_start) - 10)

                if start_pos is not None:
                    if norm_end:
                        found_end = fuzzy_find_start(
                            norm_end, full_text,
                            start_from=start_pos + len(norm_start),
                            threshold=0.85,
                            max_search_range=100000,
                        )
                        if found_end is not None:
                            end_pos = found_end + len(norm_end)
                            break
                    else:
                        end_pos = start_pos + len(norm_start)
                        break
        finally:
            doc.close()

        if start_pos is None:
            return None, False

        if end_pos is None:
            end_pos = start_pos + len(norm_start)

        overlapping = [
            pn for ps, pe, pn in page_positions
            if not (end_pos <= ps or start_pos >= pe)
        ]
        return sorted(overlapping) if overlapping else None, True

    def get_file_path_for_doc(doc_id: str) -> str | None:
        """
        Return a local path to the original document for the given doc_id.

        Lookup order:
        1. PDF_FILES_DIR/{doc_id}*.<known-ext>  – pick the newest by mtime
        2. DB file_path: local path (absolute or relative to rag-service dir)
        3. S3 download → saved to PDF_FILES_DIR/{doc_id}{original_ext} for next time

        Never raises – returns None on any failure so callers can fallback safely.
        """
        _DOC_EXTENSIONS = {
            '.pdf', '.docx', '.doc', '.md', '.txt', '.markdown',
            '.pptx', '.ppt', '.xlsx', '.xls',
        }
        try:
            from ..utils.s3 import download_file_from_s3

            pdf_files_dir = Path(config.PDF_FILES_DIR)

            # 1. Find all local variants matching {doc_id}*, pick newest by mtime
            if pdf_files_dir.exists():
                candidates = sorted(
                    [
                        p for p in pdf_files_dir.glob(f"{doc_id}*")
                        if p.suffix.lower() in _DOC_EXTENSIONS
                    ],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if candidates:
                    return str(candidates[0])

            # 2. Get file_path from DB
            conn = get_connection()
            try:
                cursor = conn.execute("SELECT file_path FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                if not row or not row["file_path"]:
                    return None
                file_path = row["file_path"]
            finally:
                conn.close()

            if file_path.startswith("s3://"):
                # 3. Download from S3, preserve original extension from S3 path
                pdf_files_dir.mkdir(parents=True, exist_ok=True)
                s3_ext = Path(file_path).suffix or '.pdf'
                local_file = pdf_files_dir / f"{doc_id}{s3_ext}"
                ok = download_file_from_s3(file_path, local_file)
                return str(local_file) if ok else None
            else:
                # Local path: try as-is first, then relative to rag-service dir
                p = Path(file_path)
                if p.is_absolute():
                    return str(p) if p.exists() else None
                # Relative path – resolve against the rag-service directory
                resolved = (config.RAG_SERVICE_DIR / p).resolve()
                return str(resolved) if resolved.exists() else None
        except Exception:
            return None

    def process_single_doc(doc_id: str, items: list) -> list[LocatePageResult]:
        """Process all items for a single document. Never raises."""
        try:
            results = []

            # Try to get the actual document file for accurate page lookup
            file_path = get_file_path_for_doc(doc_id)  # always returns None on failure

            # Preload archive cache only if needed
            doc_cache_dir = cache_dir / doc_id
            doc = None

            # Process all items for this doc
            for item in items:
                try:
                    page_numbers, found = None, False

                    # Primary: direct file search (PDF only; non-PDF returns False)
                    if file_path:
                        page_numbers, found = find_pages_in_document(
                            file_path, item.text_start, item.text_end
                        )

                    # Fallback: archive cache
                    if not found and doc_cache_dir.exists():
                        if doc is None:
                            try:
                                doc = PageableDocument.load(doc_cache_dir)
                            except Exception:
                                doc = None
                        if doc is not None:
                            page_numbers, found = find_pages_for_text(
                                doc, item.text_start, item.text_end
                            )

                    # Last resort: return page 1 rather than leaving caller empty
                    if not found:
                        page_numbers = [1]

                    results.append(LocatePageResult(
                        request_id=item.request_id,
                        doc_id=doc_id,
                        page_numbers=page_numbers,
                        found=found,
                        error=None if found else "Text not found; defaulting to page 1",
                    ))
                except Exception as e:
                    # Item-level failure → page 1, never propagate
                    results.append(LocatePageResult(
                        request_id=item.request_id,
                        doc_id=doc_id,
                        page_numbers=[1],
                        found=False,
                        error=f"Search error: {str(e)}",
                    ))

            return results
        except Exception as e:
            # Doc-level failure → all items get page 1
            return [
                LocatePageResult(
                    request_id=item.request_id,
                    doc_id=doc_id,
                    page_numbers=[1],
                    found=False,
                    error=f"Doc processing error: {str(e)}",
                )
                for item in items
            ]
    
    # Process all docs concurrently
    loop = asyncio.get_event_loop()
    
    with ThreadPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, process_single_doc, doc_id, items)
            for doc_id, items in doc_groups.items()
        ]
        all_results = await asyncio.gather(*futures)
    
    # Flatten results
    results = []
    for doc_results in all_results:
        results.extend(doc_results)
    
    return ApiResponse(
        success=True,
        code=200,
        data=LocatePageResponse(results=results)
    )
