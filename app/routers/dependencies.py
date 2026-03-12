"""Document dependency rules API endpoints.

A dependency rule says: whenever doc X is selected, also bring in doc Y.

Rule field prefix conventions:
  lender:<name>   - applies to all docs from this lender
  overlay:<value> - applies to docs whose metadata.overlays contains this value
  doc:<doc_id>    - applies to this specific document"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import re

from ..database import get_connection, now, generate_id
from ..domain.deps import Rule
from ..schemas import ApiResponse, MessageResponse

router = APIRouter(prefix="/datasets", tags=["dependencies"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class DependencyCreate(BaseModel):
    rule: str
    target_doc_id: str


class DependencyResponse(BaseModel):
    id: str
    dataset_id: str
    rule: str
    target_doc_id: str
    target_file_name: Optional[str] = None
    created_at: str
    updated_at: str


class BatchDependenciesRequest(BaseModel):
    doc_ids: List[str]


class DocDependencyBreakdown(BaseModel):
    """Dependency doc_ids grouped by the rule protocol that triggered them."""
    lender: List[str] = []
    overlay: List[str] = []
    doc: List[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_exists(cursor, dataset_id: str) -> bool:
    cursor.execute("SELECT id FROM datasets WHERE id = ?", (dataset_id,))
    return cursor.fetchone() is not None


def _normalize(s: str) -> str:
    """Normalize a name for loose matching: lowercase + strip all non-alphanumeric chars.

    Examples:
      "FannieMae" -> "fanniemae"
      "Fannie Mae" -> "fanniemae"
      "fannie_mae" -> "fanniemae"
      "Fannie-Mae" -> "fanniemae"
      "General Mortgage Capital Corporation" -> "generalmortgagecapitalcorporation"
    """
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _source_key(prefix: str, value: str) -> str:
    """Build a normalized rules_map key for lender/overlay sources."""
    return f"{prefix}:{_normalize(value)}"


def _load_rules_and_meta(cursor, dataset_id: str, seed_doc_ids: List[str]) -> tuple:
    """Load all rules and pre-fetch metadata for seed docs + all rule targets.

    Returns (rules_map, meta_map):
      - rules_map: {source_str: set(target_doc_id)}
      - meta_map:  {doc_id: metadata_dict}

    Pre-loading all target doc metadata means the BFS runs entirely in memory
    with no additional SQL queries per hop.
    """
    # Load all dependency rules for the dataset
    cursor.execute(
        "SELECT rule, target_doc_id FROM dependencies WHERE dataset_id = ?",
        (dataset_id,)
    )
    rules_map: dict = {}
    all_target_ids: set = set()
    for row in cursor.fetchall():
        rule_str: str = row["rule"]
        # Normalize lender:/overlay: value; doc: IDs are kept as-is
        if rule_str.startswith("lender:"):
            key = _source_key("lender", rule_str[len("lender:"):])
        elif rule_str.startswith("overlay:"):
            key = _source_key("overlay", rule_str[len("overlay:"):])
        else:
            key = rule_str  # doc:<id> or unknown prefix, no normalization
        rules_map.setdefault(key, set()).add(row["target_doc_id"])
        all_target_ids.add(row["target_doc_id"])

    # Batch-fetch metadata for seed docs + all possible targets
    all_ids = list(set(seed_doc_ids) | all_target_ids)
    placeholders = ",".join("?" * len(all_ids))
    cursor.execute(
        f"SELECT id, metadata FROM documents WHERE id IN ({placeholders}) AND dataset_id = ?",
        (*all_ids, dataset_id)
    )
    meta_map: dict = {
        row["id"]: (json.loads(row["metadata"]) if row["metadata"] else {})
        for row in cursor.fetchall()
    }
    return rules_map, meta_map


def _resolve_transitive(seed_doc_ids: List[str], rules_map: dict, meta_map: dict) -> DocDependencyBreakdown:
    """BFS transitive dependency resolution using pre-loaded data (no SQL calls).

    Each dep is attributed to the rule source type that directly triggered it:
      - lender: matched via lender:<name> rule
      - overlay: matched via overlay:<value> rule
      - doc: matched via doc:<id> rule
    """
    input_set = set(seed_doc_ids)
    lender_deps: set = set()
    overlay_deps: set = set()
    doc_deps: set = set()

    visited: set = set()
    queue: list = list(seed_doc_ids)

    while queue:
        current_id = queue.pop(0)
        if current_id in visited:
            continue
        visited.add(current_id)

        meta = meta_map.get(current_id, {})
        lender = meta.get("lender", "")
        overlays = meta.get("overlays") or []

        # lender-level rule
        for target in rules_map.get(_source_key("lender", lender), set()):
            if target not in input_set and target not in lender_deps:
                lender_deps.add(target)
                queue.append(target)

        # overlay-level rules
        for o in overlays:
            for target in rules_map.get(_source_key("overlay", o), set()):
                if target not in input_set and target not in overlay_deps:
                    overlay_deps.add(target)
                    queue.append(target)

        # doc-specific rule (doc IDs are not normalized)
        for target in rules_map.get(f"doc:{current_id}", set()):
            if target not in input_set and target not in doc_deps:
                doc_deps.add(target)
                queue.append(target)

    return DocDependencyBreakdown(
        lender=sorted(lender_deps),
        overlay=sorted(overlay_deps),
        doc=sorted(doc_deps),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/dependencies", response_model=ApiResponse[List[DependencyResponse]])
def list_dependencies(dataset_id: str):
    """List all dependency rules for a dataset."""
    conn = get_connection()
    cursor = conn.cursor()

    if not _dataset_exists(cursor, dataset_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    cursor.execute(
        """
        SELECT d.id, d.dataset_id, d.rule, d.target_doc_id,
               doc.file_name AS target_file_name,
               d.created_at, d.updated_at
        FROM dependencies d
        LEFT JOIN documents doc ON doc.id = d.target_doc_id
        WHERE d.dataset_id = ?
        ORDER BY d.created_at DESC
        """,
        (dataset_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    items = [
        DependencyResponse(
            id=row["id"],
            dataset_id=row["dataset_id"],
            rule=row["rule"],
            target_doc_id=row["target_doc_id"],
            target_file_name=row["target_file_name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
        for row in rows
    ]
    return ApiResponse(success=True, code=200, data=items)


@router.post("/{dataset_id}/dependencies", response_model=ApiResponse[DependencyResponse])
def create_dependency(dataset_id: str, body: DependencyCreate):
    """Create a new dependency rule.

    Validates that target_doc_id exists in the documents table.
    Validates source format (must be one of: doc:, lender:, overlay:).
    """
    conn = get_connection()
    cursor = conn.cursor()

    if not _dataset_exists(cursor, dataset_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate rule format
    try:
        Rule.parse(body.rule)
    except ValueError as e:
        conn.close()
        raise HTTPException(status_code=400, detail=str(e))

    # Validate target document exists
    cursor.execute("SELECT id, file_name FROM documents WHERE id = ?", (body.target_doc_id,))
    target_doc = cursor.fetchone()
    if not target_doc:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail=f"Target document '{body.target_doc_id}' not found"
        )

    timestamp = now()
    dep_id = generate_id()

    cursor.execute(
        """
        INSERT INTO dependencies (id, dataset_id, rule, target_doc_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (dep_id, dataset_id, body.rule, body.target_doc_id, timestamp, timestamp)
    )
    conn.commit()
    conn.close()

    return ApiResponse(
        success=True,
        code=200,
        message="Dependency created successfully",
        data=DependencyResponse(
            id=dep_id,
            dataset_id=dataset_id,
            rule=body.rule,
            target_doc_id=body.target_doc_id,
            target_file_name=target_doc["file_name"],
            created_at=timestamp,
            updated_at=timestamp,
        )
    )


@router.delete("/{dataset_id}/dependencies/{dep_id}", response_model=ApiResponse[MessageResponse])
def delete_dependency(dataset_id: str, dep_id: str):
    """Delete a dependency rule by id."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT id FROM dependencies WHERE id = ? AND dataset_id = ?",
        (dep_id, dataset_id)
    )
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Dependency not found")

    cursor.execute("DELETE FROM dependencies WHERE id = ?", (dep_id,))
    conn.commit()
    conn.close()

    return ApiResponse(
        success=True,
        code=200,
        message="Dependency deleted successfully",
        data=MessageResponse(message="Dependency deleted successfully")
    )


@router.get("/{dataset_id}/{doc_id}/dependencies", response_model=ApiResponse[DocDependencyBreakdown])
def get_doc_dependencies(dataset_id: str, doc_id: str):
    """Return transitive dependency doc_ids for a document, grouped by rule protocol."""
    conn = get_connection()
    cursor = conn.cursor()

    if not _dataset_exists(cursor, dataset_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify doc exists
    cursor.execute("SELECT id FROM documents WHERE id = ? AND dataset_id = ?", (doc_id, dataset_id))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Document not found")

    rules_map, meta_map = _load_rules_and_meta(cursor, dataset_id, [doc_id])
    conn.close()

    breakdown = _resolve_transitive([doc_id], rules_map, meta_map)
    return ApiResponse(success=True, code=200, data=breakdown)


@router.post("/{dataset_id}/dependencies/resolve", response_model=ApiResponse[Dict[str, DocDependencyBreakdown]])
def resolve_dependencies(dataset_id: str, body: BatchDependenciesRequest):
    """Return transitive dependency breakdown for each doc_id in the batch.

    Response: {doc_id: {lender: [...], overlay: [...], doc: [...]}, ...}
    """
    if not body.doc_ids:
        return ApiResponse(success=True, code=200, data={})

    conn = get_connection()
    cursor = conn.cursor()

    if not _dataset_exists(cursor, dataset_id):
        conn.close()
        raise HTTPException(status_code=404, detail="Dataset not found")

    rules_map, meta_map = _load_rules_and_meta(cursor, dataset_id, body.doc_ids)
    conn.close()

    result: dict = {
        doc_id: _resolve_transitive([doc_id], rules_map, meta_map)
        for doc_id in body.doc_ids
        if doc_id in meta_map
    }
    return ApiResponse(success=True, code=200, data=result)
