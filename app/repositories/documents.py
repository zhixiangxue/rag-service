"""Document database access."""
from __future__ import annotations

import sqlite3
from typing import Optional


class DocumentRepository:
    """All SQL access for the documents table.

    Constructor injection: caller owns the connection lifecycle.
    Every method returns plain dicts, never sqlite3.Row objects.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    @staticmethod
    def _row(row) -> Optional[dict]:
        return dict(row) if row else None

    # ── reads ────────────────────────────────────────────────────────────

    def get(self, doc_id: str, dataset_id: Optional[str] = None) -> Optional[dict]:
        """Return document row as dict, or None.

        If dataset_id is given, the query also filters by dataset_id.
        """
        if dataset_id is not None:
            cur = self._conn.execute(
                "SELECT * FROM documents WHERE id = ? AND dataset_id = ?",
                (doc_id, dataset_id),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM documents WHERE id = ?", (doc_id,)
            )
        return self._row(cur.fetchone())

    def get_by_hash(self, dataset_id: str, file_hash: str) -> Optional[dict]:
        """Return document row matching dataset_id + file_hash, or None."""
        cur = self._conn.execute(
            "SELECT id, file_name, file_path FROM documents"
            " WHERE dataset_id = ? AND file_hash = ?",
            (dataset_id, file_hash),
        )
        return self._row(cur.fetchone())

    def list_by_dataset(
        self, dataset_id: str, status: Optional[str] = None
    ) -> list[dict]:
        """Return documents for a dataset, optionally filtered by status."""
        if status:
            cur = self._conn.execute(
                "SELECT * FROM documents WHERE dataset_id = ? AND status = ?"
                " ORDER BY created_at DESC",
                (dataset_id, status),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM documents WHERE dataset_id = ? ORDER BY created_at DESC",
                (dataset_id,),
            )
        return [dict(row) for row in cur.fetchall()]

    def list_for_catalog(self, dataset_id: str) -> list[dict]:
        """Return id, file_name, metadata for all docs in a dataset (catalog view)."""
        cur = self._conn.execute(
            "SELECT id, file_name, metadata FROM documents WHERE dataset_id = ?",
            (dataset_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_metadata_by_ids(self, ids: list, dataset_id: str) -> list[dict]:
        """Return id + metadata for the given doc IDs within a dataset."""
        placeholders = ",".join("?" * len(ids))
        cur = self._conn.execute(
            f"SELECT id, metadata FROM documents"
            f" WHERE id IN ({placeholders}) AND dataset_id = ?",
            (*ids, dataset_id),
        )
        return [dict(row) for row in cur.fetchall()]

    # ── writes ───────────────────────────────────────────────────────────

    def create(
        self,
        id: str,
        dataset_id: str,
        file_name: str,
        file_path: str,
        workspace_dir: str,
        file_size: int,
        file_type: str,
        file_hash: str,
        metadata_json: str,
        status: str,
        timestamp: str,
    ) -> None:
        """Insert a new document row.

        Raises sqlite3.IntegrityError on duplicate (dataset_id, file_hash) — caller
        is responsible for calling conn.rollback() and handling the race condition.
        """
        self._conn.execute(
            """
            INSERT INTO documents
                (id, dataset_id, file_name, file_path, workspace_dir,
                 file_size, file_type, file_hash, metadata, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                id, dataset_id, file_name, file_path, workspace_dir,
                file_size, file_type, file_hash, metadata_json, status,
                timestamp, timestamp,
            ),
        )
        self._conn.commit()

    def update_file_path(self, doc_id: str, file_path: str, updated_at: str) -> None:
        """Update the stored file_path (used when re-uploading a duplicate)."""
        self._conn.execute(
            "UPDATE documents SET file_path = ?, updated_at = ? WHERE id = ?",
            (file_path, updated_at, doc_id),
        )
        self._conn.commit()

    def update_metadata(
        self, doc_id: str, metadata_json: str, updated_at: str
    ) -> None:
        """Replace the metadata JSON for a document."""
        self._conn.execute(
            "UPDATE documents SET metadata = ?, updated_at = ? WHERE id = ?",
            (metadata_json, updated_at, doc_id),
        )
        self._conn.commit()

    def update_status(self, doc_id: str, status: str, updated_at: str) -> None:
        """Update only the document status."""
        self._conn.execute(
            "UPDATE documents SET status = ?, updated_at = ? WHERE id = ?",
            (status, updated_at, doc_id),
        )
        self._conn.commit()

    def update_task_link(
        self, doc_id: str, status: str, task_id: str, updated_at: str
    ) -> None:
        """Set status + task_id together (called when a processing task is created)."""
        self._conn.execute(
            "UPDATE documents SET status = ?, task_id = ?, updated_at = ? WHERE id = ?",
            (status, task_id, updated_at, doc_id),
        )
        self._conn.commit()

    def update_unit_count(self, doc_id: str, unit_count: int) -> None:
        """Update the cached unit count for a document."""
        self._conn.execute(
            "UPDATE documents SET unit_count = ? WHERE id = ?",
            (unit_count, doc_id),
        )
        self._conn.commit()

    def delete(self, doc_id: str) -> None:
        """Delete a document by ID."""
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()
