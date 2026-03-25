"""Task database access."""
from __future__ import annotations

from typing import Optional


class TaskRepository:
    """All SQL access for the tasks table.

    Constructor injection: caller owns the connection lifecycle.
    Every method returns plain dicts, never sqlite3.Row objects.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    @staticmethod
    def _row(row) -> Optional[dict]:
        return dict(row) if row else None

    # ── reads ────────────────────────────────────────────────────────────

    def get(self, task_id: str) -> Optional[dict]:
        """Return task row as dict, or None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)
        )
        return self._row(cur.fetchone())

    def list_all(
        self, status: Optional[str] = None, limit: int = 10
    ) -> list[dict]:
        """Return recent tasks, optionally filtered by status."""
        if status:
            cur = self._conn.execute(
                "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            )
        else:
            cur = self._conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in cur.fetchall()]

    def list_by_dataset(self, dataset_id: str) -> list[dict]:
        """Return all tasks for a dataset, newest first."""
        cur = self._conn.execute(
            "SELECT * FROM tasks WHERE dataset_id = ? ORDER BY created_at DESC",
            (dataset_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_by_doc(self, dataset_id: str, doc_id: str) -> list[dict]:
        """Return tasks for a document with doc_metadata joined from documents."""
        cur = self._conn.execute(
            """
            SELECT t.*, d.metadata as doc_metadata
            FROM tasks t
            LEFT JOIN documents d ON t.doc_id = d.id
            WHERE t.dataset_id = ? AND t.doc_id = ?
            ORDER BY t.created_at DESC
            """,
            (dataset_id, doc_id),
        )
        return [dict(row) for row in cur.fetchall()]

    def count_total(self) -> int:
        """Return total number of tasks."""
        cur = self._conn.execute("SELECT COUNT(*) FROM tasks")
        return cur.fetchone()[0]

    def count_by_status(self) -> dict:
        """Return {status: count} mapping for all statuses present in the table."""
        cur = self._conn.execute(
            "SELECT status, COUNT(*) as count FROM tasks GROUP BY status"
        )
        return {row["status"]: row["count"] for row in cur.fetchall()}

    # ── writes ───────────────────────────────────────────────────────────

    def create(
        self,
        id: str,
        dataset_id: str,
        doc_id: str,
        mode: str,
        reader: str,
        status: str,
        progress: int,
        timestamp: str,
    ) -> None:
        """Insert a new task row."""
        self._conn.execute(
            """
            INSERT INTO tasks
                (id, dataset_id, doc_id, mode, reader, status, progress, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (id, dataset_id, doc_id, mode, reader, status, progress, timestamp, timestamp),
        )
        self._conn.commit()

    def update(self, task_id: str, fields: dict, updated_at: str) -> None:
        """Update arbitrary task fields. `fields` must be non-empty.

        Example:
            repo.update(task_id, {"status": "PROCESSING", "progress": 10}, ts)
        """
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [updated_at, task_id]
        self._conn.execute(
            f"UPDATE tasks SET {set_clause}, updated_at = ? WHERE id = ?",
            params,
        )
        self._conn.commit()

    def reset(self, task_id: str, updated_at: str) -> None:
        """Reset task to PENDING, clearing progress and error_message."""
        self._conn.execute(
            "UPDATE tasks SET status = 'PENDING', progress = 0,"
            " error_message = NULL, updated_at = ? WHERE id = ?",
            (updated_at, task_id),
        )
        self._conn.commit()

    def cancel(self, task_id: str, updated_at: str) -> None:
        """Set task status to CANCELLED."""
        self._conn.execute(
            "UPDATE tasks SET status = 'CANCELLED', updated_at = ? WHERE id = ?",
            (updated_at, task_id),
        )
        self._conn.commit()

    def delete_by_doc(self, doc_id: str) -> None:
        """Delete all tasks associated with a document."""
        self._conn.execute(
            "DELETE FROM tasks WHERE doc_id = ?", (doc_id,)
        )
        self._conn.commit()
