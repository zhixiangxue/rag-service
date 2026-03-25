"""Dependency rules database access."""
from __future__ import annotations

from typing import Optional


class DependencyRepository:
    """All SQL access for the dependencies table.

    Constructor injection: caller owns the connection lifecycle.
    Every method returns plain dicts, never sqlite3.Row objects.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    @staticmethod
    def _row(row) -> Optional[dict]:
        return dict(row) if row else None

    # ── reads ────────────────────────────────────────────────────────────

    def get(self, dep_id: str, dataset_id: Optional[str] = None) -> Optional[dict]:
        """Return dependency row as dict, or None.

        If dataset_id is given, the query also filters by dataset_id.
        """
        if dataset_id is not None:
            cur = self._conn.execute(
                "SELECT id FROM dependencies WHERE id = ? AND dataset_id = ?",
                (dep_id, dataset_id),
            )
        else:
            cur = self._conn.execute(
                "SELECT id FROM dependencies WHERE id = ?", (dep_id,)
            )
        return self._row(cur.fetchone())

    def list_by_dataset(self, dataset_id: str) -> list[dict]:
        """Return all dependency rows for a dataset with joined document metadata."""
        cur = self._conn.execute(
            """
            SELECT d.id, d.dataset_id, d.rule, d.target_doc_id,
                   doc.file_name AS target_file_name,
                   doc.file_path AS target_file_path,
                   src_doc.file_path AS source_file_path,
                   d.created_at, d.updated_at
            FROM dependencies d
            LEFT JOIN documents doc ON doc.id = d.target_doc_id
            LEFT JOIN documents src_doc
                ON d.rule LIKE 'doc:%' AND src_doc.id = SUBSTR(d.rule, 5)
            WHERE d.dataset_id = ?
            ORDER BY d.created_at DESC
            """,
            (dataset_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_rules(self, dataset_id: str) -> list[dict]:
        """Return all rule + target_doc_id rows for a dataset (used in BFS resolve)."""
        cur = self._conn.execute(
            "SELECT rule, target_doc_id FROM dependencies WHERE dataset_id = ?",
            (dataset_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def list_by_target(self, target_doc_id: str, dataset_id: str) -> list[dict]:
        """Return dependency rows where the given doc is the target."""
        cur = self._conn.execute(
            "SELECT rule, target_doc_id FROM dependencies"
            " WHERE target_doc_id = ? AND dataset_id = ?",
            (target_doc_id, dataset_id),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_by_rule_target(
        self, dataset_id: str, rule: str, target_doc_id: str
    ) -> Optional[dict]:
        """Return existing dependency matching rule + target_doc_id, or None."""
        cur = self._conn.execute(
            "SELECT id FROM dependencies"
            " WHERE dataset_id = ? AND rule = ? AND target_doc_id = ?",
            (dataset_id, rule, target_doc_id),
        )
        return self._row(cur.fetchone())

    # ── writes ───────────────────────────────────────────────────────────

    def create(
        self,
        id: str,
        dataset_id: str,
        rule: str,
        target_doc_id: str,
        timestamp: str,
    ) -> None:
        """Insert a new dependency rule."""
        self._conn.execute(
            """
            INSERT INTO dependencies (id, dataset_id, rule, target_doc_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (id, dataset_id, rule, target_doc_id, timestamp, timestamp),
        )
        self._conn.commit()

    def delete(self, dep_id: str) -> None:
        """Delete a dependency by ID."""
        self._conn.execute("DELETE FROM dependencies WHERE id = ?", (dep_id,))
        self._conn.commit()

    def delete_by_rule(self, rule: str, dataset_id: str) -> None:
        """Delete all dependencies with the given rule in a dataset."""
        self._conn.execute(
            "DELETE FROM dependencies WHERE rule = ? AND dataset_id = ?",
            (rule, dataset_id),
        )
        self._conn.commit()
