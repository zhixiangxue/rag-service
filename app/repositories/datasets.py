"""Dataset database access."""
from __future__ import annotations

from typing import Optional


class DatasetRepository:
    """All SQL access for the datasets table.

    Constructor injection: caller owns the connection lifecycle.
    Every method returns plain dicts, never sqlite3.Row objects.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    @staticmethod
    def _row(row) -> Optional[dict]:
        return dict(row) if row else None

    # ── reads ────────────────────────────────────────────────────────────

    def get(self, dataset_id: str) -> Optional[dict]:
        """Return dataset row as dict, or None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        )
        return self._row(cur.fetchone())

    def get_by_name(self, name: str) -> Optional[dict]:
        """Return dataset row by name, or None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM datasets WHERE name = ?", (name,)
        )
        return self._row(cur.fetchone())

    def exists(self, dataset_id: str) -> bool:
        """Return True if dataset exists."""
        cur = self._conn.execute(
            "SELECT id FROM datasets WHERE id = ?", (dataset_id,)
        )
        return cur.fetchone() is not None

    def list_all(self) -> list[dict]:
        """Return all datasets ordered by created_at DESC."""
        cur = self._conn.execute(
            "SELECT * FROM datasets ORDER BY created_at DESC"
        )
        return [dict(row) for row in cur.fetchall()]

    # ── writes ───────────────────────────────────────────────────────────

    def create(
        self,
        id: str,
        name: str,
        description: Optional[str],
        engine: str,
        config_json: Optional[str],
        timestamp: str,
    ) -> None:
        """Insert a new dataset row."""
        self._conn.execute(
            """
            INSERT INTO datasets (id, name, description, engine, config, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (id, name, description, engine, config_json, timestamp, timestamp),
        )
        self._conn.commit()

    def update(self, dataset_id: str, fields: dict, updated_at: str) -> None:
        """Update arbitrary fields. `fields` must be non-empty.

        Example:
            repo.update(dataset_id, {"name": "new-name", "engine": "qdrant"}, ts)
        """
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [updated_at, dataset_id]
        self._conn.execute(
            f"UPDATE datasets SET {set_clause}, updated_at = ? WHERE id = ?",
            params,
        )
        self._conn.commit()

    def delete(self, dataset_id: str) -> None:
        """Delete a dataset by ID."""
        self._conn.execute(
            "DELETE FROM datasets WHERE id = ?", (dataset_id,)
        )
        self._conn.commit()
