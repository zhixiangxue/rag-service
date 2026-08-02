"""Initialize database schema (datasets, documents, tasks, dependencies).

Works with both SQLite and rqlite. Reads DATABASE_URI from .env.

Usage:
    python -m scripts.init_db          # uses DATABASE_URI from .env
"""
import sys
from pathlib import Path

# Ensure project root is importable when run from anywhere
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.database import get_connection

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS datasets (
        id          TEXT PRIMARY KEY,
        name        TEXT UNIQUE NOT NULL,
        description TEXT,
        engine      TEXT DEFAULT 'qdrant',
        config      TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS documents (
        id           TEXT PRIMARY KEY,
        dataset_id   TEXT NOT NULL,
        file_name    TEXT NOT NULL,
        file_path    TEXT NOT NULL,
        workspace_dir TEXT,
        file_size    INTEGER,
        file_type    TEXT,
        file_hash    TEXT,
        metadata     TEXT,
        status       TEXT DEFAULT 'PENDING',
        task_id      TEXT,
        unit_count   INTEGER,
        created_at   TEXT NOT NULL,
        updated_at   TEXT NOT NULL,
        UNIQUE (dataset_id, file_hash),
        FOREIGN KEY (dataset_id) REFERENCES datasets(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id          TEXT PRIMARY KEY,
        dataset_id  TEXT NOT NULL,
        doc_id      TEXT NOT NULL,
        mode        TEXT DEFAULT 'classic',
        reader      TEXT DEFAULT 'mineru',
        status      TEXT DEFAULT 'PENDING',
        progress    INTEGER DEFAULT 0,
        error_message TEXT,
        callback    TEXT,
        created_at  TEXT NOT NULL,
        updated_at  TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id),
        FOREIGN KEY (doc_id) REFERENCES documents(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS dependencies (
        id             TEXT PRIMARY KEY,
        dataset_id     TEXT NOT NULL,
        rule           TEXT NOT NULL,
        target_doc_id  TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        FOREIGN KEY (dataset_id) REFERENCES datasets(id),
        FOREIGN KEY (target_doc_id) REFERENCES documents(id)
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_documents_dataset ON documents(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(dataset_id, file_hash)",
    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(dataset_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_dataset ON tasks(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_doc ON tasks(doc_id)",
    "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
    "CREATE INDEX IF NOT EXISTS idx_dependencies_dataset ON dependencies(dataset_id)",
    "CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_doc_id)",
]


def main():
    print("[init_db] Connecting to database...")
    conn = get_connection()

    print("[init_db] Creating tables...")
    for stmt in SCHEMA:
        conn.execute(stmt)
        # Extract table name for logging
        name = stmt.split("CREATE TABLE IF NOT EXISTS")[1].split("(")[0].strip()
        print(f"  ✅ {name}")

    print("[init_db] Creating indexes...")
    for stmt in INDEXES:
        conn.execute(stmt)
        idx_name = stmt.split("idx_")[1].split(" ")[0].rstrip("_") if "idx_" in stmt else "?"
        print(f"  ✅ idx_{idx_name}")

    conn.commit()
    conn.close()
    print("\n[init_db] Done. Database initialized successfully.")


if __name__ == "__main__":
    main()
