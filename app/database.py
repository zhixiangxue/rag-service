"""Database models and connection management."""
import sqlite3
from typing import Optional
from datetime import datetime
from nanoid import generate
from pathlib import Path

from .config import DATABASE_PATH


def generate_id() -> str:
    """Generate nanoid for primary key."""
    return generate()


def get_connection():
    """Get database connection with WAL mode for better concurrency."""
    # Auto-create parent directory if it doesn't exist
    db_path = Path(DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    
    # Enable WAL mode for better read/write concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    # Set busy timeout to 5 seconds (wait if database is locked)
    conn.execute("PRAGMA busy_timeout=5000")
    
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Dataset table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            engine TEXT NOT NULL DEFAULT 'qdrant',
            config TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Document table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            workspace_dir TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT,
            metadata TEXT,
            status TEXT NOT NULL,
            task_id TEXT,
            unit_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id)
        )
    """)
    
    # Create index for duplicate detection
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_dataset_hash 
        ON documents(dataset_id, file_hash)
    """)
    
    # Create UNIQUE constraint for duplicate prevention (if not exists)
    # Note: SQLite doesn't support ALTER TABLE ADD CONSTRAINT directly
    # So we check and create it only if table is being created
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uniq_documents_dataset_hash 
        ON documents(dataset_id, file_hash)
    """)
    
    # Create index for fast document lookup by id and dataset_id
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_id_dataset 
        ON documents(id, dataset_id)
    """)
    
    # Create index for dataset_id queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_dataset 
        ON documents(dataset_id)
    """)
    
    # Task table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'classic',
            status TEXT NOT NULL,
            progress INTEGER DEFAULT 0,
            error_message TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id),
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        )
    """)
    
    conn.commit()
    conn.close()



def now() -> str:
    """Get current timestamp as ISO string."""
    return datetime.utcnow().isoformat() + "Z"
