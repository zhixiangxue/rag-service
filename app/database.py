"""Database models and connection management."""
import sqlite3
from typing import Optional
from datetime import datetime
import uuid

from .config import DATABASE_PATH


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    return conn


def init_db():
    """Initialize database tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Dataset table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            workspace_dir TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            file_type TEXT NOT NULL,
            file_hash TEXT,
            status TEXT NOT NULL,
            task_id INTEGER,
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
    
    # Task table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL,
            doc_id INTEGER NOT NULL,
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
