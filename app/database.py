"""Database connection management."""
import sqlite3
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from nanoid import generate

from .config import DATABASE_URI


def generate_id() -> str:
    """Generate nanoid for primary key."""
    return generate()


def get_connection():
    """Get database connection.

    Parses DATABASE_URI to select the appropriate driver:
      sqlite:///./path  →  local SQLite
      http://host:port  →  rqlite over HTTP
      https://host:port →  rqlite over HTTPS
    """
    if DATABASE_URI.startswith("sqlite:///"):
        path = DATABASE_URI[len("sqlite:///"):]
        db_path = Path(path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # rqlite: http:// or https://
    from .pyrqlite import dbapi2
    parsed = urlparse(DATABASE_URI)
    return dbapi2.connect(
        scheme=parsed.scheme,
        host=parsed.hostname,
        port=parsed.port or 4001,
    )


def now() -> str:
    """Get current timestamp as ISO string."""
    return datetime.utcnow().isoformat() + "Z"
