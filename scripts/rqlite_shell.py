"""
Interactive SQL shell for rqlite / SQLite.
Reuses the project's database.py connection logic.

Usage:
    cd rag-service
    python scripts/rqlite_shell.py

Meta-commands:
    .tables          list all tables
    .schema [table]  show CREATE statement(s)
    .q / .quit       exit
    ;                terminate a multi-line statement
"""

import os
import sys
import csv
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

# Load .env from rag-service/
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

# Add rag-service to path so we can import app.*
rag_service_dir = str(Path(__file__).parent.parent)
if rag_service_dir not in sys.path:
    sys.path.insert(0, rag_service_dir)

from app.database import get_connection, DATABASE_URI  # noqa: E402


# ── last result cache (for .export) ──────────────────────────────────────────

_last_headers: list = []
_last_rows: list = []


# ── formatting ────────────────────────────────────────────────────────────────

def _col_widths(headers, rows):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val) if val is not None else "NULL"))
    return widths


def print_table(headers, rows):
    if not headers:
        print("(no columns returned)")
        return
    widths = _col_widths(headers, rows)
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"
    print(sep)
    print(fmt.format(*[str(h) for h in headers]))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(v) if v is not None else "NULL" for v in row]))
    print(sep)
    print(f"({len(rows)} row{'s' if len(rows) != 1 else ''})")


# ── meta-commands ─────────────────────────────────────────────────────────────

def cmd_tables(conn):
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    rows = cur.fetchall()
    if not rows:
        print("(no tables)")
        return
    for r in rows:
        print(r[0] if not hasattr(r, 'keys') else list(r.values())[0])


def cmd_schema(conn, table=None):
    if table:
        cur = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
        )
    else:
        cur = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    rows = cur.fetchall()
    if not rows:
        print("(not found)")
        return
    for r in rows:
        sql = r[0] if not hasattr(r, 'keys') else list(r.values())[0]
        if sql:
            print(sql)
            print()


# ── query execution ───────────────────────────────────────────────────────────

def run_sql(conn, sql):
    global _last_headers, _last_rows
    sql = sql.strip()
    if not sql:
        return

    # rqlite cursor vs sqlite3 cursor differ slightly in row representation
    cur = conn.execute(sql)

    # For DML (UPDATE/INSERT/DELETE), pyrqlite sets description=None.
    # Calling fetchall() on such a cursor triggers an infinite loop bug in
    # pyrqlite (rownumber never advances when _rows is empty but rowcount > 0).
    # Check description first to avoid this.
    if cur.description is None:
        rc = getattr(cur, 'rowcount', -1)
        if rc >= 0:
            print(f"OK  ({rc} row{'s' if rc != 1 else ''} affected)")
        else:
            print("OK")
        return

    rows = cur.fetchall()
    headers = [d[0] for d in cur.description]
    # normalise rows: sqlite3.Row → list, pyrqlite Row → list
    normalized = []
    for r in rows:
        if hasattr(r, 'keys'):
            normalized.append([r[k] for k in r.keys()])
        elif hasattr(r, 'values'):
            normalized.append(list(r.values()))
        else:
            normalized.append(list(r))
    _last_headers = headers
    _last_rows = normalized
    print_table(headers, normalized)
    print("(tip: type .export to save this result as CSV)")


def cmd_export():
    """Export the last query result to a CSV file in the system tmp directory."""
    if not _last_headers:
        print("Nothing to export — run a SELECT query first.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(tempfile.gettempdir()) / f"rqlite_export_{timestamp}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_last_headers)
        writer.writerows(_last_rows)
    print(f"Exported {len(_last_rows)} rows → {out_path}")


# ── REPL ──────────────────────────────────────────────────────────────────────

def repl(conn):
    buf = []
    print(f"Connected: {DATABASE_URI}")
    print("Type SQL statements (end with ;) or .tables / .schema / .export / .q\n")

    while True:
        prompt = "   > " if buf else "sql> "
        try:
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            break

        stripped = line.strip()
        # For meta-command detection, ignore a trailing semicolon
        meta_stripped = stripped.rstrip(";")

        # meta-commands (only at the start of a fresh statement)
        if not buf and meta_stripped.startswith("."):
            parts = meta_stripped.split()
            cmd = parts[0].lower()
            if cmd in (".q", ".quit", ".exit"):
                break
            elif cmd == ".export":
                try:
                    cmd_export()
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd == ".tables":
                try:
                    cmd_tables(conn)
                except Exception as e:
                    print(f"Error: {e}")
            elif cmd == ".schema":
                table = parts[1] if len(parts) > 1 else None
                try:
                    cmd_schema(conn, table)
                except Exception as e:
                    print(f"Error: {e}")
            else:
                print(f"Unknown command: {cmd}")
            continue

        buf.append(line)

        # execute when statement ends with ;
        if stripped.endswith(";"):
            sql = " ".join(buf).rstrip(";")
            buf.clear()
            try:
                run_sql(conn, sql)
            except Exception as e:
                print(f"Error: {e}")
                if os.getenv("DEBUG"):
                    traceback.print_exc()

    print("bye.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        conn = get_connection()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    try:
        repl(conn)
    finally:
        conn.close()
