"""
Migrate ALL units from Qdrant to Meilisearch.

Reads connection config from rag-service/.env:
  VECTOR_STORE_HOST / VECTOR_STORE_PORT  ->  Qdrant
  DEFAULT_COLLECTION_NAME                ->  default collection / index name
  MEILISEARCH_HOST                       ->  Meilisearch

All settings can be overridden interactively before migration starts.

Usage:
    python rag-service/scripts/migrate_qdrant_to_meili.py
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List

# ------------------------------------------------------------------
# Load .env from rag-service directory (sibling of this script's dir)
# ------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).parent.resolve()
_ENV_FILE = _SCRIPT_DIR.parent / ".env"
if _ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ENV_FILE)
    print(f"[Config] Loaded .env from: {_ENV_FILE}")
else:
    print(f"[Config] WARNING: .env not found at {_ENV_FILE}, falling back to env vars / defaults")

from qdrant_client import QdrantClient
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from zag.schemas.unit import BaseUnit
from zag.storages.vector.qdrant import QdrantVectorStore
from zag.indexers.fulltext_indexer import FullTextIndexer

# ------------------------------------------------------------------
# Defaults (read from .env)
# ------------------------------------------------------------------
_DEFAULT_QDRANT_HOST  = os.getenv("VECTOR_STORE_HOST", "localhost")
_DEFAULT_QDRANT_PORT  = int(os.getenv("VECTOR_STORE_PORT", "6333"))
_DEFAULT_COLLECTION   = os.getenv("DEFAULT_COLLECTION_NAME", "mai")
_DEFAULT_MEILI_URL    = os.getenv("MEILISEARCH_HOST", "http://localhost:7700")

SCROLL_BATCH = 500   # points per Qdrant scroll
WRITE_BATCH  = 200   # units per Meili add() call

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _prompt(label: str, default: str) -> str:
    """Show a prompt with default value; return default if user presses Enter."""
    value = input(f"  {label} [{default}]: ").strip()
    return value if value else default


def _confirm_yn(label: str) -> bool:
    """Yes/No prompt, default No."""
    value = input(f"  {label} [y/N]: ").strip().lower()
    return value in ("y", "yes")


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _write_batch(indexer: FullTextIndexer, units: List[BaseUnit]) -> None:
    """Write a batch of units to Meilisearch, with exponential-backoff retry."""
    indexer.add(units)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print("=" * 62)
    print("        Qdrant  →  Meilisearch  Migration Tool")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Step 1: Interactive config confirmation
    # ------------------------------------------------------------------
    print("\n[Step 1] Confirm migration settings\n")
    print("  Press Enter to accept the value shown in brackets.\n")

    qdrant_host = _prompt("Qdrant host         (source)", _DEFAULT_QDRANT_HOST)
    qdrant_port = int(_prompt("Qdrant port         (source)", str(_DEFAULT_QDRANT_PORT)))
    qdrant_col  = _prompt("Qdrant collection   (source)", _DEFAULT_COLLECTION)

    meili_url   = _prompt("Meilisearch URL     (dest)  ", _DEFAULT_MEILI_URL)
    meili_index = _prompt("Meilisearch index   (dest)  ", _DEFAULT_COLLECTION)

    clear_index = _confirm_yn(
        f"Clear existing Meilisearch index '{meili_index}' before migration?"
    )

    # Print summary box
    w = 58
    print(f"\n  ┌{'─' * w}┐")
    print(f"  │  {'Migration Summary':^{w - 2}}  │")
    print(f"  ├{'─' * w}┤")
    print(f"  │  Source : Qdrant  {qdrant_host}:{qdrant_port:<5}  collection → {qdrant_col:<15}│")
    print(f"  │  Dest   : Meili   {meili_url:<30}  index → {meili_index:<7}│")
    print(f"  │  Clear  : {'YES  ⚠️  (existing data will be deleted)' if clear_index else 'NO   (append / upsert)':^{w - 2}}│")
    print(f"  └{'─' * w}┘")

    proceed = input("\n  Type 'yes' to start migration: ").strip()
    if proceed != "yes":
        print("  Aborted.")
        return

    # ------------------------------------------------------------------
    # Step 2: Init clients
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Connecting to Qdrant and Meilisearch...")

    raw_client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60)
    print(f"  Qdrant  : {qdrant_host}:{qdrant_port}  OK")

    # Reuse _point_to_unit without needing an embedder
    store = object.__new__(QdrantVectorStore)
    store.client = raw_client
    store.collection_name = qdrant_col

    indexer = FullTextIndexer(
        url=meili_url,
        index_name=meili_index,
        primary_key="unit_id",
        auto_create_index=True,
    )
    print(f"  Meili   : {meili_url}  index={meili_index}  OK")

    if clear_index:
        try:
            indexer.clear()
            print(f"  Cleared Meilisearch index '{meili_index}'.")
        except Exception as e:
            print(f"  Clear skipped: {e}")

    # Configure filterable / searchable attributes
    indexer.configure_settings(
        searchable_attributes=["*"],
        filterable_attributes=[
            "unit_id",
            "doc_id",
            "unit_type",
            "metadata.context_path",
            "metadata.page_numbers",
            "metadata.keywords",
            "metadata.document.file_name",
            "metadata.custom.mode",
            "metadata.custom.overlays",
            "metadata.custom.guideline",
            "metadata.custom.lender",
        ],
    )

    # ------------------------------------------------------------------
    # Step 3: Scroll Qdrant + write to Meili
    # ------------------------------------------------------------------
    total_written = 0
    offset        = None
    batch_num     = 0
    t0            = time.time()

    print(f"\n[Step 3] Migrating  (scroll_batch={SCROLL_BATCH}, write_batch={WRITE_BATCH})...")

    while True:
        points, next_offset = raw_client.scroll(
            collection_name=qdrant_col,
            with_payload=True,
            with_vectors=False,
            limit=SCROLL_BATCH,
            offset=offset,
        )

        if not points:
            break

        units = [store._point_to_unit(p) for p in points]

        for i in range(0, len(units), WRITE_BATCH):
            sub = units[i: i + WRITE_BATCH]
            _write_batch(indexer, sub)
            total_written += len(sub)

        batch_num += 1
        elapsed = time.time() - t0
        speed   = total_written / elapsed if elapsed > 0 else 0
        print(
            f"  Batch {batch_num:4d} | +{len(units):4d} units | "
            f"total={total_written:6d} | {speed:.0f} u/s"
        )

        if not next_offset:
            break
        offset = next_offset

    # ------------------------------------------------------------------
    # Step 4: Final verification
    # ------------------------------------------------------------------
    elapsed  = time.time() - t0
    stats    = indexer.index.get_stats()
    in_meili = getattr(stats, "number_of_documents", None) or (
        stats.get("numberOfDocuments") if isinstance(stats, dict) else "?"
    )

    print(f"\n[Done] Written {total_written} units in {elapsed:.1f}s")
    print(f"       Meilisearch '{meili_index}' now has {in_meili} documents")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted, exiting...")
