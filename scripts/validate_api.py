"""
Full-coverage validation for every RagClient endpoint.

Ensures the schema refactoring (PageableDocument / Word / PlainText) has
not broken any server-side API contract.

Test strategy:
  1. list_documents      - connectivity + auto-discover real doc_ids
  2. get_document        - single doc metadata
  3. get_views           - LOD views
  4. query (fusion)      - hybrid retrieval
  5. query_vector        - pure vector retrieval
  6. query_skeleton      - tree skeleton
  7. get_catalog         - lender catalog
  8. resolve_dependencies - dependency graph
  9. locate_pages        - text-to-page; uses fitz to extract real snippets
                           from local PDFs (same approach as bench_locate.py)

Usage:
    python playground/validate_api.py
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

# Make playground importable as top-level package so client.py loads cleanly
# (its TYPE_CHECKING-guarded relative import is skipped at runtime)
sys.path.insert(0, str(Path(__file__).parent))

from client import RagClient  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE   = "http://localhost:8000/" #local
# API_BASE   = "http://13.56.109.233:8000/" #dev
# API_BASE   = "http://54.215.128.27:8000/" #prod
# API_BASE   = "http://18.144.156.86:8000/" #prod
# API_BASE = "http://internal-prod-269090500.us-west-1.elb.amazonaws.com/rag"
DATASET_ID = "mvUisSWfRQx2Ap836jyIy"
TIMEOUT    = 90.0

# Local PDF cache written by bench_locate.py and the worker
PDFS_DIR   = Path(r"C:\Users\xue\.zag\cache\pdfs")

# Generic mortgage query used for query/vector/skeleton tests
QUERY_TEXT = "loan to value ratio maximum"

# A doc_id that is confirmed to have LOD units; used as fallback for skeleton test.
# Update this when the dataset changes.
SKELETON_DOC_ID = "a6764b62b95053aa"

# Critical doc_ids that MUST exist in production dataset
# FannieMae and FreddieMac guidelines are core documents
REQUIRED_DOC_IDS = {
    "883489afd016d9e4": "FannieMae Selling Guide",
    "86ffa34b6ce3cecc": "FreddieMac Selling Guide",
}

EXTRACT_LEN = 120  # chars for text_start snippet
END_LEN     = 120  # chars for text_end snippet


# ── Output helpers ─────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def passed(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  ✅  {label}{suffix}")


def failed(label: str, err: object) -> None:
    print(f"  ❌  {label}  =>  {err}")


# ── Text extraction (fitz) ─────────────────────────────────────────────────────

def extract_snippet(pdf_path: Path) -> tuple[str | None, str | None, int | None]:
    """
    Pull a real (text_start, text_end, expected_page_1based) triple from a PDF.
    Prefers pages from the latter half of the document.
    Tries cross-page first (higher locate coverage), falls back to same-page.
    Returns (None, None, None) when nothing usable is found.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        pages_text = [
            (i, doc[i].get_text().strip())
            for i in range(len(doc))
            if len(doc[i].get_text().strip()) > 200
        ]
        doc.close()

        if not pages_text:
            return None, None, None

        # Bias toward latter half of the document
        half = max(1, len(pages_text) // 2)
        candidates = pages_text[half:]  # latter half first
        if not candidates:
            candidates = pages_text

        cross_page = random.random() < 0.5 and len(candidates) >= 2

        if cross_page:
            idx = random.randint(0, len(candidates) - 2)
            page_idx_a, text_a = candidates[idx]
            page_idx_b, text_b = candidates[idx + 1]
            start = text_a[-EXTRACT_LEN:].strip()
            end   = text_b[:END_LEN].strip()
            if len(start) >= 50 and len(end) >= 50:
                # text_start is on page_idx_a; expected answer is that page (1-based)
                return start, end, page_idx_a + 1

        # Same-page fallback — still prefer latter half
        pool = candidates[:]
        random.shuffle(pool)
        for page_idx, text in pool[:6]:
            if len(text) < EXTRACT_LEN + END_LEN + 50:
                continue
            # Start from ~40% into the page so snippet is clearly mid-document
            offset = max(0, len(text) * 2 // 5)
            start = text[offset: offset + EXTRACT_LEN].strip()
            end   = text[offset + EXTRACT_LEN + 50: offset + EXTRACT_LEN + 50 + END_LEN].strip()
            if len(start) >= 50 and len(end) >= 50:
                return start, end, page_idx + 1

    except Exception:
        pass
    return None, None, None


# ── Individual endpoint tests ──────────────────────────────────────────────────

async def test_health(client: RagClient) -> None:
    section("0 / 10  GET /health  (should be filtered from file log)")
    t = time.perf_counter()
    try:
        resp = await client._http.get(client._base + "/health")
        elapsed = time.perf_counter() - t
        if resp.status_code == 200:
            passed("GET /health", f"status=200, {elapsed:.2f}s")
        else:
            failed("GET /health", f"status={resp.status_code}")
    except Exception as e:
        failed("GET /health", e)


async def test_list_documents(client: RagClient) -> list[dict]:
    section("1 / 10  list_documents()")
    t = time.perf_counter()
    docs = await client.list_documents(limit=10)
    elapsed = time.perf_counter() - t
    if docs:
        passed("list_documents", f"{len(docs)} docs (limit=10), {elapsed:.2f}s")
    else:
        failed("list_documents", "returned empty list")
    return docs or []


async def test_get_document(client: RagClient, doc_id: str) -> None:
    section(f"2 / 10  get_document({doc_id[:16]}…)")
    t = time.perf_counter()
    try:
        doc = await client.get_document(doc_id)
        elapsed = time.perf_counter() - t
        if doc:
            name = doc.get("file_name") or doc.get("name") or "(no name field)"
            passed("get_document", f"file_name={name!r}, {elapsed:.2f}s")
        else:
            failed("get_document", "returned empty dict")
    except Exception as e:
        failed("get_document", e)


async def test_get_views(client: RagClient, doc_id: str) -> None:
    section(f"3 / 10  get_views({doc_id[:16]}…)")
    t = time.perf_counter()
    try:
        views = await client.get_views(doc_id, level="low")
        elapsed = time.perf_counter() - t
        if views is not None:
            passed("get_views", f"{len(views)} view(s), {elapsed:.2f}s")
        else:
            failed("get_views", "returned None")
    except Exception as e:
        failed("get_views", e)


async def test_query(client: RagClient) -> list[dict]:
    section(f"4 / 10  query(fusion)  query={QUERY_TEXT!r}")
    t = time.perf_counter()
    try:
        results = await client.query(QUERY_TEXT, top_k=5)
        elapsed = time.perf_counter() - t
        if results:
            passed("query (fusion)", f"{len(results)} results, {elapsed:.2f}s")
            # Show top-1 doc_id so we can cross-check
            top = results[0]
            print(f"    top-1 doc_id={top.get('doc_id', '?')[:16]}  "
                  f"score={top.get('score', '?'):.3f}")
        else:
            failed("query (fusion)", "returned empty list")
        return results
    except Exception as e:
        failed("query (fusion)", e)
        return []


async def test_query_vector(client: RagClient) -> None:
    section(f"5 / 10  query_vector  query={QUERY_TEXT!r}")
    t = time.perf_counter()
    try:
        results = await client.query_vector(QUERY_TEXT, top_k=5)
        elapsed = time.perf_counter() - t
        if results:
            passed("query_vector", f"{len(results)} results, {elapsed:.2f}s")
        else:
            failed("query_vector", "returned empty list")
    except Exception as e:
        failed("query_vector", e)


async def test_query_skeleton(client: RagClient, doc_id: str) -> None:
    section(f"6 / 10  query_skeleton(doc={doc_id[:16]}…)")
    t = time.perf_counter()
    try:
        result = await client.query_skeleton(QUERY_TEXT, doc_id=doc_id)
        elapsed = time.perf_counter() - t
        if result:
            keys = list(result.keys())[:4]
            passed("query_skeleton", f"keys={keys}, {elapsed:.2f}s")
        else:
            failed("query_skeleton", "returned empty dict")
    except Exception as e:
        failed("query_skeleton", e)


async def test_get_catalog(client: RagClient) -> None:
    section("7 / 10  get_catalog()")
    t = time.perf_counter()
    try:
        catalog = await client.get_catalog()
        elapsed = time.perf_counter() - t
        if catalog:
            keys = list(catalog.keys())[:4]
            passed("get_catalog", f"keys={keys}, {elapsed:.2f}s")
        else:
            failed("get_catalog", "returned empty dict")
    except Exception as e:
        failed("get_catalog", e)


async def test_resolve_dependencies(client: RagClient, doc_ids: list[str]) -> None:
    section(f"8 / 10  resolve_dependencies({len(doc_ids)} doc_ids)")
    t = time.perf_counter()
    try:
        result = await client.resolve_dependencies(doc_ids)
        elapsed = time.perf_counter() - t
        if result is not None:
            passed("resolve_dependencies", f"{len(result)} entries, {elapsed:.2f}s")
        else:
            failed("resolve_dependencies", "returned None")
    except Exception as e:
        failed("resolve_dependencies", e)


async def test_locate_pages(client: RagClient, docs: list[dict]) -> None:
    section("9 / 10  locate_pages()  (fitz extraction from local PDFs)")

    items: list[dict] = []

    # Primary: scan PDFS_DIR exactly like bench_locate.py
    if PDFS_DIR.exists():
        pdfs = list(PDFS_DIR.glob("*.pdf"))
        random.shuffle(pdfs)
        for pdf in pdfs[:20]:  # try up to 20, stop when we have 5 good items
            doc_id = pdf.stem[:16]
            text_start, text_end, expected_page = extract_snippet(pdf)
            if text_start:
                items.append({
                    "request_id": doc_id,
                    "doc_id": doc_id,
                    "text_start": text_start,
                    "text_end": text_end,
                    "_expected_page": expected_page,
                })
            if len(items) >= 5:
                break
    else:
        print(f"  (PDFS_DIR not found: {PDFS_DIR})")

    # Fallback: use doc_ids from list_documents with a generic query string.
    # This won't score "found=true" but at least validates the endpoint is alive.
    if not items and docs:
        print("  (no local PDFs available; sending generic snippet as smoke test)")
        for doc in docs[:5]:
            doc_id = (doc.get("id") or doc.get("doc_id") or
                      doc.get("document_id") or "")
            if doc_id:
                items.append({
                    "request_id": doc_id,
                    "doc_id": doc_id,
                    "text_start": "loan to value ratio",
                    "text_end": None,
                })

    if not items:
        failed("locate_pages", "could not build any items (no PDFs, no doc_ids)")
        return

    t = time.perf_counter()
    try:
        page_map = await client.locate_pages(items)
        elapsed = time.perf_counter() - t

        found = sum(1 for v in page_map.values() if v > 0)
        passed(
            "locate_pages",
            f"{found}/{len(items)} located, {elapsed:.2f}s"
        )

        # Per-item breakdown
        for item in items:
            rid      = item["request_id"]
            page     = page_map.get(rid, 0)
            expected = item.get("_expected_page")
            found_s  = f"p{page}" if page > 0 else "not found"
            if expected is not None:
                match = "OK" if page == expected else "MISMATCH"
                expect_s = f"expected=p{expected}  found={found_s}  [{match}]"
            else:
                expect_s = f"found={found_s}"
            snip = item["text_start"][:50].replace("\n", " ")
            print(f"    [{rid[:16]}]  {expect_s:<42}  '{snip}…'")

    except Exception as e:
        failed("locate_pages", e)


async def test_required_documents(client: RagClient) -> None:
    """Verify that critical documents (FannieMae, FreddieMac) exist in the dataset."""
    section("10 / 10  required_documents (FannieMae / FreddieMac)")

    all_passed = True
    for doc_id, name in REQUIRED_DOC_IDS.items():
        try:
            doc = await client.get_document(doc_id)
            if doc and doc.get("id") or doc.get("doc_id") or doc.get("file_name"):
                file_name = doc.get("file_name") or doc.get("name") or "(no name)"
                unit_count = doc.get("unit_count")
                unit_str = f", units={unit_count}" if unit_count is not None else "  ⚠️ units=null"
                passed(f"doc_id={doc_id[:16]}…", f"{name} ({file_name}{unit_str})")
            else:
                failed(f"doc_id={doc_id[:16]}…", f"{name} - returned empty response")
                all_passed = False
        except Exception as e:
            failed(f"doc_id={doc_id[:16]}…", f"{name} - {e}")
            all_passed = False

    if not all_passed:
        print("\n  WARNING: Critical documents missing! Check dataset integrity.")


# ── Main ───────────────────────────────────────────────────────────────────────

def _extract_doc_id(doc: dict) -> str:
    """Best-effort extraction of doc_id from any list_documents() entry."""
    for key in ("id", "doc_id", "document_id", "hash", "md5"):
        val = doc.get(key)
        if val and isinstance(val, str):
            return val
    return ""


async def main() -> None:
    print("=" * 64)
    print("  RagClient endpoint validation")
    print(f"  API     : {API_BASE}")
    print(f"  Dataset : {DATASET_ID}")
    print("=" * 64)

    client = RagClient(API_BASE, DATASET_ID, timeout=TIMEOUT)

    try:
        await test_health(client)

        # 1. list_documents — must succeed; all later tests depend on it
        docs = await test_list_documents(client)
        if not docs:
            print("\n  Aborting: list_documents returned nothing.\n")
            return

        # Resolve doc_ids
        doc_ids = [_extract_doc_id(d) for d in docs]
        doc_ids = [d for d in doc_ids if d]
        if not doc_ids:
            print(f"\n  Cannot determine doc_id field.  Sample keys: {list(docs[0].keys())}\n")
            return

        test_id = random.choice(doc_ids[:min(10, len(doc_ids))])
        print(f"\n  doc_ids resolved ({len(doc_ids)} total).  Using {test_id[:16]}… for single-doc tests.")

        # 2-9 in order
        await test_get_document(client, test_id)
        await test_get_views(client, test_id)
        query_results = await test_query(client)
        await test_query_vector(client)

        # Skeleton works only on docs with LOD units; use known-good id, fall back to query top-1
        skeleton_id = SKELETON_DOC_ID or (
            query_results[0].get("doc_id", "") if query_results else ""
        ) or test_id
        await test_query_skeleton(client, skeleton_id)

        await test_get_catalog(client)
        await test_resolve_dependencies(client, doc_ids[:3])
        await test_locate_pages(client, docs)
        await test_required_documents(client)

        print(f"\n{'=' * 64}")
        print("  Validation complete.")
        print(f"{'=' * 64}\n")

    except Exception as e:
        print(f"\n  Unexpected top-level error: {e}")
        raise
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
