"""Unified HTTP client for all RAG service API endpoints.

One RagClient instance per session; all tools share it.
Eliminates N independent httpx connection pools, centralizes
retry/timeout config, and exposes locate_pages as a first-class method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import httpx

if TYPE_CHECKING:
    from .evidence import EvidenceBlock


class RagClient:
    """Single httpx connection pool wrapping every RAG service endpoint.

    Intended lifecycle: create once per session, inject into every tool,
    close after the session ends.

    Args:
        base_url:   RAG service root (e.g. "http://13.56.109.233:8000").
        dataset_id: Dataset identifier (e.g. "mvUisSWfRQx2Ap836jyIy").
        timeout:    Default request timeout in seconds.
    """

    def __init__(self, base_url: str, dataset_id: str, timeout: float = 30.0, access_key: str = "") -> None:
        self._base = base_url.rstrip("/")
        self._dataset_id = dataset_id
        # trust_env=False: ignore system HTTP_PROXY/HTTPS_PROXY so localhost
        # requests are not routed through a system proxy.
        _headers = {"X-Api-Key": access_key} if access_key else {}
        self._http = httpx.AsyncClient(timeout=timeout, trust_env=False, headers=_headers)

    # ── Convenience read-only properties ──────────────────────────────────────

    @property
    def base_url(self) -> str:
        """RAG service root URL (no trailing slash)."""
        return self._base

    @property
    def dataset_id(self) -> str:
        """Dataset identifier."""
        return self._dataset_id

    # ── Query endpoints ────────────────────────────────────────────────────────

    async def query(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """POST /datasets/{id}/query/fusion — hybrid (BM25 + vector) retrieval."""
        url = f"{self._base}/datasets/{self._dataset_id}/query/fusion"
        payload = {"query": query, "top_k": top_k, "filters": filters or {}, "min_score": min_score}
        r = await self._http.post(url, json=payload)
        r.raise_for_status()
        return r.json()["data"]

    async def query_vector(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """POST /datasets/{id}/query/vector — pure vector similarity search."""
        url = f"{self._base}/datasets/{self._dataset_id}/query/vector"
        payload = {"query": query, "top_k": top_k, "filters": filters or {}}
        r = await self._http.post(url, json=payload)
        r.raise_for_status()
        return r.json()["data"]

    async def query_skeleton(
        self,
        query: str,
        doc_id: str,
        mode: str = "accurate",
    ) -> dict:
        """POST /datasets/{id}/query/tree/skeleton — document skeleton search."""
        url = f"{self._base}/datasets/{self._dataset_id}/query/tree/skeleton"
        payload = {"query": query, "doc_id": doc_id, "mode": mode}
        r = await self._http.post(url, json=payload)
        r.raise_for_status()
        return r.json().get("data") or {}

    # ── Document endpoints ─────────────────────────────────────────────────────

    async def get_views(self, doc_id: str, level: str = "low") -> list[dict]:
        """GET /datasets/{id}/documents/{doc_id}/views — LOD view content."""
        url = f"{self._base}/datasets/{self._dataset_id}/documents/{doc_id}/views"
        r = await self._http.get(url, params={"level": level})
        r.raise_for_status()
        return r.json().get("data") or []

    async def get_document(self, doc_id: str) -> dict:
        """GET /datasets/{id}/documents/{doc_id} — document metadata."""
        url = f"{self._base}/datasets/{self._dataset_id}/documents/{doc_id}"
        r = await self._http.get(url)
        r.raise_for_status()
        return r.json().get("data") or {}

    async def list_documents(self, limit: Optional[int] = None) -> list[dict]:
        """GET /datasets/{id}/documents — document list (use limit to avoid huge payloads)."""
        url = f"{self._base}/datasets/{self._dataset_id}/documents"
        params = {}
        if limit is not None:
            params["limit"] = limit
        r = await self._http.get(url, params=params)
        r.raise_for_status()
        return r.json().get("data") or []

    # ── Catalog endpoints ──────────────────────────────────────────────────────

    async def get_catalog(self) -> dict:
        """GET /datasets/{id}/catalog — lender/program catalog tree."""
        url = f"{self._base}/datasets/{self._dataset_id}/catalog"
        r = await self._http.get(url)
        r.raise_for_status()
        return r.json().get("data") or {}

    async def resolve_dependencies(self, doc_ids: list[str]) -> dict:
        """POST /datasets/{id}/dependencies/resolve — raw dependency graph.

        Returns the raw ``data`` dict from the API (keyed by doc_id with
        ``lender``, ``overlay``, ``doc`` bucket lists).  Callers are
        responsible for merging the result with their seed list.
        """
        url = f"{self._base}/datasets/{self._dataset_id}/dependencies/resolve"
        r = await self._http.post(url, json={"doc_ids": doc_ids})
        r.raise_for_status()
        return r.json().get("data") or {}

    # ── Page location endpoint ─────────────────────────────────────────────────

    async def locate_pages(self, items: list[dict]) -> dict[str, int]:
        """POST /datasets/{id}/documents/locate — resolve text snippets to page numbers.

        Args:
            items: List of ``{request_id, doc_id, text_start, text_end?}`` dicts.

        Returns:
            Mapping ``{request_id: first_page_number}``; 0 when not found.
            Empty dict on any error — page info is diagnostic; callers must
            never depend on it for correctness.
        """
        if not items:
            return {}
        url = f"{self._base}/datasets/{self._dataset_id}/documents/locate"
        try:
            r = await self._http.post(url, json={"items": items})
            r.raise_for_status()
            results = (r.json().get("data") or {}).get("results") or []
            page_map: dict[str, int] = {}
            for entry in results:
                rid = entry.get("request_id") or ""
                pages = entry.get("page_numbers") or []
                page_map[rid] = pages[0] if pages else 0
            return page_map
        except Exception:
            return {}  # best-effort; never block the caller

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the underlying connection pool."""
        await self._http.aclose()


