"""FastAPI application entry point."""
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fastapi import FastAPI, Request, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, HTMLResponse
import os

from .routers import datasets, documents, tasks, query, units, health, demo, graph, utility
from .routers import dependencies
from . import config

# Logging configuration: timestamp + level + logger name + message
_LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-5s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=_LOG_FORMAT,
    datefmt=_LOG_DATEFMT,
)

# File handler: plain-text (no ANSI), 10 MB × 5 backups
_log_dir = Path(__file__).parent.parent / "log"
_log_dir.mkdir(parents=True, exist_ok=True)
_file_handler = RotatingFileHandler(
    _log_dir / "access.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))


class _FileLogFilter(logging.Filter):
    """Suppress noisy access records from the file log."""
    _SKIP = {
        ("GET", "/health"),
    }

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for method, path in self._SKIP:
            if msg.startswith(method) and path in msg:
                return False
        return True


_file_handler.addFilter(_FileLogFilter())
_root_logger = logging.getLogger()
if not any(h.baseFilename == str(_log_dir / "access.log") for h in _root_logger.handlers if hasattr(h, "baseFilename")):
    _root_logger.addHandler(_file_handler)

# Enable debug-level sub-timing from query router with: LOG_LEVEL=DEBUG
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.getLogger("rag").setLevel(getattr(logging, _log_level, logging.INFO))

logger = logging.getLogger("rag.main")


def verify_api_key(x_api_key: str = Header(default="")) -> None:
    """Validate X-Api-Key header. Skipped when ACCESS_KEY env var is empty."""
    if not config.ACCESS_KEY:
        return
    if x_api_key != config.ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# Create FastAPI app
_dev_mode = os.getenv("DEV_MODE", "false").lower() == "true"

# DOCS_TOKEN: when set, /docs and /openapi.json require ?token=<value> in URL.
# Empty = docs disabled (DEV_MODE=false behavior). DEV_MODE=true overrides all checks.
_docs_token = os.getenv("DOCS_TOKEN", "").strip()
_docs_enabled = _dev_mode or bool(_docs_token)

# Always disable built-in docs — served manually below so we can pass token to openapi_url
app = FastAPI(
    title="RAG Service",
    description="RAG service layer built on top of Zag framework",
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Access log middleware: logs method, path, status, and elapsed time for every request
_access_logger = logging.getLogger("rag.access")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Guard /openapi.json with token when DOCS_TOKEN is set and not in dev mode
    # (/docs is handled by its own route below, which injects the token into openapi_url)
    if _docs_token and not _dev_mode:
        if request.url.path == "/openapi.json":
            if request.query_params.get("token") != _docs_token:
                return HTMLResponse(status_code=403, content="403 Forbidden")

    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0
    client = request.client.host if request.client else "-"
    _access_logger.info(
        "%s %s %d %.3fs %s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
        client,
    )
    return response

# Mount static files for distributed worker access
from .config import UPLOAD_DIR
upload_path = os.path.abspath(UPLOAD_DIR)
if os.path.exists(upload_path):
    app.mount("/files", StaticFiles(directory=upload_path), name="files")
    print(f"[Main] Static files mounted at /files -> {upload_path}")
else:
    print(f"[Main] WARNING: Upload directory not found: {upload_path}")

# Register routers
app.include_router(health.router)
_auth = [Depends(verify_api_key)]
app.include_router(datasets.router,    dependencies=_auth)
app.include_router(documents.router,   dependencies=_auth)
app.include_router(tasks.router,       dependencies=_auth)
app.include_router(query.router,       dependencies=_auth)
app.include_router(units.router,       dependencies=_auth)
app.include_router(demo.router,        dependencies=_auth)
app.include_router(graph.router,       dependencies=_auth)
app.include_router(dependencies.router, dependencies=_auth)
app.include_router(utility.router,     dependencies=_auth)


@app.get("/")
async def root():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


if _docs_enabled:
    from fastapi.openapi.docs import get_swagger_ui_html

    @app.get("/docs", include_in_schema=False)
    async def swagger_ui(token: str = ""):
        """Serve Swagger UI, protected by DOCS_TOKEN when set."""
        if _docs_token and not _dev_mode and token != _docs_token:
            return HTMLResponse(status_code=403, content="403 Forbidden")
        # Inject token into the openapi.json URL so Swagger UI can fetch the schema
        openapi_url = f"/openapi.json?token={token}" if (_docs_token and not _dev_mode) else "/openapi.json"
        return get_swagger_ui_html(openapi_url=openapi_url, title=app.title)


# ── Startup dependency check (runs for both gunicorn and python -m app.main) ────

def check_dependencies() -> bool:
    """Check DB, Qdrant, Redis, and Meilisearch. Fail-fast: stops on first failure and marks rest as skipped."""
    import requests as _requests

    def _check_db():
        db_uri = config.DATABASE_URI
        if db_uri.lower().startswith("sqlite"):
            db_path_str = db_uri[len("sqlite:///"):]
            db_path = Path(db_path_str)
            if not db_path.is_absolute():
                db_path = (Path.cwd() / db_path).resolve()
            if db_path.exists():
                return True, f"[DB] SQLite file found: {db_path}"
            return False, f"[DB] SQLite file not found: {db_path}"
        elif db_uri.startswith("http://") or db_uri.startswith("https://"):
            try:
                resp = _requests.get(f"{db_uri.rstrip('/')}/status", timeout=5)
                resp.raise_for_status()
                return True, f"[DB] rqlite reachable: {db_uri}"
            except Exception as exc:
                return False, f"[DB] Cannot connect to rqlite: {exc}"
        return False, f"[DB] Unrecognised DB URI scheme: {db_uri}"

    def _check_qdrant():
        qdrant_url = f"http://{config.VECTOR_STORE_HOST}:{config.VECTOR_STORE_PORT}"
        try:
            resp = _requests.get(f"{qdrant_url}/healthz", timeout=15)
            resp.raise_for_status()
            return True, f"[Qdrant] Reachable: {qdrant_url}"
        except Exception as exc:
            return False, f"[Qdrant] Cannot connect to {qdrant_url}: {exc}"

    def _check_redis():
        try:
            import redis as _redis
            r = _redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            r.ping()
            return True, f"[Redis] Reachable: {config.REDIS_HOST}:{config.REDIS_PORT}"
        except Exception as exc:
            return False, f"[Redis] Cannot connect to {config.REDIS_HOST}:{config.REDIS_PORT}: {exc}"

    def _check_meilisearch():
        host = config.MEILISEARCH_HOST
        if not host:
            return True, "[Meilisearch] Not configured, skipping"
        url = f"{host.rstrip('/')}/health"
        try:
            resp = _requests.get(url, timeout=5)
            resp.raise_for_status()
            return True, f"[Meilisearch] Reachable: {host}"
        except Exception as exc:
            return False, f"[Meilisearch] Cannot connect to {host}: {exc}"

    checks = [
        ("[DB]",          _check_db),
        ("[Qdrant]",      _check_qdrant),
        ("[Redis]",       _check_redis),
        ("[Meilisearch]", _check_meilisearch),
    ]

    print("[Main] Checking dependencies...")
    for i, (label, fn) in enumerate(checks):
        ok, msg = fn()
        if ok:
            print(f"  ✅ {msg}")
        else:
            print(f"  ❌ {msg}")
            for skipped_label, _ in checks[i + 1:]:
                print(f"  ⌛ {skipped_label} (not checked)")
            return False
    return True


if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT

    def print_config_summary():
        """Print configuration summary before startup."""
        import os
        print("\n" + "=" * 60)
        print("  RAG Service Configuration Summary")
        print("=" * 60)
        print(f"\n[Database]")
        print(f"  DATABASE_URI: {config.DATABASE_URI}")
        print(f"\n[Cache]")
        _pdf_dir = os.path.abspath(config.PDF_FILES_DIR)
        _pdf_count = len(os.listdir(_pdf_dir)) if os.path.isdir(_pdf_dir) else 0
        print(f"  PDF_FILES_DIR: {_pdf_dir} ({_pdf_count})")
        print(f"\n[Vector Store]")
        print(f"  VECTOR_STORE_TYPE: {config.VECTOR_STORE_TYPE}")
        print(f"  VECTOR_STORE_HOST: {config.VECTOR_STORE_HOST}")
        print(f"  VECTOR_STORE_PORT: {config.VECTOR_STORE_PORT}")
        print(f"  VECTOR_STORE_GRPC_PORT: {config.VECTOR_STORE_GRPC_PORT}")
        print(f"\n[Meilisearch]")
        print(f"  MEILISEARCH_HOST: {config.MEILISEARCH_HOST or '❌ not set'}")
        print(f"  MEILISEARCH_API_KEY: {'✅ set' if config.MEILISEARCH_API_KEY else '❌ not set'}")
        print(f"\n[Embedding]")
        print(f"  EMBEDDING_URI: {config.EMBEDDING_URI}")
        print(f"  OPENAI_API_KEY: {'✅ set' if config.OPENAI_API_KEY else '❌ not set'}")
        print(f"\n[Reranker]")
        print(f"  RERANKER_URI: {config.RERANKER_URI}")
        print(f"  COHERE_API_KEY: {'✅ set' if config.COHERE_API_KEY else '❌ not set'}")
        print(f"\n[LLM]")
        print(f"  LLM_PROVIDER: {config.LLM_PROVIDER}")
        print(f"  LLM_MODEL: {config.LLM_MODEL}")
        print(f"\n[Redis]")
        print(f"  REDIS_HOST: {config.REDIS_HOST}")
        print(f"  REDIS_PORT: {config.REDIS_PORT}")
        print(f"\n[API Server]")
        print(f"  API_HOST: {API_HOST}")
        print(f"  API_PORT: {API_PORT}")
        print("=" * 60)

    def confirm_config() -> bool:
        """Interactive confirmation of configuration before startup."""
        print_config_summary()

        if not sys.stdin.isatty():
            print("\n[Main] Running in non-interactive mode, skipping confirmation.")
            return True

        try:
            response = input("\n[Main] Do you want to proceed with this configuration? [Y/n]: ").strip().lower()
            if response in ('', 'y', 'yes'):
                print("[Main] Configuration confirmed. Starting service...\n")
                return True
            else:
                print("[Main] Configuration rejected. Exiting.")
                return False
        except (EOFError, KeyboardInterrupt):
            print("\n[Main] Interrupted. Exiting.")
            return False

    if not confirm_config():
        sys.exit(1)

    if not check_dependencies():
        sys.exit(1)

    # Note: For /query/web demo endpoint, use --timeout-keep-alive 180 to allow slow LLM pipeline
    # workers=4: each process handles ~8 concurrent requests at current latency,
    # combined capacity covers 30 concurrent with headroom.
    # On Linux/production: replace with gunicorn:
    #   gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 4 --timeout 180 -b 0.0.0.0:8000
    uvicorn.run(
        "app.main:app",  # Module path (run from rag-service/ directory)
        host=API_HOST,
        port=API_PORT,
        workers=4,
        timeout_keep_alive=180,  # 3 minutes for slow /query/web endpoint
        timeout_graceful_shutdown=5,  # Only wait 5 seconds for graceful shutdown
        access_log=False,  # Disabled: replaced by rag.access middleware with latency info
    )
