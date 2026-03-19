"""FastAPI application entry point."""
import sys
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from .database import init_db
from .routers import datasets, documents, tasks, query, units, health, demo, graph, utility
from .routers import dependencies
from . import config

# Logging configuration: timestamp + level + logger name + message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Enable debug-level sub-timing from query router with: LOG_LEVEL=DEBUG
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.getLogger("rag").setLevel(getattr(logging, _log_level, logging.INFO))

logger = logging.getLogger("rag.main")

# Initialize database
init_db()


# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    description="RAG service layer built on top of Zag framework",
    version="0.1.0",
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
app.include_router(datasets.router)
app.include_router(documents.router)
app.include_router(tasks.router)
app.include_router(query.router)
app.include_router(units.router)
app.include_router(demo.router)
app.include_router(graph.router)
app.include_router(dependencies.router)
app.include_router(utility.router)


@app.get("/")
async def root():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


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
        print(f"  DATABASE_PATH: {os.path.abspath(config.DATABASE_PATH)}")
        print(f"\n[Cache]")
        _archives_dir = os.path.abspath(config.ARCHIVES_DIR)
        _pdf_dir = os.path.abspath(config.PDF_FILES_DIR)
        _archives_count = len(os.listdir(_archives_dir)) if os.path.isdir(_archives_dir) else 0
        _pdf_count = len(os.listdir(_pdf_dir)) if os.path.isdir(_pdf_dir) else 0
        print(f"  ARCHIVES_DIR: {_archives_dir} ({_archives_count})")
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
