"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os

from .database import init_db
from .routers import datasets, documents, tasks, query, units, health, demo, graph

# Initialize database
init_db()

# Create FastAPI app
app = FastAPI(
    title="RAG Service",
    description="RAG service layer built on top of Zag framework",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/")
async def root():
    """Redirect to API docs."""
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT
    
    # Note: For /query/web demo endpoint, use --timeout-keep-alive 180 to allow slow LLM pipeline
    uvicorn.run(
        "app.main:app",  # Module path (run from rag-service/ directory)
        host=API_HOST, 
        port=API_PORT,
        # reload=True,  # Enable auto-reload for development
        timeout_keep_alive=180,  # 3 minutes for slow /query/web endpoint
        timeout_graceful_shutdown=5  # Only wait 5 seconds for graceful shutdown
    )
