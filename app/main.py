"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import os
import gradio as gr

from .database import init_db
from .routers import datasets, documents, tasks, query, units, health, demo

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

# Register routers
app.include_router(health.router)
app.include_router(datasets.router)
app.include_router(documents.router)
app.include_router(tasks.router)
app.include_router(query.router)
app.include_router(units.router)
app.include_router(demo.router)

# Mount Gradio UI
try:
    from .ui.gradio_ui import gradio_app
    app = gr.mount_gradio_app(app, gradio_app, path="/ui")
    print("✅ Gradio UI mounted at /ui")
except ImportError as e:
    print(f"⚠️  Gradio UI not available: {e}")
except Exception as e:
    print(f"⚠️  Failed to mount Gradio UI: {e}")

# Mount static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """Redirect to Gradio UI."""
    return RedirectResponse(url="/ui")


if __name__ == "__main__":
    import uvicorn
    from .config import API_HOST, API_PORT
    
    # Note: For /query/web demo endpoint, use --timeout-keep-alive 180 to allow slow LLM pipeline
    # IMPORTANT: Must use full module path "rag-service.app.main:app" when running from zag-ai/ directory
    uvicorn.run(
        "rag-service.app.main:app",  # Full path (run from zag-ai/ for relative imports)
        host=API_HOST, 
        port=API_PORT,
        reload=True,  # Enable auto-reload for development
        timeout_keep_alive=180,  # 3 minutes for slow /query/web endpoint
        timeout_graceful_shutdown=5  # Only wait 5 seconds for graceful shutdown
    )
