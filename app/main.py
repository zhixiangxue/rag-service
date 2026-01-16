"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .database import init_db
from .routers import datasets, documents, tasks, query, units, health

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
app.include_router(health.router)  # Health check (no prefix)
app.include_router(datasets.router)
app.include_router(documents.router)
app.include_router(tasks.router)
app.include_router(query.router)
app.include_router(units.router)


@app.get("/")
def root():
    return {"message": "RAG Service API", "version": "0.1.0"}
