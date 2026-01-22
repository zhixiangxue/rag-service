#!/bin/bash
# RAG Service startup script

echo "Starting RAG Service..."
echo "API Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PARENT_DIR"
export PYTHONPATH="$PARENT_DIR:$PYTHONPATH"
uvicorn rag-service.app.main:app --host 0.0.0.0 --reload --port 8000 --timeout-keep-alive 180
