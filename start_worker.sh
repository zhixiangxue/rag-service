#!/bin/bash
# RAG Worker startup script

echo "Starting RAG Worker..."
echo "Worker will poll API at: $API_BASE_URL (default: http://localhost:8000)"
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PARENT_DIR"
export PYTHONPATH="$PARENT_DIR:$PYTHONPATH"
python -m rag-service.worker.main
