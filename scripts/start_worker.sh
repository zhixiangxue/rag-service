#!/bin/bash
# RAG Worker startup script

echo "Starting RAG Worker..."
echo "Worker will poll API at: $API_BASE_URL (default: http://localhost:8000)"
echo ""

cd "$(dirname "$0")/.."
python -m worker.main
