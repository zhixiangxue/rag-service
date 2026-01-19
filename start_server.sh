#!/bin/bash
# RAG Service startup script

echo "Starting RAG Service..."
echo "API Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"
uvicorn app.main:app --host 0.0.0.0 --reload --port 8000 --timeout-keep-alive 180
