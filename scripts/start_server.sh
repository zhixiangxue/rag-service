#!/bin/bash
# RAG Service startup script for Linux/Mac

echo ""
echo "========================================"
echo "  RAG Service Startup"
echo "========================================"
echo "API Server:  http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
echo "Gradio UI:   http://localhost:8000/ui"
echo "========================================"
echo ""

# Get script directory and navigate to rag-service root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAG_SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RAG_SERVICE_DIR"

# Validate rag-service directory structure
if [ ! -f "$RAG_SERVICE_DIR/app/main.py" ]; then
    echo ""
    echo "❌ ERROR: Invalid rag-service directory structure"
    echo "Expected: $RAG_SERVICE_DIR/app/main.py"
    echo ""
    echo "This script must be run from:"
    echo "  rag-service/scripts/start_server.sh"
    echo ""
    echo "Current location: $RAG_SERVICE_DIR"
    exit 1
fi

echo "[1/3] Working directory: $RAG_SERVICE_DIR"

# Function to find virtual environment
find_venv() {
    local start_path="$1"
    local venv_names=(".venv" "venv")
    local search_paths=(
        "$start_path"                    # Current directory (rag-service/)
        "$start_path/.."                 # Parent directory (zag-ai/)
        "$start_path/../.."              # Grandparent directory
    )
    
    for path in "${search_paths[@]}"; do
        for venv_name in "${venv_names[@]}"; do
            venv_activate="$path/$venv_name/bin/activate"
            if [ -f "$venv_activate" ]; then
                echo "$venv_activate"
                return 0
            fi
        done
    done
    
    return 1
}

# Find and activate virtual environment
VENV_PATH=$(find_venv "$RAG_SERVICE_DIR")

if [ -n "$VENV_PATH" ]; then
    echo "[2/3] Virtual environment: $VENV_PATH"
    source "$VENV_PATH"
else
    echo ""
    echo "❌ ERROR: Virtual environment not found"
    echo "Searched in:"
    echo "  - $RAG_SERVICE_DIR/.venv"
    echo "  - $RAG_SERVICE_DIR/venv"
    echo "  - $RAG_SERVICE_DIR/../.venv"
    echo "  - $RAG_SERVICE_DIR/../venv"
    echo ""
    echo "Please create virtual environment first:"
    echo "  python -m venv .venv"
    echo ""
    exit 1
fi

# Run as module from rag-service directory
echo "[3/3] Starting server (python -m app.main)..."
echo ""
echo "========================================"
echo ""
python -m app.main
