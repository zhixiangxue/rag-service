#!/bin/bash
# RAG Worker startup script for Linux/Mac

echo ""
echo "========================================"
echo "  RAG Worker Startup"
echo "========================================"
echo "API Endpoint: http://localhost:8000"
echo "========================================"
echo ""

# Get script directory and navigate to rag-service root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RAG_SERVICE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$RAG_SERVICE_DIR"
echo "[1/4] Working directory: $RAG_SERVICE_DIR"

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
    echo "[2/4] Virtual environment: $VENV_PATH"
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

# Change to project root (zag-ai/) to allow relative imports
PROJECT_ROOT="$(cd "$RAG_SERVICE_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "[3/4] Changed to project root: $PROJECT_ROOT"

# Run worker as module: python -m rag-service.worker.main
echo "[4/4] Starting worker (python -m rag-service.worker.main)..."
echo ""
echo "========================================"
echo ""
python -m rag-service.worker.main
