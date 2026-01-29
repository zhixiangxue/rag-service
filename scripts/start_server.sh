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

# Run as module: python -m rag-service.app.main
echo "[4/4] Starting server (python -m rag-service.app.main)..."
echo ""

# Debug: Show database configuration (use actual config.py)
echo "========================================"
echo "  Database Configuration"
echo "========================================"
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from pathlib import Path

try:
    # Import actual config used by the server
    from rag-service.app.config import DATABASE_PATH
    
    abs_db_path = Path(DATABASE_PATH).resolve()
    
    print(f'DATABASE_PATH (config): {DATABASE_PATH}')
    print(f'DATABASE_PATH (abs):    {abs_db_path}')
    print(f'Parent dir exists:      {abs_db_path.parent.exists()}')
    print(f'DB file exists:         {abs_db_path.exists()}')
except Exception as e:
    import traceback
    print(f'Error loading config: {e}')
    traceback.print_exc()
"
echo "========================================"
echo ""
python -m rag-service.app.main
