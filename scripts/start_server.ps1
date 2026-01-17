# RAG Service startup script for Windows
# PowerShell version

Write-Host "Starting RAG Service..." -ForegroundColor Green
Write-Host "API Server: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Change to project root directory
Set-Location (Join-Path $PSScriptRoot "..")

# Activate virtual environment if it exists
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment: $venvPath" -ForegroundColor Yellow
    & $venvPath
} else {
    Write-Host "Warning: No virtual environment found at $venvPath. Using system Python." -ForegroundColor Yellow
}

# Start uvicorn server
python -m uvicorn app.main:app --reload --port 8000 --timeout-keep-alive 180
