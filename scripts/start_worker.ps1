# RAG Worker startup script for Windows
# PowerShell version

# Get API base URL from environment variable or use default
$apiBaseUrl = if ($env:API_BASE_URL) { $env:API_BASE_URL } else { "http://localhost:8000" }

Write-Host "Starting RAG Worker..." -ForegroundColor Green
Write-Host "Worker will poll API at: $apiBaseUrl" -ForegroundColor Cyan
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

# Start worker
python -m worker.main
