# RAG Service startup script for Windows
# PowerShell version

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  RAG Service Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Server:  http://localhost:8000" -ForegroundColor Green
Write-Host "API Docs:    http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Gradio UI:   http://localhost:8000/ui" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

# Get script directory and navigate to rag-service root
$scriptDir = $PSScriptRoot
$ragServiceDir = Join-Path $scriptDir ".."
Set-Location $ragServiceDir

# Validate rag-service directory structure
if (-not (Test-Path (Join-Path $ragServiceDir "app\main.py"))) {
    Write-Host "`n❌ ERROR: Invalid rag-service directory structure" -ForegroundColor Red
    Write-Host "Expected: $ragServiceDir\app\main.py" -ForegroundColor Yellow
    Write-Host "`nThis script must be run from:" -ForegroundColor Yellow
    Write-Host "  rag-service/scripts/start_server.ps1" -ForegroundColor Yellow
    Write-Host "`nCurrent location: $ragServiceDir" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/3] Working directory: $ragServiceDir" -ForegroundColor Yellow

# Function to find virtual environment
function Find-VirtualEnv {
    param (
        [string]$StartPath
    )
    
    $venvNames = @(".venv", "venv")
    $searchPaths = @(
        $StartPath,                          # Current directory (rag-service/)
        (Join-Path $StartPath ".."),         # Parent directory (zag-ai/)
        (Join-Path $StartPath "..\.")        # Grandparent directory
    )
    
    foreach ($path in $searchPaths) {
        foreach ($venvName in $venvNames) {
            $venvPath = Join-Path $path "$venvName\Scripts\Activate.ps1"
            if (Test-Path $venvPath) {
                return $venvPath
            }
        }
    }
    
    return $null
}

# Find and activate virtual environment
$venvPath = Find-VirtualEnv -StartPath $ragServiceDir

if ($venvPath) {
    Write-Host "[2/3] Virtual environment: $venvPath" -ForegroundColor Green
    & $venvPath
} else {
    Write-Host "`n❌ ERROR: Virtual environment not found" -ForegroundColor Red
    Write-Host "Searched in:" -ForegroundColor Yellow
    Write-Host "  - $ragServiceDir\.venv" -ForegroundColor Yellow
    Write-Host "  - $ragServiceDir\venv" -ForegroundColor Yellow
    Write-Host "  - $((Join-Path $ragServiceDir ".."))\.venv" -ForegroundColor Yellow
    Write-Host "  - $((Join-Path $ragServiceDir ".."))\venv" -ForegroundColor Yellow
    Write-Host "`nPlease create virtual environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv`n" -ForegroundColor Yellow
    exit 1
}

# Run as module from rag-service directory
Write-Host "[3/3] Starting server (python -m app.main)..." -ForegroundColor Green
Write-Host "`n========================================`n" -ForegroundColor Cyan
python -m app.main
