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
Write-Host "[1/4] Working directory: $ragServiceDir" -ForegroundColor Yellow

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
    Write-Host "[2/4] Virtual environment: $venvPath" -ForegroundColor Green
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

# Start server (working directory must be zag-ai/ for relative imports)
# Change to project root (zag-ai/) to allow relative imports
$projectRoot = Join-Path $ragServiceDir ".."
Set-Location $projectRoot
Write-Host "[3/4] Changed to project root: $projectRoot" -ForegroundColor Green

# Run as module: python -m rag-service.app.main
Write-Host "[4/4] Starting server (python -m rag-service.app.main)..." -ForegroundColor Green
Write-Host "`n========================================`n" -ForegroundColor Cyan
python -m rag-service.app.main
