# RAG Worker startup script for Windows
# PowerShell version

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  RAG Worker Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Endpoint: http://localhost:8000" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan

# Get script directory and navigate to rag-service root
$scriptDir = $PSScriptRoot
$ragServiceDir = Join-Path $scriptDir ".."
Set-Location $ragServiceDir
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
Write-Host "[3/3] Starting worker (python -m worker.main)..." -ForegroundColor Green
Write-Host "`n========================================`n" -ForegroundColor Cyan

$proc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "worker.main" `
    -NoNewWindow `
    -PassThru

Write-Host "PID: $($proc.Id)  |  Ctrl+C to stop" -ForegroundColor DarkGray

try {
    while (-not $proc.HasExited) {
        Start-Sleep -Milliseconds 300
    }
} finally {
    if (-not $proc.HasExited) {
        Write-Host "`nShutting down (PID $($proc.Id))..." -ForegroundColor Yellow
        taskkill /T /F /PID $proc.Id 2>&1 | Out-Null
        Write-Host "Worker stopped." -ForegroundColor Green
    }
}
