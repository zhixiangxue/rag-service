# RAG Worker startup script for Windows
# PowerShell version

Write-Host "Starting RAG Worker..." -ForegroundColor Green
Write-Host "Worker will poll API for tasks..." -ForegroundColor Cyan
Write-Host ""

# Get script directory and navigate to rag-service root
$scriptDir = $PSScriptRoot
$ragServiceDir = Join-Path $scriptDir ".."
Set-Location $ragServiceDir
Write-Host "Working directory: $ragServiceDir" -ForegroundColor Yellow

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
    Write-Host "Activating virtual environment: $venvPath" -ForegroundColor Green
    & $venvPath
} else {
    Write-Host "Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "Searched in:" -ForegroundColor Yellow
    Write-Host "  - $ragServiceDir\.venv" -ForegroundColor Yellow
    Write-Host "  - $ragServiceDir\venv" -ForegroundColor Yellow
    Write-Host "  - $((Join-Path $ragServiceDir ".."))\.venv" -ForegroundColor Yellow
    Write-Host "  - $((Join-Path $ragServiceDir ".."))\venv" -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Yellow
    Write-Host "Please create virtual environment first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# Start worker (working directory must be zag-ai/ for relative imports)
Write-Host "Starting worker..." -ForegroundColor Green

# Change to project root (zag-ai/) to allow relative imports
$projectRoot = Join-Path $ragServiceDir ".."
Set-Location $projectRoot
Write-Host "Changed to project root: $projectRoot" -ForegroundColor Yellow

# Run as module: python -m rag-service.worker.worker
python -m rag-service.worker.worker
