# Start Qdrant Server with data directory in tmp folder
# Qdrant is a vector search engine with REST API and gRPC support

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$DataDir = Join-Path $ProjectRoot "tmp\qdrant_data"
$QdrantExe = Join-Path $ScriptDir "qdrant.exe"
$QdrantConfig = Join-Path $ScriptDir "qdrant_config.yaml"

Write-Host "========================================"
Write-Host "Starting Qdrant Server"
Write-Host "========================================"
Write-Host "Data directory: $DataDir"
Write-Host "Executable:     $QdrantExe"
Write-Host "Config file:    $QdrantConfig"
Write-Host "REST API:       http://localhost:16333"
Write-Host "gRPC API:       http://localhost:16334"
Write-Host "Web Dashboard:  http://localhost:16333/dashboard"
Write-Host ""

# Create data directory if not exists
if (-not (Test-Path $DataDir)) {
    New-Item -ItemType Directory -Path $DataDir | Out-Null
}

# Check if qdrant.exe exists
if (-not (Test-Path $QdrantExe)) {
    Write-Host "Error: qdrant.exe not found at $QdrantExe" -ForegroundColor Red
    Write-Host "Please download from: https://github.com/qdrant/qdrant/releases"
    exit 1
}

# Start Qdrant server
# Custom ports (using 16333/16334 to avoid common port conflicts):
#   - 16333: HTTP REST API
#   - 16334: gRPC API
Write-Host "Starting Qdrant server..."
Write-Host "Press Ctrl+C to stop"
Write-Host ""

Set-Location $DataDir
& $QdrantExe --config-path $QdrantConfig
