@echo off
REM Start Qdrant Server with data directory in tmp folder
REM Qdrant is a vector search engine with REST API and gRPC support

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DATA_DIR=%PROJECT_ROOT%\tmp\qdrant_data
set QDRANT_EXE=%SCRIPT_DIR%qdrant.exe
set QDRANT_CONFIG=%SCRIPT_DIR%qdrant_config.yaml

echo ========================================
echo Starting Qdrant Server
echo ========================================
echo Data directory: %DATA_DIR%
echo Executable: %QDRANT_EXE%
echo Config file: %QDRANT_CONFIG%
echo REST API: http://localhost:16333
echo gRPC API: http://localhost:16334
echo Web Dashboard: http://localhost:16333/dashboard
echo.

REM Create data directory if not exists
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Check if qdrant.exe exists
if not exist "%QDRANT_EXE%" (
    echo Error: qdrant.exe not found at %QDRANT_EXE%
    echo Please download from: https://github.com/qdrant/qdrant/releases
    pause
    exit /b 1
)

REM Start Qdrant server
REM Qdrant will use the config file for port settings
REM Custom ports (using 16333/16334 to avoid common port conflicts):
REM   - 16333: HTTP REST API (used by Python client)
REM   - 16334: gRPC API (optional, for high-performance scenarios)
echo Starting Qdrant server...
echo Press Ctrl+C to stop
echo.
cd /d "%DATA_DIR%"
"%QDRANT_EXE%" --config-path "%QDRANT_CONFIG%"
