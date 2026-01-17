@echo off
REM Start Chroma Server with data directory in tmp folder
REM Chroma is a Python-based vector database that runs via CLI

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DATA_DIR=%PROJECT_ROOT%\tmp\chroma_data
set CHROMA_EXE=%PROJECT_ROOT%\.venv\Scripts\chroma.exe

echo ========================================
echo Starting Chroma Server
echo ========================================
echo Data directory: %DATA_DIR%
echo Executable: %CHROMA_EXE%
echo Server: http://localhost:18000
echo.

REM Create data directory if not exists
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Check if chroma.exe exists
if not exist "%CHROMA_EXE%" (
    echo Error: chroma.exe not found at %CHROMA_EXE%
    echo Please install chromadb: pip install chromadb
    pause
    exit /b 1
)

REM Start Chroma server
REM --path: Data persistence directory
REM --host: Server host address
REM --port: Server port (using 18000 to avoid common port conflicts)
echo Starting Chroma server at http://localhost:18000
echo Press Ctrl+C to stop
echo.
"%CHROMA_EXE%" run --path "%DATA_DIR%" --host localhost --port 18000
