@echo off
REM Start Meilisearch with data directory in tmp folder

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set DATA_DIR=%PROJECT_ROOT%\tmp\meilisearch_data

echo ========================================
echo Starting Meilisearch
echo ========================================
echo Data directory: %DATA_DIR%
echo.

REM Create data directory if not exists
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Start Meilisearch
"%SCRIPT_DIR%meilisearch-windows-amd64.exe" --db-path "%DATA_DIR%" --http-addr "127.0.0.1:7700"
