"""Unified logging module for rag-service based on loguru.

A simple and easy-to-use logging system that can be imported and used anywhere.

Basic Usage:
    from app.utils import logger
    
    logger.info("API request received")
    logger.debug(f"Query parameters: {params}")
    logger.warning("Slow database query detected")
    logger.error("Failed to connect to vector store")

Advanced Usage:
    # Dynamic level adjustment at runtime
    from app.utils import logger, set_level
    
    set_level("DEBUG")  # Switch to debug mode
    logger.debug("This will now be visible")
    
    # Context binding for request tracking
    logger = logger.bind(request_id="abc-123")
    logger.info("Processing request...")  # Output: ... | abc-123 | Processing request...

Configuration via Environment Variables:
    # Set log level (default: INFO)
    export LOG_LEVEL=DEBUG
    
    # Enable file output (default: false)
    export LOG_TO_FILE=true
    export LOG_FILE=logs/rag-service.log
    
    # Example for development
    LOG_LEVEL=DEBUG LOG_TO_FILE=true python -m uvicorn app.main:app

Log Formats:
    - INFO/WARNING/ERROR: Simple format (time | level | message)
    - DEBUG: Detailed format (time | level | file:function:line | message)
    
File Logging Features:
    - Auto rotation: 10 MB per file
    - Retention: 30 days
    - Auto compression: Old logs are zipped
    - UTF-8 encoding support
"""

import os
import sys
from pathlib import Path

from loguru import logger as _logger

# Remove loguru's default handler
_logger.remove()

# ===== Configuration (read from environment variables) =====
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() == "true"
LOG_FILE = os.getenv("LOG_FILE", "logs/rag-service.log")

# ===== Log Formats =====
# Simple format: time | level | message
SIMPLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

# Detailed format: time | level | location | message
DETAILED_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# Choose format based on log level (DEBUG uses detailed, others use simple)
LOG_FORMAT = DETAILED_FORMAT if LOG_LEVEL == "DEBUG" else SIMPLE_FORMAT


# ===== Logger Setup =====
_logger_initialized = False

def _setup_logger():
    """Configure logger with lazy initialization (auto-executed on first use)."""
    global _logger_initialized
    if _logger_initialized:
        return
    
    # Read environment variables
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_to_file = os.getenv("LOG_TO_FILE", "false").lower() == "true"
    log_file = os.getenv("LOG_FILE", "logs/rag-service.log")
    log_format = DETAILED_FORMAT if log_level == "DEBUG" else SIMPLE_FORMAT
    
    # 1. Console output (always enabled)
    _logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # 2. File output (optional)
    if log_to_file:
        # Ensure log directory exists
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        _logger.add(
            log_file,
            format=DETAILED_FORMAT,  # Files always use detailed format
            level=log_level,
            rotation="10 MB",  # Max 10MB per file
            retention="30 days",  # Keep for 30 days
            compression="zip",  # Compress old logs
            backtrace=True,
            diagnose=True,
            encoding="utf-8"
        )
        _logger.info(f"File logging enabled: {log_file}")
    
    _logger_initialized = True


# Wrap logger for lazy initialization
class _LazyLogger:
    """Lazy initialization wrapper for logger."""
    
    def __getattr__(self, name):
        # Initialize logger on first access to any method
        _setup_logger()
        # Then return the actual logger's attribute
        return getattr(_logger, name)


# Export logger (users use this directly)
logger = _LazyLogger()

def set_level(level: str):
    """Dynamically set log level at runtime.
    
    Args:
        level: Log level (DEBUG/INFO/WARNING/ERROR)
    
    Example:
        from app.utils import set_level
        set_level("DEBUG")  # Switch to debug mode
    """
    level = level.upper()
    _logger.remove()  # Remove all handlers
    # Re-add console output
    _logger.add(
        sys.stdout,
        format=DETAILED_FORMAT if level == "DEBUG" else SIMPLE_FORMAT,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )

__all__ = ["logger", "set_level"]
