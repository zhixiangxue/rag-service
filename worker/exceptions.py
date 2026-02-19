"""Custom exceptions for worker module."""
from typing import Any, Dict, Optional


class ProcessingErrorCode:
    """Standardized error codes for document processing."""
    
    # ===== Permanent Failures (should NOT retry) =====
    # LOD-specific unsuitable scenarios
    UNSUITABLE_FOR_LOD_COMPRESSION_FAILED = "UNSUITABLE_FOR_LOD_COMPRESSION_FAILED"
    
    # File-level issues
    UNSUPPORTED_FILE_FORMAT = "UNSUPPORTED_FILE_FORMAT"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_FILE = "EMPTY_FILE"
    
    # ===== Transient Failures (should retry) =====
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE"
    NETWORK_ERROR = "NETWORK_ERROR"
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    MEILISEARCH_UNAVAILABLE = "MEILISEARCH_UNAVAILABLE"
    
    # ===== System Failures (need manual intervention) =====
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    GPU_ERROR = "GPU_ERROR"
    DISK_FULL = "DISK_FULL"
    
    # ===== Unknown Errors =====
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ProcessingErrorType:
    """Error type classification."""
    PERMANENT = "permanent"  # Should NOT retry, switch mode or skip
    TRANSIENT = "transient"  # Should retry (network, rate limit, etc.)
    SYSTEM = "system"  # Need manual intervention
    UNKNOWN = "unknown"  # Unexpected errors


# Error code to type mapping
ERROR_TYPE_MAPPING = {
    # Permanent
    ProcessingErrorCode.UNSUITABLE_FOR_LOD_COMPRESSION_FAILED: ProcessingErrorType.PERMANENT,
    ProcessingErrorCode.UNSUPPORTED_FILE_FORMAT: ProcessingErrorType.PERMANENT,
    ProcessingErrorCode.CORRUPTED_FILE: ProcessingErrorType.PERMANENT,
    ProcessingErrorCode.FILE_TOO_LARGE: ProcessingErrorType.PERMANENT,
    ProcessingErrorCode.EMPTY_FILE: ProcessingErrorType.PERMANENT,
    
    # Transient
    ProcessingErrorCode.LLM_RATE_LIMIT: ProcessingErrorType.TRANSIENT,
    ProcessingErrorCode.LLM_TIMEOUT: ProcessingErrorType.TRANSIENT,
    ProcessingErrorCode.LLM_SERVICE_UNAVAILABLE: ProcessingErrorType.TRANSIENT,
    ProcessingErrorCode.NETWORK_ERROR: ProcessingErrorType.TRANSIENT,
    ProcessingErrorCode.VECTOR_STORE_UNAVAILABLE: ProcessingErrorType.TRANSIENT,
    ProcessingErrorCode.MEILISEARCH_UNAVAILABLE: ProcessingErrorType.TRANSIENT,
    
    # System
    ProcessingErrorCode.OUT_OF_MEMORY: ProcessingErrorType.SYSTEM,
    ProcessingErrorCode.GPU_ERROR: ProcessingErrorType.SYSTEM,
    ProcessingErrorCode.DISK_FULL: ProcessingErrorType.SYSTEM,
}


def get_error_type(error_code: str) -> str:
    """Get error type from error code."""
    return ERROR_TYPE_MAPPING.get(error_code, ProcessingErrorType.UNKNOWN)


class ProcessingError(Exception):
    """Structured processing error with error code and details."""
    
    def __init__(
        self,
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        """
        Initialize processing error.
        
        Args:
            error_code: Error code from ProcessingErrorCode
            message: Human-readable error message
            details: Additional error details (for debugging/filtering)
            suggestion: Suggested action for user
        """
        super().__init__(message)
        self.error_code = error_code
        self.error_type = get_error_type(error_code)
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API error response."""
        result = {
            "error_code": self.error_code,
            "error_type": self.error_type,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result
    
    def __str__(self) -> str:
        """String representation for logging."""
        parts = [f"[{self.error_code}] {self.message}"]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)
