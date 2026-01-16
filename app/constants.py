"""Constants and enums for RAG service."""
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DocumentStatus(str, Enum):
    """Document status enum."""
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"
