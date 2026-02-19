"""Constants for worker module."""


class TaskStatus:
    """Task status constants."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProcessingMode:
    """Document processing mode constants."""
    CLASSIC = "classic"
    LOD = "lod"
    GRAPH = "graph"
