"""Constants and enums for RAG service."""
from enum import Enum


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    
    @staticmethod
    def is_valid_transition(from_status: str, to_status: str) -> bool:
        """Check if status transition is valid.
        
        Args:
            from_status: Current status
            to_status: Target status
            
        Returns:
            True if transition is allowed, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            TaskStatus.PENDING: [TaskStatus.PROCESSING],
            TaskStatus.PROCESSING: [TaskStatus.PROCESSING, TaskStatus.COMPLETED, TaskStatus.FAILED],  # Allow PROCESSING→PROCESSING for progress updates
            TaskStatus.COMPLETED: [],  # Terminal state
            TaskStatus.FAILED: []  # Terminal state
        }
        
        return to_status in valid_transitions.get(from_status, [])


class DocumentStatus(str, Enum):
    """Document status enum."""
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    DISABLED = "DISABLED"
    
    @staticmethod
    def is_valid_transition(from_status: str, to_status: str) -> bool:
        """Check if status transition is valid.
        
        Args:
            from_status: Current status
            to_status: Target status
            
        Returns:
            True if transition is allowed, False otherwise
        """
        # Define valid transitions
        valid_transitions = {
            DocumentStatus.PROCESSING: [DocumentStatus.COMPLETED, DocumentStatus.FAILED],
            DocumentStatus.COMPLETED: [DocumentStatus.DISABLED],  # Can be disabled
            DocumentStatus.FAILED: [DocumentStatus.DISABLED],  # Can be disabled
            DocumentStatus.DISABLED: []  # Terminal state
        }
        
        return to_status in valid_transitions.get(from_status, [])
