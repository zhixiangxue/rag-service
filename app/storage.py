"""File storage abstraction layer."""
import os
import shutil
import random
import string
from abc import ABC, abstractmethod
from typing import BinaryIO
from pathlib import Path

from . import config


def generate_random_folder() -> str:
    """Generate random 6-character folder name."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))


class FileStorage(ABC):
    """Abstract file storage interface."""
    
    @abstractmethod
    def save(self, file_content: BinaryIO, filename: str, dataset_id: str) -> str:
        """Save file and return file path/url."""
        pass
    
    @abstractmethod
    def delete(self, file_path: str) -> None:
        """Delete file."""
        pass
    
    @abstractmethod
    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        pass


class LocalFileStorage(FileStorage):
    """Local filesystem storage."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
    
    def save(self, file_content: BinaryIO, filename: str, dataset_id: str) -> str:
        """Save file to local filesystem.
        
        Args:
            file_content: File content to save
            filename: Original filename
            dataset_id: Dataset ID for organizing files
            
        Returns:
            File path in POSIX format (forward slashes) for cross-platform compatibility
        """
        # Use Path for cross-platform compatibility
        dataset_dir = Path(self.base_dir) / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create random 6-char folder for this file
        random_folder = generate_random_folder()
        file_dir = dataset_dir / random_folder
        
        # Ensure uniqueness
        while file_dir.exists():
            random_folder = generate_random_folder()
            file_dir = dataset_dir / random_folder
        
        file_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = file_dir / filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file_content, f)
        
        # Return POSIX path string for cross-platform compatibility
        return file_path.as_posix()
    
    def delete(self, file_path: str) -> None:
        """Delete file from local filesystem."""
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists."""
        return os.path.exists(file_path)


class S3FileStorage(FileStorage):
    """S3 storage (placeholder for future implementation)."""
    
    def __init__(self, bucket: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.region = region
        # TODO: Initialize S3 client
        raise NotImplementedError("S3 storage not implemented yet")
    
    def save(self, file_content: BinaryIO, filename: str, dataset_id: str) -> str:
        """Save file to S3."""
        # TODO: Implement S3 upload
        raise NotImplementedError("S3 storage not implemented yet")
    
    def delete(self, file_path: str) -> None:
        """Delete file from S3."""
        # TODO: Implement S3 delete
        raise NotImplementedError("S3 storage not implemented yet")
    
    def exists(self, file_path: str) -> bool:
        """Check if file exists in S3."""
        # TODO: Implement S3 exists check
        raise NotImplementedError("S3 storage not implemented yet")


def get_storage() -> FileStorage:
    """Get file storage instance based on config."""
    storage_type = config.STORAGE_TYPE if hasattr(config, 'STORAGE_TYPE') else 'local'
    
    if storage_type == 'local':
        return LocalFileStorage(config.UPLOAD_DIR)
    elif storage_type == 's3':
        # TODO: Get S3 config from environment
        bucket = config.S3_BUCKET
        region = config.S3_REGION
        return S3FileStorage(bucket, region)
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
