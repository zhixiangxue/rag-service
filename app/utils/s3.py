"""
S3 utilities for file operations
"""

import os
import asyncio
import boto3
from pathlib import Path
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional


def download_file_from_s3(
    s3_url: str,
    local_path: Path | str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None
) -> bool:
    """
    Download a file from S3 URL to local path (synchronous)
    
    Args:
        s3_url: S3 URL (e.g., s3://bucket-name/path/to/file.pdf)
        local_path: Local file path to save
        aws_access_key_id: AWS access key ID (defaults to env variable)
        aws_secret_access_key: AWS secret access key (defaults to env variable)
        
    Returns:
        True if successful, False otherwise
        
    Raises:
        ValueError: If S3 URL format is invalid
        NoCredentialsError: If AWS credentials are not found
        ClientError: If S3 operation fails
        
    Example:
        >>> from pathlib import Path
        >>> download_file_from_s3(
        ...     "s3://my-bucket/docs/file.pdf",
        ...     Path("local/file.pdf")
        ... )
        True
    """
    local_path = Path(local_path)
    
    # Parse S3 URL
    if not s3_url.startswith('s3://'):
        raise ValueError(f"Invalid S3 URL format: {s3_url}")
    
    # Extract bucket and key from s3://bucket-name/path/to/file
    url_parts = s3_url[5:].split('/', 1)
    if len(url_parts) != 2:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")
    
    bucket_name = url_parts[0]
    object_key = url_parts[1]
    
    # Get AWS credentials
    access_key = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = aws_secret_access_key or os.getenv('AWS_SECRET_KEY')
    
    if not access_key or not secret_key:
        raise NoCredentialsError()
    
    # Create parent directories if needed
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize S3 client (same as download_s3_files.py)
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )
    
    # Download file
    try:
        s3_client.download_file(bucket_name, object_key, str(local_path))
        return True
    except ClientError as e:
        # Raise original error without wrapping to avoid message duplication
        raise


def get_s3_object_info(
    s3_url: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None
) -> dict:
    """
    Get S3 object metadata (size, content_type) without downloading.
    Uses a HEAD request (head_object) — fast and cheap.

    Args:
        s3_url: S3 URL (e.g., s3://bucket-name/path/to/file.pdf)
        aws_access_key_id: AWS access key ID (defaults to env variable)
        aws_secret_access_key: AWS secret access key (defaults to env variable)

    Returns:
        dict with 'size' (int) and 'content_type' (str)

    Raises:
        ValueError: If S3 URL format is invalid
        FileNotFoundError: If the object does not exist in S3
        NoCredentialsError: If AWS credentials are not found
        ClientError: If another S3 operation error occurs
    """
    if not s3_url.startswith('s3://'):
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    url_parts = s3_url[5:].split('/', 1)
    if len(url_parts) != 2:
        raise ValueError(f"Invalid S3 URL format: {s3_url}")

    bucket_name = url_parts[0]
    object_key = url_parts[1]

    access_key = aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = aws_secret_access_key or os.getenv('AWS_SECRET_KEY')

    if not access_key or not secret_key:
        raise NoCredentialsError()

    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        return {
            'size': response['ContentLength'],
            'content_type': response.get('ContentType', 'application/octet-stream'),
        }
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ('404', 'NoSuchKey'):
            raise FileNotFoundError(f"Object not found in S3: {s3_url}")
        raise


async def download_file_from_s3_async(
    s3_url: str,
    local_path: Path | str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None
) -> bool:
    """
    Download a file from S3 URL to local path (asynchronous, non-blocking)
    
    This function runs the synchronous boto3 download in a thread pool
    to avoid blocking the event loop.
    
    Args:
        s3_url: S3 URL (e.g., s3://bucket-name/path/to/file.pdf)
        local_path: Local file path to save
        aws_access_key_id: AWS access key ID (defaults to env variable)
        aws_secret_access_key: AWS secret access key (defaults to env variable)
        
    Returns:
        True if successful
        
    Raises:
        ValueError: If S3 URL format is invalid
        NoCredentialsError: If AWS credentials are not found
        ClientError: If S3 operation fails
        
    Example:
        >>> await download_file_from_s3_async(
        ...     "s3://my-bucket/docs/file.pdf",
        ...     Path("local/file.pdf")
        ... )
        True
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        download_file_from_s3,
        s3_url,
        local_path,
        aws_access_key_id,
        aws_secret_access_key
    )
