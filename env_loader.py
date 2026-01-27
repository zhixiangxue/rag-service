"""Environment configuration loader.

Automatically loads the appropriate .env file based on availability.
Priority order: .env.local > .env.dev > .env.test > .env.prod

Usage:
    from env_loader import load_environment
    load_environment()
"""
import os
from pathlib import Path
from dotenv import load_dotenv


def load_environment():
    """Load environment variables from .env files.
    
    The function will search for .env files in the following order:
    1. .env.local (highest priority, for personal local config, not in git)
    2. .env.dev (development environment)
    3. .env.test (test environment)
    4. .env.prod (production environment)
    
    If ENV environment variable is set, it will try to load .env.{ENV} first.
    Otherwise, it will search for the first available file from the list above.
    
    Note: Searches for .env files in the rag-service/ directory (not cwd).
    
    Raises:
        FileNotFoundError: If no .env file is found.
    """
    # Get the directory where this script is located (rag-service/)
    script_dir = Path(__file__).parent
    
    # If ENV is explicitly set, try to use it first
    env = os.getenv('ENV')
    if env:
        env_file = script_dir / f'.env.{env}'
        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded environment from: {env_file}")
            return
        else:
            print(f"Warning: ENV={env} but {env_file} not found, falling back to auto-detect")
    
    # Auto-detect: search for env files in priority order
    env_files = [
        '.env.local',
        '.env.dev',
        '.env.test',
        '.env.prod',
    ]
    
    for env_file_name in env_files:
        env_file = script_dir / env_file_name
        if env_file.exists():
            load_dotenv(env_file)
            print(f"Loaded environment from: {env_file}")
            return
    
    # No env file found
    raise FileNotFoundError(
        f"No .env file found in {script_dir}. Please create one of: "
        f"{', '.join(env_files)}"
    )
