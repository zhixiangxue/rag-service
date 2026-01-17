"""
Pytest configuration file

This file is automatically loaded by pytest and sets up the test environment.
"""
import sys
import os
from pathlib import Path

# Add the parent directory (rag-service) to Python path
# This allows importing from 'app', 'worker' modules
project_root = Path(__file__).parent.parent

# Insert at the beginning to ensure our 'app' takes precedence
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify we're importing the correct 'app' module
try:
    import app
    app_path = Path(app.__file__).parent
    expected_path = project_root / "app"
    
    if app_path.resolve() != expected_path.resolve():
        raise ImportError(
            f"Module conflict detected!\n"
            f"Expected 'app' from: {expected_path}\n"
            f"But got 'app' from: {app_path}\n"
            f"Please check for naming conflicts with installed packages."
        )
except ImportError as e:
    if "Module conflict detected" in str(e):
        raise
    # 'app' not imported yet, which is fine


# Test configuration
# Can be overridden by environment variables
def pytest_configure(config):
    """Pytest hook to set up test configuration."""
    # Set BASE_URL from environment or use default
    import pytest
    import requests
    pytest.BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
    
    # Create a shared requests.Session for all tests to reuse HTTP connections
    # This significantly improves test performance (from 2s to 5ms per request)
    pytest.requests_session = requests.Session()
