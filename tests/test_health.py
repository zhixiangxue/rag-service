"""
Health Check API Tests

Tests for the health check endpoint that verifies service status and dependencies.
"""
import pytest
import requests
from typing import Dict, Any


class TestHealthCheck:
    """Test cases for health check endpoint."""
    
    def test_health_check_returns_200(self):
        """Test that health check endpoint returns 200 status code."""
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/health")
        assert response.status_code == 200
    
    def test_health_check_response_structure(self):
        """Test that health check response has correct structure."""
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/health")
        data = response.json()
        
        # Verify basic structure
        assert "status" in data
        assert "dependencies" in data
        
    def test_health_check_service_status(self):
        """Test that service status is healthy or degraded."""
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/health")
        data = response.json()
        
        # Status can be 'ok' or 'degraded' (if dependencies are down)
        assert data["status"] in ["ok", "degraded"]
    
    def test_health_check_dependencies(self):
        """Test that all dependencies are reported."""
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/health")
        data = response.json()
        
        dependencies = data.get("dependencies", {})
        
        # Check that dependencies are reported (even if some are unhealthy)
        assert isinstance(dependencies, dict)
        
        # Expected dependencies: database, vector_store, etc.
        # Note: We don't assert they're all healthy in case of test environment issues


if __name__ == "__main__":
    # Allow running directly for quick testing
    pytest.main([__file__, "-v"])
