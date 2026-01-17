"""
Query API Tests - Fulltext Search

Tests for fulltext search query functionality.
NOTE: Fulltext search is not yet implemented (returns 501).
These tests verify the API structure and will be updated when implementation is complete.
"""
import pytest
import requests
from typing import Dict, Any, List





class TestFulltextQuery:
    """Test cases for fulltext search query operations."""
    
    @pytest.fixture
    def dataset_id(self):
        """Fixture that creates a dataset for testing."""
        dataset_response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_fulltext_query_dataset"}
        )
        dataset_id = dataset_response.json()["data"]["dataset_id"]
        yield dataset_id
        
        # Cleanup
        try:
            pytest.requests_session.delete(f"{pytest.BASE_URL}/datasets/{dataset_id}")
        except:
            pass
    
    def test_fulltext_endpoint_exists(self, dataset_id):
        """Test that fulltext endpoint exists and returns 501 (not implemented)."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/fulltext",
            json={"query": "test", "top_k": 5}
        )
        
        # Currently returns 501 Not Implemented
        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"].lower()
    
    def test_fulltext_nonexistent_dataset(self):
        """Test fulltext query on nonexistent dataset."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/nonexistent_dataset_id/query/fulltext",
            json={"query": "test", "top_k": 5}
        )
        
        # Should return 501 (not implemented) or 404 (dataset not found)
        # depending on which check happens first
        assert response.status_code in [404, 501]
    
    @pytest.mark.skip(reason="Fulltext search not yet implemented (501)")
    def test_fulltext_query_basic(self, dataset_id):
        """Test basic fulltext search query (skipped until implemented)."""
        pass
    
    @pytest.mark.skip(reason="Fulltext search not yet implemented (501)")
    def test_fulltext_query_with_top_k(self, dataset_id):
        """Test fulltext query with different top_k values (skipped until implemented)."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
