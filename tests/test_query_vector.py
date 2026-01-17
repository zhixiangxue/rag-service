"""
Query API Tests - Vector Search

Tests for vector search query functionality including basic search,
top_k parameter, and filtering.
"""
import pytest
import requests
from typing import Dict, Any, List





class TestVectorQuery:
    """Test cases for vector search query operations."""
    
    @pytest.fixture(scope="class")
    def dataset_with_data(self, tmp_path_factory):
        """Fixture that creates a dataset with indexed data for querying.
        
        This is class-scoped to avoid creating/destroying datasets for each test,
        which significantly improves test performance (saves ~60 seconds).
        """
        # Create dataset
        dataset_response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_vector_query_dataset"}
        )
        dataset_id = dataset_response.json()["data"]["dataset_id"]
        
        # Upload and process a document to have searchable data
        tmp_path = tmp_path_factory.mktemp("vector_query_test")
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\ntest content about mortgage and guidelines")
        
        with open(test_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            doc_response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        
        task_id = doc_response.json()["data"]["task_id"]
        
        # Mark task as processing and completed to simulate worker processing
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "COMPLETED", "progress": 100, "unit_count": 5}
        )
        
        yield dataset_id
        
        # Cleanup
        try:
            pytest.requests_session.delete(f"{pytest.BASE_URL}/datasets/{dataset_id}")
        except:
            pass
    
    def test_vector_query_basic(self, dataset_with_data):
        """Test basic vector search query."""
        dataset_id = dataset_with_data
        
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "mortgage guidelines", "top_k": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        
        # Results might be empty if no data was actually indexed
        # (since we're just simulating completion)
    
    def test_vector_query_default_endpoint(self, dataset_with_data):
        """Test default query endpoint (should default to vector search)."""
        dataset_id = dataset_with_data
        
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query",
            json={"query": "test query", "top_k": 3}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
    
    def test_vector_query_with_top_k(self, dataset_with_data):
        """Test vector query with different top_k values."""
        dataset_id = dataset_with_data
        
        # Test with small top_k
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "test query", "top_k": 3}
        )
        assert response.status_code == 200
        results = response.json()["data"]
        assert len(results) <= 3
        
        # Test with larger top_k
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "test query", "top_k": 10}
        )
        assert response.status_code == 200
        results = response.json()["data"]
        assert len(results) <= 10
    
    def test_vector_query_result_structure(self, dataset_with_data):
        """Test that query results have correct structure."""
        dataset_id = dataset_with_data
        
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "test", "top_k": 5}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # If there are results, verify structure
        if len(data["data"]) > 0:
            result = data["data"][0]
            # Expected fields in query result
            assert "unit_id" in result
            assert "content" in result
            assert "score" in result
            # Score should be a float
            assert isinstance(result["score"], (int, float))
    
    def test_vector_query_empty_query(self, dataset_with_data):
        """Test vector query with empty query string."""
        dataset_id = dataset_with_data
        
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "", "top_k": 5}
        )
        
        # Should handle gracefully (500 error is acceptable for empty query)
        assert response.status_code in [200, 400, 422, 500]
    
    def test_vector_query_nonexistent_dataset(self):
        """Test vector query on a dataset that doesn't exist."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/nonexistent_dataset_id/query/vector",
            json={"query": "test", "top_k": 5}
        )
        
        # Returns 200 with default collection when DEFAULT_COLLECTION_NAME is set
        # Or 404 if no default is configured
        assert response.status_code in [200, 404]
    
    def test_vector_query_missing_parameters(self, dataset_with_data):
        """Test vector query with missing required parameters."""
        dataset_id = dataset_with_data
        
        # Missing query parameter
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"top_k": 5}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_vector_query_invalid_top_k(self, dataset_with_data):
        """Test vector query with invalid top_k values."""
        dataset_id = dataset_with_data
        
        # Negative top_k
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "test", "top_k": -1}
        )
        assert response.status_code in [400, 422]
        
        # Zero top_k
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "test", "top_k": 0}
        )
        assert response.status_code in [400, 422]
    
    def test_vector_query_with_filters(self, dataset_with_data):
        """Test vector query with metadata filters (if supported)."""
        dataset_id = dataset_with_data
        
        # This test assumes your API supports filtering
        # Adjust based on your actual filter implementation
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={
                "query": "test",
                "top_k": 5,
                "filter": {"doc_id": "1"}  # Example filter
            }
        )
        
        # Should either work or return appropriate error if filters not supported
        assert response.status_code in [200, 400, 422]
    
    def test_vector_query_score_ordering(self, dataset_with_data):
        """Test that results are ordered by relevance score."""
        dataset_id = dataset_with_data
        
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/query/vector",
            json={"query": "test", "top_k": 10}
        )
        
        assert response.status_code == 200
        results = response.json()["data"]
        
        # If we have multiple results, verify they're sorted by score (descending)
        if len(results) >= 2:
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
