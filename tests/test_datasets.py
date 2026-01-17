"""
Dataset API Tests

Tests for dataset CRUD operations including creation, listing, retrieval,
update, and deletion of datasets.
"""
import pytest
import requests
from typing import Dict, Any, List


class TestDatasetCRUD:
    """Test cases for dataset CRUD operations."""

    @pytest.fixture
    def created_dataset_id(self):
        """Fixture that creates a dataset and returns its ID for testing."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_dataset_fixture"}
        )
        dataset_id = response.json()["data"]["dataset_id"]
        yield dataset_id
        # Cleanup
        try:
            pytest.requests_session.delete(
                f"{pytest.BASE_URL}/datasets/{dataset_id}")
        except:
            pass

    def test_create_dataset_minimal(self):
        """Test creating a dataset with minimal required fields."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_dataset_minimal"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "test_dataset_minimal"
        assert "dataset_id" in data["data"]
        assert "collection_name" in data["data"]

        # Cleanup
        dataset_id = data["data"]["dataset_id"]
        pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id}")

    def test_create_dataset_full_fields(self):
        """Test creating a dataset with all optional fields."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={
                "name": "test_dataset_full",
                "description": "A full test dataset",
                "config": {"embedding_model": "bge-large", "chunk_size": 512}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "test_dataset_full"
        assert data["data"]["description"] == "A full test dataset"
        assert data["data"]["config"]["embedding_model"] == "bge-large"

        # Cleanup
        dataset_id = data["data"]["dataset_id"]
        pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id}")

    def test_create_dataset_duplicate_name(self):
        """Test that creating datasets with duplicate names is allowed."""
        # Create first dataset
        response1 = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "duplicate_name_test"}
        )
        dataset_id1 = response1.json()["data"]["dataset_id"]

        # Create second dataset with same name (should be allowed)
        response2 = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "duplicate_name_test"}
        )

        assert response2.status_code == 200
        dataset_id2 = response2.json()["data"]["dataset_id"]

        # IDs should be different
        assert dataset_id1 != dataset_id2

        # Cleanup
        pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id1}")
        pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id2}")

    def test_list_datasets(self):
        """Test listing all datasets."""
        # Create a dataset first
        create_response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_list_datasets"}
        )
        dataset_id = create_response.json()["data"]["dataset_id"]

        # List datasets
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/datasets")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

        # Verify our dataset is in the list
        dataset_ids = [ds["dataset_id"] for ds in data["data"]]
        assert dataset_id in dataset_ids

        # Cleanup
        pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id}")

    def test_get_dataset(self, created_dataset_id):
        """Test retrieving a specific dataset by ID."""
        response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{created_dataset_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["dataset_id"] == created_dataset_id
        assert "name" in data["data"]
        assert "collection_name" in data["data"]
        assert "created_at" in data["data"]

    def test_get_nonexistent_dataset(self):
        """Test retrieving a dataset that doesn't exist."""
        response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/nonexistent_id_12345")

        assert response.status_code == 404

    def test_update_dataset_name(self, created_dataset_id):
        """Test updating dataset name."""
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/datasets/{created_dataset_id}",
            json={"name": "updated_name"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "updated_name"

    def test_update_dataset_description(self, created_dataset_id):
        """Test updating dataset description."""
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/datasets/{created_dataset_id}",
            json={"description": "Updated description"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["description"] == "Updated description"

    def test_update_dataset_config(self, created_dataset_id):
        """Test updating dataset configuration."""
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/datasets/{created_dataset_id}",
            json={"config": {"new_key": "new_value"}}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["config"]["new_key"] == "new_value"

    def test_update_dataset_multiple_fields(self, created_dataset_id):
        """Test updating multiple dataset fields at once."""
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/datasets/{created_dataset_id}",
            json={
                "name": "multi_update",
                "description": "Multiple fields updated",
                "config": {"multi": True}
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "multi_update"
        assert data["data"]["description"] == "Multiple fields updated"
        assert data["data"]["config"]["multi"] is True

    def test_update_nonexistent_dataset(self):
        """Test updating a dataset that doesn't exist."""
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/datasets/nonexistent_id_12345",
            json={"name": "should_fail"}
        )

        assert response.status_code == 404

    def test_delete_dataset(self):
        """Test deleting a dataset."""
        # Create dataset
        create_response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_delete"}
        )
        dataset_id = create_response.json()["data"]["dataset_id"]

        # Delete dataset
        delete_response = pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id}")

        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data["success"] is True

        # Verify deletion
        get_response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_dataset(self):
        """Test deleting a dataset that doesn't exist."""
        response = pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/nonexistent_id_12345")

        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
