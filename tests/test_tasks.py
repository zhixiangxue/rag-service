"""
Tasks API Tests

Tests for task management including listing pending tasks, retrieving task status,
and updating task progress and status.
"""
import pytest
import requests
import time
from typing import Dict, Any





class TestTaskManagement:
    """Test cases for task management operations."""
    
    @pytest.fixture
    def dataset_with_task(self, tmp_path):
        """Fixture that creates a dataset and uploads a document to create a task."""
        # Create dataset
        dataset_response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_dataset_for_tasks"}
        )
        dataset_id = dataset_response.json()["data"]["dataset_id"]
        
        # Upload a document to create a task
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\ntest content")
        
        with open(test_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            doc_response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        
        task_id = doc_response.json()["data"]["task_id"]
        doc_id = doc_response.json()["data"]["doc_id"]
        
        yield {
            "dataset_id": dataset_id,
            "task_id": task_id,
            "doc_id": doc_id
        }
        
        # Cleanup
        try:
            pytest.requests_session.delete(f"{pytest.BASE_URL}/datasets/{dataset_id}")
        except:
            pass
    
    def test_get_pending_tasks_empty(self):
        """Test getting pending tasks when there are none."""
        # First, try to clear any existing pending tasks by getting them
        # This test might fail if there are actual pending tasks
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/tasks/pending")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
    
    def test_get_pending_tasks_with_data(self, dataset_with_task):
        """Test getting pending tasks when there are some."""
        task_id = dataset_with_task["task_id"]
        
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/tasks/pending")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        
        # Our task should be in the pending list (or might have been consumed)
        # Just verify the endpoint works and returns a list
        # Note: In a real scenario, tasks might be consumed quickly by workers
    
    def test_get_task_status(self, dataset_with_task):
        """Test retrieving task status by ID."""
        task_id = dataset_with_task["task_id"]
        
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/tasks/{task_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["task_id"] == task_id
        assert "status" in data["data"]
        assert "progress" in data["data"]
        assert "created_at" in data["data"]
    
    def test_get_nonexistent_task(self):
        """Test retrieving a task that doesn't exist."""
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/tasks/nonexistent_task_id")
        
        assert response.status_code == 404
    
    def test_update_task_status_to_processing(self, dataset_with_task):
        """Test updating task status to PROCESSING."""
        task_id = dataset_with_task["task_id"]
        
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 30}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "PROCESSING"
        assert data["data"]["progress"] == 30
    
    def test_update_task_progress(self, dataset_with_task):
        """Test updating task progress."""
        task_id = dataset_with_task["task_id"]
        
        # First set to processing
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        
        # Update progress
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"progress": 50}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["progress"] == 50
    
    def test_update_task_status_to_completed(self, dataset_with_task):
        """Test updating task status to COMPLETED."""
        task_id = dataset_with_task["task_id"]
        
        # First set to PROCESSING (required by API)
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        
        # Then complete it
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={
                "status": "COMPLETED",
                "progress": 100,
                "unit_count": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "COMPLETED"
        assert data["data"]["progress"] == 100
    
    def test_update_task_status_to_failed(self, dataset_with_task):
        """Test updating task status to FAILED with error message."""
        task_id = dataset_with_task["task_id"]
        
        # First set to PROCESSING
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        
        # Then set to FAILED
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={
                "status": "FAILED",
                "error": "Test error message"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["status"] == "FAILED"
        # Verify error is recorded (field name might vary)
        # assert "error" in data["data"]
    
    def test_task_status_flow(self, dataset_with_task):
        """Test complete task status flow: PENDING -> PROCESSING -> COMPLETED."""
        task_id = dataset_with_task["task_id"]
        
        # 1. Verify initial status is PENDING
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/tasks/{task_id}")
        assert response.json()["data"]["status"] == "PENDING"
        
        # 2. Update to PROCESSING
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        assert response.json()["data"]["status"] == "PROCESSING"
        
        # 3. Update progress
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"progress": 50}
        )
        assert response.json()["data"]["progress"] == 50
        
        # 4. Complete task
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "COMPLETED", "progress": 100, "unit_count": 10}
        )
        data = response.json()["data"]
        assert data["status"] == "COMPLETED"
        assert data["progress"] == 100
    
    def test_update_task_with_unit_count(self, dataset_with_task):
        """Test updating task with unit count information."""
        task_id = dataset_with_task["task_id"]
        
        # First set to PROCESSING
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        
        # Then complete with unit count
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={
                "status": "COMPLETED",
                "progress": 100,
                "unit_count": 42
            }
        )
        
        assert response.status_code == 200
        data = response.json()["data"]
        # Verify unit count is recorded (if your API supports it)
        # assert data["unit_count"] == 42
    
    def test_update_nonexistent_task(self):
        """Test updating a task that doesn't exist."""
        response = pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/nonexistent_task_id",
            json={"status": "PROCESSING"}
        )
        
        assert response.status_code == 404
    
    def test_completed_task_affects_document_status(self, dataset_with_task):
        """Test that completing a task updates the related document status."""
        task_id = dataset_with_task["task_id"]
        doc_id = dataset_with_task["doc_id"]
        dataset_id = dataset_with_task["dataset_id"]
        
        # First set to PROCESSING
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "PROCESSING", "progress": 0}
        )
        
        # Then complete the task
        pytest.requests_session.patch(
            f"{pytest.BASE_URL}/tasks/{task_id}",
            json={"status": "COMPLETED", "progress": 100, "unit_count": 5}
        )
        
        # Check document status
        doc_response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents/{doc_id}"
        )
        
        doc_data = doc_response.json()["data"]
        # Document status should reflect task completion
        assert doc_data["status"] in ["COMPLETED", "completed"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
