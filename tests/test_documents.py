"""
Documents API Tests

Tests for document management including file upload, listing, retrieval, and deletion.
"""
import pytest
import requests
import os
from pathlib import Path
from typing import Dict, Any





class TestDocumentManagement:
    """Test cases for document management operations."""
    
    @pytest.fixture
    def dataset_id(self):
        """Fixture that creates a dataset for document testing."""
        response = pytest.requests_session.post(
            f"{pytest.BASE_URL}/datasets",
            json={"name": "test_dataset_for_docs"}
        )
        dataset_id = response.json()["data"]["dataset_id"]
        yield dataset_id
        # Cleanup
        try:
            pytest.requests_session.delete(f"{pytest.BASE_URL}/datasets/{dataset_id}")
        except:
            pass
    
    @pytest.fixture
    def sample_pdf_file(self, tmp_path):
        """Fixture that creates a sample PDF file for testing."""
        # Create a simple text file as a placeholder
        # In real tests, you'd use an actual PDF
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4\ntest content")
        return test_file
    
    def test_upload_document(self, dataset_id, sample_pdf_file):
        """Test uploading a document file."""
        with open(sample_pdf_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "doc_id" in data["data"]
        assert "task_id" in data["data"]
        
        # Verify a task was created
        task_id = data["data"]["task_id"]
        assert task_id is not None
    
    def test_upload_document_to_nonexistent_dataset(self, sample_pdf_file):
        """Test uploading a document to a dataset that doesn't exist."""
        with open(sample_pdf_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/nonexistent_dataset_id/documents",
                files=files
            )
        
        assert response.status_code == 404
    
    def test_list_documents_empty(self, dataset_id):
        """Test listing documents when dataset has no documents."""
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/datasets/{dataset_id}/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 0
    
    def test_list_documents_with_data(self, dataset_id, sample_pdf_file):
        """Test listing documents after uploading some."""
        # Upload a document
        with open(sample_pdf_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            upload_response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        doc_id = upload_response.json()["data"]["doc_id"]
        
        # List documents
        response = pytest.requests_session.get(f"{pytest.BASE_URL}/datasets/{dataset_id}/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1
        
        # Verify our document is in the list
        doc_ids = [doc["doc_id"] for doc in data["data"]]
        assert doc_id in doc_ids
    
    def test_list_documents_pagination(self, dataset_id, sample_pdf_file):
        """Test document listing with pagination parameters."""
        # Upload a document first
        with open(sample_pdf_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        
        # Test pagination
        response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
            params={"page": 1, "size": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
    
    def test_get_document(self, dataset_id, sample_pdf_file):
        """Test retrieving a specific document by ID."""
        # Upload a document
        with open(sample_pdf_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            upload_response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        doc_id = upload_response.json()["data"]["doc_id"]
        
        # Get document details
        response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents/{doc_id}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["doc_id"] == doc_id
        assert "status" in data["data"]
        assert "file_name" in data["data"]
        assert "created_at" in data["data"]
    
    def test_get_nonexistent_document(self, dataset_id):
        """Test retrieving a document that doesn't exist."""
        response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents/nonexistent_doc_id"
        )
        
        assert response.status_code == 404
    
    def test_delete_document(self, dataset_id, sample_pdf_file):
        """Test deleting (soft delete) a document."""
        # Upload a document
        with open(sample_pdf_file, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            upload_response = pytest.requests_session.post(
                f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                files=files
            )
        doc_id = upload_response.json()["data"]["doc_id"]
        
        # Delete document
        delete_response = pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents/{doc_id}"
        )
        
        assert delete_response.status_code == 200
        data = delete_response.json()
        assert data["success"] is True
        
        # Verify document status changed to DISABLED
        get_response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents/{doc_id}"
        )
        assert get_response.status_code == 200
        doc_data = get_response.json()["data"]
        # Note: Adjust this assertion based on your actual status value
        assert doc_data["status"] in ["DISABLED", "disabled"]
    
    def test_delete_nonexistent_document(self, dataset_id):
        """Test deleting a document that doesn't exist."""
        response = pytest.requests_session.delete(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents/nonexistent_doc_id"
        )
        
        assert response.status_code == 404
    
    def test_upload_multiple_documents(self, dataset_id, sample_pdf_file):
        """Test uploading multiple documents to the same dataset."""
        doc_ids = []
        
        # Upload 3 documents
        for i in range(3):
            with open(sample_pdf_file, "rb") as f:
                files = {"file": (f"test_{i}.pdf", f, "application/pdf")}
                response = pytest.requests_session.post(
                    f"{pytest.BASE_URL}/datasets/{dataset_id}/documents",
                    files=files
                )
                assert response.status_code == 200
                doc_ids.append(response.json()["data"]["doc_id"])
        
        # Verify all documents are listed
        list_response = pytest.requests_session.get(
            f"{pytest.BASE_URL}/datasets/{dataset_id}/documents"
        )
        listed_doc_ids = [doc["doc_id"] for doc in list_response.json()["data"]]
        
        for doc_id in doc_ids:
            assert doc_id in listed_doc_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
