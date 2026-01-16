"""Simple e2e test for Document + Task API.

Usage:
    1. Make sure server is running: uvicorn app.main:app --reload --port 8000
    2. Run this script: python test_document_task_api.py
"""
import requests
import os

from app.constants import TaskStatus, DocumentStatus

BASE_URL = "http://localhost:8000"
TEST_FILE = "/Users/zhixiang.xue/zeitro/zag-ai/rag-service/tmp/usda.pdf"


def test_document_task_api():
    """Test Document and Task operations."""
    print("=" * 50)
    print("Testing Document + Task API")
    print("=" * 50)
    
    # 0. Create a dataset first
    print("\n0. Creating dataset...")
    response = requests.post(
        f"{BASE_URL}/datasets",
        json={"name": "test_dataset_for_docs"}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    assert response.status_code == 200
    dataset_id = response.json()["data"]["dataset_id"]
    print(f"✓ Created dataset: {dataset_id}")
    
    # 1. Ingest file
    print("\n1. Ingesting file...")
    if not os.path.exists(TEST_FILE):
        print(f"✗ Test file not found: {TEST_FILE}")
        exit(1)
    
    with open(TEST_FILE, "rb") as f:
        files = {"file": ("usda.pdf", f, "application/pdf")}
        response = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/documents/ingest",
            files=files
        )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    doc_id = data["data"]["doc_id"]
    task_id = data["data"]["task_id"]
    print(f"✓ File ingested, doc_id: {doc_id}, task_id: {task_id}")
    
    # 2. Get pending tasks
    print("\n2. Getting pending tasks...")
    response = requests.get(f"{BASE_URL}/tasks/pending")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) >= 1
    print(f"✓ Found {len(data['data'])} pending task(s)")
    
    # 3. Get task status
    print("\n3. Getting task status...")
    response = requests.get(f"{BASE_URL}/tasks/{task_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == TaskStatus.PENDING
    print(f"✓ Task status: {data['data']['status']}")
    
    # 4. Update task status to PROCESSING
    print("\n4. Updating task to PROCESSING...")
    response = requests.patch(
        f"{BASE_URL}/tasks/{task_id}/status",
        json={"status": TaskStatus.PROCESSING, "progress": 50}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == TaskStatus.PROCESSING
    assert data["data"]["progress"] == 50
    print(f"✓ Task updated: {data['data']['status']}, progress: {data['data']['progress']}")
    
    # 5. List documents
    print("\n5. Listing documents...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/documents")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) >= 1
    print(f"✓ Found {len(data['data'])} document(s)")
    
    # 6. Get document details
    print("\n6. Getting document details...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["doc_id"] == doc_id
    assert data["data"]["status"] == DocumentStatus.PROCESSING
    print(f"✓ Document status: {data['data']['status']}")
    
    # 7. Update task to COMPLETED
    print("\n7. Completing task...")
    response = requests.patch(
        f"{BASE_URL}/tasks/{task_id}/status",
        json={"status": TaskStatus.COMPLETED, "progress": 100, "unit_count": 5}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == TaskStatus.COMPLETED
    print(f"✓ Task completed")
    
    # 8. Verify document status changed to COMPLETED
    print("\n8. Verifying document status...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == DocumentStatus.COMPLETED
    assert data["data"]["unit_count"] == 5
    print(f"✓ Document status: {data['data']['status']}, unit_count: {data['data']['unit_count']}")
    
    # 9. Delete document (soft delete)
    print("\n9. Deleting document...")
    response = requests.delete(f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print("✓ Document disabled")
    
    # 10. Verify document is DISABLED
    print("\n10. Verifying document is disabled...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["data"]["status"] == DocumentStatus.DISABLED
    print(f"✓ Document status: {data['data']['status']}")
    
    # Cleanup: delete dataset
    print("\n11. Cleaning up...")
    requests.delete(f"{BASE_URL}/datasets/{dataset_id}")
    print("✓ Cleanup done")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    try:
        test_document_task_api()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to server. Make sure the server is running:")
        print("  uvicorn app.main:app --reload --port 8000")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
