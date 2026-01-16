"""Simple e2e test for Dataset API.

Usage:
    1. Start the server: uvicorn app.main:app --reload --port 8000
    2. Run this script: python test_dataset_api.py
"""
import requests

BASE_URL = "http://localhost:8000"


def test_dataset_api():
    """Test Dataset CRUD operations."""
    print("=" * 50)
    print("Testing Dataset API")
    print("=" * 50)
    
    # 1. Create dataset
    print("\n1. Creating dataset...")
    response = requests.post(
        f"{BASE_URL}/datasets",
        json={
            "name": "test_dataset",
            "description": "A test dataset",
            "config": {"key": "value"}
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["name"] == "test_dataset"
    dataset_id = data["data"]["dataset_id"]
    print(f"✓ Created dataset: {dataset_id}")
    
    # 2. List datasets
    print("\n2. Listing datasets...")
    response = requests.get(f"{BASE_URL}/datasets")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) >= 1
    print(f"✓ Found {len(data['data'])} dataset(s)")
    
    # 3. Get dataset
    print("\n3. Getting dataset...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["dataset_id"] == dataset_id
    print(f"✓ Retrieved dataset: {data['data']['name']}")
    
    # 4. Update dataset
    print("\n4. Updating dataset...")
    response = requests.patch(
        f"{BASE_URL}/datasets/{dataset_id}",
        json={"name": "updated_name", "description": "updated"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["data"]["name"] == "updated_name"
    print(f"✓ Updated dataset name to: {data['data']['name']}")
    
    # 5. Delete dataset
    print("\n5. Deleting dataset...")
    response = requests.delete(f"{BASE_URL}/datasets/{dataset_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    print("✓ Deleted dataset")
    
    # 6. Verify deletion
    print("\n6. Verifying deletion...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}")
    assert response.status_code == 404
    print("✓ Dataset not found (as expected)")
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)


if __name__ == "__main__":
    try:
        test_dataset_api()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        exit(1)
    except requests.exceptions.ConnectionError:
        print("\n✗ Cannot connect to server. Make sure the server is running:")
        print("  uvicorn app.main:app --reload --port 8000")
        exit(1)
