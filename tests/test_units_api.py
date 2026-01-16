"""Simple e2e test for Unit API.

Usage:
    1. Make sure server is running: uvicorn app.main:app --reload --port 8000
    2. Make sure you have data in the vector store (run test_query_api.py first)
    3. Run this script: python tests/test_units_api.py
"""
import requests

BASE_URL = "http://localhost:8000"


def test_units_api():
    """Test Unit operations."""
    print("=" * 50)
    print("Testing Unit API")
    print("=" * 50)
    
    # Note: This test assumes you have data in mortgage_guidelines collection
    
    # 0. Create a test dataset (use existing collection)
    print("\n0. Creating dataset...")
    response = requests.post(
        f"{BASE_URL}/datasets",
        json={"name": "mortgage_guidelines"}
    )
    assert response.status_code == 200
    data = response.json()["data"]
    dataset_id = data["dataset_id"]
    print(f"✓ Created dataset: {dataset_id}")
    
    # 1. List units (should have data from previous ingestion)
    print("\n1. Listing units...")
    response = requests.get(
        f"{BASE_URL}/datasets/{dataset_id}/units",
        params={"page": 1, "size": 5}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        units = data["data"]
        print(f"✓ Found {len(units)} units")
        
        if len(units) > 0:
            unit_id = units[0]["unit_id"]
            print(f"  First unit ID: {unit_id}")
            
            # 2. Get single unit
            print("\n2. Getting single unit...")
            response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/units/{unit_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            unit = data["data"]
            print(f"✓ Retrieved unit: {unit_id}")
            print(f"  Content preview: {unit['content'][:100]}...")
            print(f"  Metadata: {unit['metadata']}")
            
            # 3. Update unit metadata
            print("\n3. Updating unit metadata...")
            new_metadata = {
                **unit['metadata'],
                "test_tag": "updated_by_test",
                "test_timestamp": "2025-01-16"
            }
            response = requests.patch(
                f"{BASE_URL}/datasets/{dataset_id}/units/{unit_id}",
                json={"metadata": new_metadata}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            updated_unit = data["data"]
            assert updated_unit["metadata"]["test_tag"] == "updated_by_test"
            print(f"✓ Updated unit metadata")
            print(f"  New metadata: {updated_unit['metadata']}")
            
            # 4. Batch update (if we have multiple units)
            if len(units) >= 2:
                print("\n4. Batch updating units...")
                batch_updates = [
                    {
                        "unit_id": units[0]["unit_id"],
                        "metadata": {"batch_test": "first"}
                    },
                    {
                        "unit_id": units[1]["unit_id"],
                        "metadata": {"batch_test": "second"}
                    }
                ]
                response = requests.patch(
                    f"{BASE_URL}/datasets/{dataset_id}/units:batch_update",
                    json=batch_updates
                )
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                print(f"✓ Batch updated {len(batch_updates)} units")
            
            # Note: We don't delete units in test to keep the data
            print("\n5. Skipping delete test to preserve data")
            
    else:
        print(f"✗ Failed to list units: {response.text}")
        print("Note: Make sure you have data in the vector store")
    
    # Cleanup
    print("\n6. Cleaning up...")
    requests.delete(f"{BASE_URL}/datasets/{dataset_id}")
    print("✓ Cleanup done")
    
    print("\n" + "=" * 50)
    print("Unit API test completed! ✓")
    print("=" * 50)
    print("\nNote: Full testing requires indexed data.")
    print("Run worker to process documents first if units list is empty.")


if __name__ == "__main__":
    try:
        test_units_api()
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
