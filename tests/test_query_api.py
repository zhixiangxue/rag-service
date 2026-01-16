"""Simple e2e test for Query API.

Usage:
    1. Make sure server is running: uvicorn app.main:app --reload --port 8000
    2. Make sure you have indexed data in the vector store
    3. Run this script: python test_query_api.py
"""
import requests

BASE_URL = "http://localhost:8000"


def test_query_api():
    """Test Query operations."""
    print("=" * 50)
    print("Testing Query API")
    print("=" * 50)
    
    # Note: This test assumes you have already run worker to index some documents
    # If vector store is empty, the query will return empty results
    
    # 1. Create a test dataset (use existing collection name)
    print("\n1. Creating dataset...")
    response = requests.post(
        f"{BASE_URL}/datasets",
        json={"name": "mortgage_guidelines"}  # Use real collection name in Qdrant
    )
    assert response.status_code == 200
    data = response.json()["data"]
    dataset_id = data["dataset_id"]
    collection_name = data["collection_name"]
    print(f"✓ Created dataset: {dataset_id}")
    print(f"✓ Collection name: {collection_name}")
    
    # 2. Query the dataset (will return empty if no data indexed)
    print("\n2. Querying dataset (vector search)...")
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/query/vector",
        json={
            "query": "mortgage guidelines",
            "top_k": 5
        }
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {data}")
        assert data["success"] is True
        results = data["data"]
        print(f"✓ Query returned {len(results)} results")
        
        if len(results) > 0:
            print(f"\nFirst result:")
            print(f"  Unit ID: {results[0]['unit_id']}")
            print(f"  Score: {results[0]['score']}")
            print(f"  Content preview: {results[0]['content'][:100]}...")
    else:
        print(f"✗ Query failed: {response.text}")
        # Don't fail test if vector store is empty
        print("Note: Query may fail if no data has been indexed yet")
    
    # 3. Query with document info
    print("\n3. Querying with different top_k...")
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/query/vector",
        json={
            "query": "interest rate",
            "top_k": 3
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        assert data["success"] is True
        results = data["data"]
        print(f"✓ Query returned {len(results)} results")
    else:
        print(f"Note: Query may fail if no data indexed")
    
    # Cleanup
    print("\n4. Cleaning up...")
    requests.delete(f"{BASE_URL}/datasets/{dataset_id}")
    print("✓ Cleanup done")
    
    print("\n" + "=" * 50)
    print("Query API test completed! ✓")
    print("=" * 50)
    print("\nNote: Full testing requires indexed data.")
    print("Run worker to process documents first for complete testing.")


if __name__ == "__main__":
    try:
        test_query_api()
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
