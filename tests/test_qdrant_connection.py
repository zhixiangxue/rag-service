"""Test Qdrant connection and check collection."""
from qdrant_client import QdrantClient

# Test connection
QDRANT_HOST = "13.56.109.233"
QDRANT_PORT = 16333

print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")

try:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    
    # List all collections
    print("\nListing all collections...")
    collections = client.get_collections()
    print(f"Found {len(collections.collections)} collections:")
    for collection in collections.collections:
        print(f"  - {collection.name}")
    
    # Check mortgage_guidelines collection
    collection_name = "mortgage_guidelines"
    if any(c.name == collection_name for c in collections.collections):
        print(f"\n✓ Collection '{collection_name}' exists!")
        
        # Get collection info
        info = client.get_collection(collection_name)
        print(f"  Points count: {info.points_count}")
        print(f"  Vector size: {info.config.params.vectors.size}")
    else:
        print(f"\n✗ Collection '{collection_name}' not found!")
        
except Exception as e:
    print(f"\n✗ Connection failed: {e}")
    import traceback
    traceback.print_exc()
