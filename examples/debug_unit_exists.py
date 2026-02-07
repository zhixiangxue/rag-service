"""Debug script to check if unit exists in vector store."""
import os
from dotenv import load_dotenv
from zag.storages.vector import QdrantVectorStore
from zag.embedders import Embedder

load_dotenv()

# Config
UNIT_ID = "351864a6b9cd0bc9_lod"
COLLECTION_NAME = "xxx"

print(f"Checking unit: {UNIT_ID}")
print(f"Collection: {COLLECTION_NAME}")

# Initialize embedder
embedder = Embedder(
    uri=os.getenv("EMBEDDING_URI"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize vector store
store = QdrantVectorStore.server(
    host="localhost",
    port=16333,
    prefer_grpc=False,
    collection_name=COLLECTION_NAME,
    embedder=embedder
)

# Debug: Check hash conversion
print(f"\nDebug hash conversion:")
print(f"  Unit ID: {UNIT_ID}")
qdrant_id = abs(hash(UNIT_ID)) % (2**63)
print(f"  Qdrant ID: {qdrant_id}")

# Check if unit exists
print(f"\nChecking via get()...")
units = store.get([UNIT_ID])
print(f"Found {len(units)} units")

if units:
    unit = units[0]
    print(f"\n✓ Unit found!")
    print(f"  Unit ID: {unit.unit_id}")
    print(f"  Doc ID: {unit.metadata.doc_id if hasattr(unit.metadata, 'doc_id') else 'N/A'}")
    print(f"  Content length: {len(unit.content)}")
    
    # Check views
    if hasattr(unit, 'views') and unit.views:
        print(f"\n  Views: {len(unit.views)}")
        for view in unit.views:
            print(f"    - {view.level}: {len(str(view.content))} chars")
    else:
        print(f"\n  No views found")
else:
    print(f"\n✗ Unit NOT found via get()")
    print(f"\nSearching for any units in collection...")
    
    # Search for any units
    results = store.search(
        query="CalHFA",
        top_k=20
    )
    
    print(f"\nFound {len(results)} total units")
    for r in results[:10]:
        metadata = r.metadata if isinstance(r.metadata, dict) else r.metadata.__dict__
        mode = metadata.get('custom_metadata', {}).get('mode', 'unknown') if 'custom_metadata' in metadata else 'unknown'
        doc_id = metadata.get('doc_id', 'N/A')
        print(f"  - {r.unit_id} (doc_id: {doc_id}, mode: {mode})")
    
    # Try direct Qdrant API
    print(f"\nTrying direct Qdrant API...")
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=16333)
    
    # Get the specific point by ID
    target_qdrant_id = 7192881365615020790
    try:
        point = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[target_qdrant_id],
            with_payload=True,
            with_vectors=False
        )
        if point:
            p = point[0]
            print(f"\n✓ Found point with ID {target_qdrant_id}")
            print(f"  Stored unit_id: {p.payload.get('unit_id')}")
            print(f"  Doc_id: {p.payload.get('doc_id')}")
            print(f"  Has views: {bool(p.payload.get('views'))}")
            
            # Compare with hash
            stored_unit_id = p.payload.get('unit_id')
            computed_hash = abs(hash(stored_unit_id)) % (2**63)
            print(f"\n  Hash comparison:")
            print(f"    Computed from stored_unit_id: {computed_hash}")
            print(f"    Actual Qdrant ID: {target_qdrant_id}")
            print(f"    Match: {computed_hash == target_qdrant_id}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Scroll to get all points
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10,
        with_payload=True,
        with_vectors=False
    )
    
    print(f"\nFound {len(points)} points via scroll")
    for p in points:
        print(f"  Point ID: {p.id}, Unit ID: {p.payload.get('unit_id', 'N/A')}")

