"""Test zag VectorRetriever directly."""
import os
from dotenv import load_dotenv

load_dotenv()

from zag.embedders import Embedder
from zag.storages.vector import QdrantVectorStore
from zag.retrievers.basic import VectorRetriever

VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST", "localhost")
VECTOR_STORE_PORT = int(os.getenv("VECTOR_STORE_PORT", "6333"))
EMBEDDING_URI = os.getenv("EMBEDDING_URI", "openai/text-embedding-3-small")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"Connecting to Qdrant at {VECTOR_STORE_HOST}:{VECTOR_STORE_PORT}")
print(f"Embedding: {EMBEDDING_URI}")

try:
    # Initialize embedder
    print("\n1. Creating embedder...")
    embedder = Embedder(EMBEDDING_URI, api_key=OPENAI_API_KEY)
    print("✓ Embedder created")
    
    # Initialize vector store
    print("\n2. Creating vector store...")
    vector_store = QdrantVectorStore.server(
        host=VECTOR_STORE_HOST,
        port=VECTOR_STORE_PORT,
        prefer_grpc=False,
        collection_name="mortgage_guidelines",
        embedder=embedder,
        timeout=60
    )
    print("✓ Vector store created")
    
    # Initialize retriever
    print("\n3. Creating retriever...")
    retriever = VectorRetriever(vector_store=vector_store)
    print("✓ Retriever created")
    
    # Perform search
    print("\n4. Performing search...")
    results = retriever.retrieve(
        query="mortgage guidelines",
        top_k=2
    )
    print(f"✓ Search completed, found {len(results)} results")
    
    if results:
        print(f"\nFirst result:")
        print(f"  Unit ID: {results[0].unit_id}")
        print(f"  Score: {results[0].score}")
        print(f"  Content: {results[0].content[:100]}...")
        
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
