#!/usr/bin/env python3
"""
Query API Example
Fast query endpoint without LLM-based postprocessors
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000/query"

# Query request
query = "What optional documentation forms can a lender use to verify the income of a family looking to upsize to a larger home?"

# Send request
response = requests.post(
    API_URL,
    json={
        "query": query,
        "top_k": 5
    }
)

# Check response
if response.status_code == 200:
    data = response.json()
    
    print(f"\nQuery: {data['query']}")
    print(f"Total time: {data['total_time']:.2f}s")
    print(f"Results: {len(data['results'])}")
    print("\nTiming breakdown:")
    for stage, time_spent in data['timing'].items():
        print(f"  {stage}: {time_spent:.2f}s")
    
    print("\n" + "=" * 80)
    for i, result in enumerate(data['results'], 1):
        print(f"\n[Result {i}]")
        print(f"Score: {result['score']:.4f}")
        print(f"Unit ID: {result['unit_id']}")
        
        if result.get('metadata'):
            print("\nMetadata:")
            for key, value in result['metadata'].items():
                print(f"  {key}: {value}")
        
        print(f"\nContent:\n{result['content'][:200]}...")
        print("=" * 80)
else:
    print(f"Error: {response.status_code}")
    print(response.text)
