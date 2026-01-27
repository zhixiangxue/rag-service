"""Query API usage examples.

API Documentation（Dev环境）:
    http://13.56.109.233:8000/docs

Data Structure:
    每条查询结果包含以下字段：
    - unit_id: 文本单元ID
    - doc_id: 文档ID
    - score: 相似度分数
    - content: 文本内容
    - metadata.custom: 自定义元数据，包含：
        - lender: 贷款机构名称 (如: "JMAC Lending", "Mega Capital")
        - tags: 产品标签 (如: "[\"Conventional\"]", "[\"Jumbo\"]")
        - doc_name: 文档名称 (如: "Conforming-And-High-Balance-Guideline-Fannie-Mae")
        - pdf_name: PDF文件名
        - s3_path: S3存储路径
        - overlay: 叠加层信息
        - program_version: 版本号

Filter Syntax:
    我们使用 MongoDB 风格的查询语法来编写 filters (注意：底层不是 MongoDB，只是参考了其语法)。
    
    参考文档: https://www.mongodb.com/zh-cn/docs/manual/reference/mql/query-predicates/
    
    字段路径格式：
    - 使用点号访问嵌套字段：metadata.custom.lender
    - 简单相等直接写值：{"metadata.custom.lender": "JMAC Lending"}
    - 条件查询需要操作符：{"metadata.custom.confidence": {"$gte": 0.85}}
    
    常用操作符：
    - $eq: 等于 (可省略，直接写值)
    - $ne: 不等于
    - $in: 在列表中
    - $contains: 包含 (字符串/数组)
    - $gt/$lt: 大于/小于
    - $gte/$lte: 大于等于/小于等于
    - $or: 或条件 (使用数组)

Usage:
    python test_query_with_filters.py
"""
import requests
import json

BASE_URL = "http://13.56.109.233:8000"
DATASET_ID = "mortgage_guidelines"


def example1_basic_query():
    """Example 1: Basic query (no filters)."""
    print("\n# Example 1: Basic query")
    response = requests.post(
        f"{BASE_URL}/datasets/{DATASET_ID}/query",
        json={
            "query": "What are the mortgage guidelines?",
            "top_k": 5,
            "filters": {}
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Results: {len(response.json()['data'])} items")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def example2_filter_by_lender():
    """Example 2: Filter by lender."""
    print("\n\n# Example 2: Filter by lender")
    response = requests.post(
        f"{BASE_URL}/datasets/{DATASET_ID}/query",
        json={
            "query": "What are the mortgage guidelines?",
            "top_k": 5,
            "filters": {
                "metadata.custom.lender": "JMAC Lending"
            }
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Results: {len(response.json()['data'])} items")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def example3_filter_by_tags():
    """Example 3: Filter by tags."""
    print("\n\n# Example 3: Filter by tags")
    response = requests.post(
        f"{BASE_URL}/datasets/{DATASET_ID}/query",
        json={
            "query": "credit score requirements",
            "top_k": 5,
            "filters": {
                "metadata.custom.tags": {"$contains": "Conventional"}
            }
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Results: {len(response.json()['data'])} items")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def example4_multiple_filters():
    """Example 4: Multiple filters (AND)."""
    print("\n\n# Example 4: Multiple filters (AND)")
    response = requests.post(
        f"{BASE_URL}/datasets/{DATASET_ID}/query",
        json={
            "query": "mortgage insurance",
            "top_k": 5,
            "filters": {
                "metadata.custom.lender": "JMAC Lending",
                "metadata.custom.tags": {"$contains": "Conventional"}
            }
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Results: {len(response.json()['data'])} items")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def example5_filter_by_doc_name():
    """Example 5: Filter by doc_name."""
    print("\n\n# Example 5: Filter by doc_name")
    response = requests.post(
        f"{BASE_URL}/datasets/{DATASET_ID}/query",
        json={
            "query": "interest rate",
            "top_k": 5,
            "filters": {
                "metadata.custom.doc_name": "Sapphire-Jumbo"
            }
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Results: {len(response.json()['data'])} items")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")


def main():
    example1_basic_query()
    example2_filter_by_lender()
    example3_filter_by_tags()
    example4_multiple_filters()
    example5_filter_by_doc_name()


if __name__ == "__main__":
    main()
