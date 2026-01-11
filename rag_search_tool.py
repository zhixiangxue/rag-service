"""RAG知识库搜索工具 - 从房屋抵押贷款专业知识库中检索信息"""
import os
from dotenv import load_dotenv
import chak


# 加载环境变量
load_dotenv()


async def simulate_rag_search(query: str) -> str:
    """
    从房屋抵押贷款专业知识库中检索相关信息
    
    Args:
        query: 搜索查询关键词
        
    Returns:
        知识库中的相关专业知识条目
    """
    # 创建一个临时的千问会话来生成模拟结果
    api_key = os.getenv("BAILIAN_API_KEY")
    temp_conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key
    )
    
    prompt = f"""作为一个房屋抵押贷款知识库系统，请针对以下查询返回相关的专业知识片段：

查询: {query}

请返回2-3条简洁的知识库条目，每条包含标题和内容，格式如下：
[知识库条目1] 标题：...
内容：...

[知识库条目2] 标题：...
内容：...

保持专业且简洁。"""
    
    response = await temp_conv.asend(prompt)
    return f"[RAG搜索结果]\n{response.content}"
