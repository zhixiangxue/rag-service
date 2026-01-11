"""网络搜索工具 - 通过搜索引擎查询房屋抵押贷款相关信息"""
import os
from dotenv import load_dotenv
import chak


# 加载环境变量
load_dotenv()


async def simulate_web_search(query: str) -> str:
    """
    通过搜索引擎查询互联网上的最新信息
    
    Args:
        query: 搜索查询关键词
        
    Returns:
        搜索引擎返回的相关网页结果摘要
    """
    # 创建一个临时的千问会话来生成模拟结果
    api_key = os.getenv("BAILIAN_API_KEY")
    temp_conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key
    )
    
    prompt = f"""作为一个搜索引擎，请针对以下查询返回3-4条相关的网页搜索结果摘要：

查询: {query}

每条结果格式如下：
[搜索结果1] 来源：某某网站
摘要：...

[搜索结果2] 来源：某某网站
摘要：...

请模拟真实的搜索结果，包含来源和简短摘要。"""
    
    response = await temp_conv.asend(prompt)
    return f"[网络搜索结果]\n{response.content}"
