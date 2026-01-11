"""美国房屋抵押贷款咨询对话系统 - 基于chak和千问模型"""
import os
import asyncio
from dotenv import load_dotenv
import chak
from user_info_collector import UserInfoCollector
from rag_search_tool import simulate_rag_search
from web_search_tool import simulate_web_search
from utils import display_user_info_table


# 加载环境变量
load_dotenv()


# 系统提示词 - 定义AI助手的角色
SYSTEM_PROMPT = """你是一位专业的美国房屋抵押贷款顾问，为C端用户提供咨询服务。

你的职责：
1. 回答用户关于美国房屋抵押贷款的各种问题（如：利率、贷款类型、申请流程、资格要求等）
2. 通过自然对话方式，逐步收集用户的关键信息（贷款金额、年收入、信用分数、首付比例、工作年限）
3. 根据用户的具体情况，提供个性化的建议和解答

工具使用策略（非常重要！）：
1. 【优先使用RAG搜索工具】- 当需要查询房屏贷款专业知识时，首先使用 simulate_rag_search 工具查询内部知识库
2. 【慢用网络搜索工具】- 只有在以下情况才使用 simulate_web_search：
   - RAG知识库无法找到相关信息
   - 用户明确询问最新政策或实时信息（如“最新利率”、“2026年政策”）
   - 需要查找特定机构或地区的信息
3. 【用户信息搜集】- 当用户提供关键信息时，务必使用 UserInfoCollector 工具记录

你的沟通风格：
- 专业但友好，用通俗易懂的语言解释复杂的金融概念
- 耐心倾听用户需求，不要一次性询问过多信息
- 在对话中自然地引导用户提供必要信息
- 对于常见的贷款问题，优先从知识库（RAG）获取权威信息

注意事项：
- 保持对话的连贯性和上下文理解
- 针对用户的具体情况给出建议，不要给出过于笼统的回答
- 如果用户提供了关键信息（如收入、信用分数等），务必使用用户信息搜集工具记录下来
"""


async def main():
    """主程序 - 运行对话循环（异步）"""
    
    # 获取API密钥
    api_key = os.getenv("BAILIAN_API_KEY")
    if not api_key:
        print("错误：未找到BAILIAN_API_KEY环境变量，请检查.env文件")
        return
    
    print("=" * 60)
    print("美国房屋抵押贷款咨询系统")
    print("=" * 60)
    print("欢迎！我是您的贷款顾问助手，随时为您解答关于美国房屋抵押贷款的问题。")
    print("输入 'exit' 或 'quit' 退出系统")
    print("输入 'info' 查看已搜集的用户信息")
    print("=" * 60)
    print()
    
    # 创建用户信息搜集器实例
    user_info_collector = UserInfoCollector()
    # 跟踪上次检查时的信息状态
    last_info_state = {}
    
    # 创建对话实例，注册工具（按优先级排序）
    conv = chak.Conversation(
        "bailian/qwen-plus",
        api_key=api_key,
        system_message=SYSTEM_PROMPT,
        tools=[
            simulate_rag_search,
            user_info_collector,
            simulate_web_search
        ]
    )
    
    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("\n您: ").strip()
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("\n感谢您的咨询，祝您贷款申请顺利！")
                break
            
            # 跳过空输入
            if not user_input:
                continue
            
            # 发送消息并获取回复（流式输出）
            print("\n助手: ", end="", flush=True)
            stream = await conv.asend(user_input, stream=True)
            async for chunk in stream:  # type: ignore
                if chunk.content:  # type: ignore
                    print(chunk.content, end="", flush=True)  # type: ignore
            print()  # 换行
            
            # 检查并展示新搜集的用户信息
            last_info_state = display_user_info_table(user_info_collector, last_info_state)
            
        except KeyboardInterrupt:
            print("\n\n程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            print("请重试或输入 'exit' 退出")


if __name__ == "__main__":
    asyncio.run(main())
