"""工具函数 - 用于展示和辅助功能"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# 初始化Rich Console
console = Console()


def display_user_info_table(user_info_collector, last_info_state):
    """
    用Rich美化展示已搜集的用户信息
    
    Args:
        user_info_collector: 用户信息搜集器实例
        last_info_state: 上次信息状态字典
        
    Returns:
        更新后的信息状态字典
    """
    current_info = {
        '贷款金额': user_info_collector.loan_amount,
        '年收入': user_info_collector.annual_income,
        '信用分数': user_info_collector.credit_score,
        '首付比例': user_info_collector.down_payment_percent,
        '工作年限': user_info_collector.employment_years
    }
    
    # 检查是否有新信息
    has_new_info = False
    for key, value in current_info.items():
        if value is not None and last_info_state.get(key) != value:
            has_new_info = True
            break
    
    if has_new_info:
        # 创建表格
        table = Table(title="✨ 已搜集的用户信息", show_header=True, header_style="bold magenta")
        table.add_column("信息项", style="cyan", width=15)
        table.add_column("数值", style="green", width=30)
        table.add_column("状态", justify="center", width=8)
        
        # 添加数据行
        if user_info_collector.loan_amount is not None:
            table.add_row(
                "贷款金额",
                f"${user_info_collector.loan_amount:,.2f}",
                "✅" if last_info_state.get('贷款金额') != user_info_collector.loan_amount else ""
            )
        
        if user_info_collector.annual_income is not None:
            table.add_row(
                "年收入",
                f"${user_info_collector.annual_income:,.2f}",
                "✅" if last_info_state.get('年收入') != user_info_collector.annual_income else ""
            )
        
        if user_info_collector.credit_score is not None:
            table.add_row(
                "信用分数",
                str(user_info_collector.credit_score),
                "✅" if last_info_state.get('信用分数') != user_info_collector.credit_score else ""
            )
        
        if user_info_collector.down_payment_percent is not None:
            table.add_row(
                "首付比例",
                f"{user_info_collector.down_payment_percent}%",
                "✅" if last_info_state.get('首付比例') != user_info_collector.down_payment_percent else ""
            )
        
        if user_info_collector.employment_years is not None:
            table.add_row(
                "工作年限",
                f"{user_info_collector.employment_years}年",
                "✅" if last_info_state.get('工作年限') != user_info_collector.employment_years else ""
            )
        
        # 显示表格
        console.print()
        console.print(Panel(table, border_style="bright_blue", padding=(1, 2)))
    
    # 返回更新后的状态
    return current_info.copy()
