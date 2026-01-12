"""用户信息搜集工具类 - 用于收集美国房屋抵押贷款申请所需的用户关键信息"""


class UserInfoCollector:
    """搜集用户申请美国房屋抵押贷款的关键信息"""
    
    def __init__(self):
        self.loan_amount = None  # 贷款金额（美元）
        self.annual_income = None  # 年收入（美元）
        self.credit_score = None  # 信用分数（300-850）
        self.down_payment_percent = None  # 首付比例（百分比）
        self.employment_years = None  # 工作年限（年）
    
    def set_loan_amount(self, amount: float) -> str:
        """
        设置贷款金额
        
        Args:
            amount: 贷款金额（美元）
            
        Returns:
            确认信息
        """
        self.loan_amount = amount
        return f"已记录贷款金额: ${amount:,.2f}"
    
    def set_annual_income(self, income: float) -> str:
        """
        设置年收入
        
        Args:
            income: 年收入（美元）
            
        Returns:
            确认信息
        """
        self.annual_income = income
        return f"已记录年收入: ${income:,.2f}"
    
    def set_credit_score(self, score: int) -> str:
        """
        设置信用分数
        
        Args:
            score: 信用分数（300-850）
            
        Returns:
            确认信息
        """
        if score < 300 or score > 850:
            return "信用分数范围应在300-850之间，请提供有效的分数"
        self.credit_score = score
        return f"已记录信用分数: {score}"
    
    def set_down_payment_percent(self, percent: float) -> str:
        """
        设置首付比例
        
        Args:
            percent: 首付比例（百分比，如20表示20%）
            
        Returns:
            确认信息
        """
        self.down_payment_percent = percent
        return f"已记录首付比例: {percent}%"
    
    def set_employment_years(self, years: float) -> str:
        """
        设置工作年限
        
        Args:
            years: 工作年限（年）
            
        Returns:
            确认信息
        """
        self.employment_years = years
        return f"已记录工作年限: {years}年"
    
    def get_all_info(self) -> dict:
        """
        获取所有已搜集的用户信息
        
        Returns:
            包含所有用户信息的字典
        """
        info = {
            "贷款金额": f"${self.loan_amount:,.2f}" if self.loan_amount else "未填写",
            "年收入": f"${self.annual_income:,.2f}" if self.annual_income else "未填写",
            "信用分数": str(self.credit_score) if self.credit_score else "未填写",
            "首付比例": f"{self.down_payment_percent}%" if self.down_payment_percent else "未填写",
            "工作年限": f"{self.employment_years}年" if self.employment_years else "未填写"
        }
        
        summary = "当前已搜集的用户信息:\n"
        for key, value in info.items():
            summary += f"  - {key}: {value}\n"
        
        return summary
    
    def is_complete(self) -> bool:
        """
        检查是否所有必要信息都已搜集
        
        Returns:
            如果所有信息都已填写返回True，否则返回False
        """
        return all([
            self.loan_amount is not None,
            self.annual_income is not None,
            self.credit_score is not None,
            self.down_payment_percent is not None,
            self.employment_years is not None
        ])
    
    def get_missing_fields(self) -> str:
        """
        获取尚未填写的字段列表
        
        Returns:
            缺失字段的描述
        """
        missing = []
        if self.loan_amount is None:
            missing.append("贷款金额")
        if self.annual_income is None:
            missing.append("年收入")
        if self.credit_score is None:
            missing.append("信用分数")
        if self.down_payment_percent is None:
            missing.append("首付比例")
        if self.employment_years is None:
            missing.append("工作年限")
        
        if not missing:
            return "所有信息已完整搜集"
        
        return f"还需要搜集以下信息: {', '.join(missing)}"
