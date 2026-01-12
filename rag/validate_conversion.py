#!/usr/bin/env python3
"""
PDF 转 Markdown 质量验证工具

自动化验证 PDF 转换质量，包括：
1. 结构完整性检查（章节、标题层级）
2. 表格质量分析
3. 内容统计对比
4. 格式规范检查
5. 生成可视化报告
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import fitz  # PyMuPDF

console = Console()


@dataclass
class ValidationMetrics:
    """验证指标数据类"""
    # 基础统计
    char_count: int = 0
    word_count: int = 0
    line_count: int = 0
    
    # 结构元素
    heading_counts: Dict[int, int] = field(default_factory=dict)  # {level: count}
    table_count: int = 0
    list_count: int = 0
    
    # 格式问题
    broken_tables: List[str] = field(default_factory=list)
    malformed_headings: List[str] = field(default_factory=list)
    encoding_issues: List[str] = field(default_factory=list)
    
    # PDF 特有
    page_count: int = 0
    pdf_tables: int = 0


class MarkdownValidator:
    """Markdown 质量验证器"""
    
    def __init__(self, md_path: Path):
        self.md_path = md_path
        self.content = md_path.read_text(encoding='utf-8')
        self.lines = self.content.split('\n')
        self.metrics = ValidationMetrics()
    
    def validate(self) -> ValidationMetrics:
        """执行完整验证"""
        console.print(f"\n[cyan]📝 验证 Markdown: {self.md_path.name}[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]分析中...", total=6)
            
            self._count_basic_stats()
            progress.update(task, advance=1)
            
            self._analyze_headings()
            progress.update(task, advance=1)
            
            self._analyze_tables()
            progress.update(task, advance=1)
            
            self._analyze_lists()
            progress.update(task, advance=1)
            
            self._check_format_issues()
            progress.update(task, advance=1)
            
            self._check_encoding()
            progress.update(task, advance=1)
        
        return self.metrics
    
    def _count_basic_stats(self):
        """统计基础指标"""
        self.metrics.char_count = len(self.content)
        self.metrics.word_count = len(re.findall(r'\b\w+\b', self.content))
        self.metrics.line_count = len(self.lines)
    
    def _analyze_headings(self):
        """分析标题结构"""
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        for i, line in enumerate(self.lines, 1):
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                self.metrics.heading_counts[level] = self.metrics.heading_counts.get(level, 0) + 1
                
                # 检查标题是否规范
                title = match.group(2).strip()
                if not title:
                    self.metrics.malformed_headings.append(f"第 {i} 行: 空标题")
                elif len(title) > 200:
                    self.metrics.malformed_headings.append(f"第 {i} 行: 标题过长 ({len(title)} 字符)")
    
    def _analyze_tables(self):
        """分析表格质量"""
        in_table = False
        table_lines = []
        table_start = 0
        
        for i, line in enumerate(self.lines, 1):
            is_table_line = bool(re.match(r'^\|.*\|$', line))
            
            if is_table_line:
                if not in_table:
                    in_table = True
                    table_start = i
                    table_lines = [line]
                else:
                    table_lines.append(line)
            elif in_table:
                # 表格结束，验证质量
                in_table = False
                self.metrics.table_count += 1
                self._validate_table(table_lines, table_start)
                table_lines = []
        
        # 处理文件末尾的表格
        if in_table:
            self.metrics.table_count += 1
            self._validate_table(table_lines, table_start)
    
    def _validate_table(self, lines: List[str], start_line: int):
        """验证单个表格的质量"""
        if len(lines) < 2:
            self.metrics.broken_tables.append(
                f"第 {start_line} 行: 表格行数过少 ({len(lines)} 行)"
            )
            return
        
        # 检查列数一致性（正确处理空列）
        col_counts = []
        for line in lines:
            # 分割后，移除首尾的空字符串（| 前后的）
            cols = line.split('|')
            # 只移除首尾，中间的空格保留
            if cols and cols[0] == '':
                cols = cols[1:]
            if cols and cols[-1] == '':
                cols = cols[:-1]
            col_counts.append(len(cols))
        
        # 允许 ±1 列的误差（因为 Markdown 表格格式灵活）
        min_cols = min(col_counts)
        max_cols = max(col_counts)
        
        if max_cols - min_cols > 1:
            self.metrics.broken_tables.append(
                f"第 {start_line} 行: 表格列数差异较大 (最少{min_cols}列, 最多{max_cols}列)"
            )
        
        # 检查是否有分隔线
        has_separator = any('---' in line or '━' in line or '─' in line for line in lines[:3])
        if not has_separator and len(lines) > 2:
            # 不报告缺少分隔线，因为有些表格确实没有
            pass
    
    def _analyze_lists(self):
        """分析列表"""
        list_pattern = re.compile(r'^\s*[-*+]\s+\S')
        self.metrics.list_count = sum(1 for line in self.lines if list_pattern.match(line))
    
    def _check_format_issues(self):
        """检查格式问题"""
        for i, line in enumerate(self.lines, 1):
            # 检查连续多个空行
            if i > 1 and not line.strip() and not self.lines[i-2].strip():
                pass  # 暂不报告，太常见
            
            # 检查行末空格（可能影响 Markdown 渲染）
            if line.endswith(' ' * 3):
                pass  # Markdown 的换行语法，正常
    
    def _check_encoding(self):
        """检查编码问题"""
        problematic_chars = ['�', '\ufffd', '\x00']
        
        for i, line in enumerate(self.lines, 1):
            for char in problematic_chars:
                if char in line:
                    preview = line[:50].replace('\n', ' ')
                    self.metrics.encoding_issues.append(
                        f"第 {i} 行: 编码问题字符 '{char}' - {preview}..."
                    )
                    break


class PDFAnalyzer:
    """PDF 文档分析器"""
    
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.metrics = ValidationMetrics()
    
    def analyze(self) -> ValidationMetrics:
        """分析 PDF 文档"""
        console.print(f"\n[cyan]📄 分析 PDF: {self.pdf_path.name}[/cyan]")
        
        try:
            doc = fitz.open(self.pdf_path)
            self.metrics.page_count = len(doc)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(
                    "[cyan]提取 PDF 信息...", 
                    total=len(doc)
                )
                
                total_chars = 0
                total_words = 0
                
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text()
                    total_chars += len(text)
                    total_words += len(re.findall(r'\b\w+\b', text))
                    
                    # 统计表格（粗略估计）
                    try:
                        tables = page.find_tables()
                        if tables and hasattr(tables, '__iter__'):
                            self.metrics.pdf_tables += len(list(tables))
                    except:
                        pass  # 忽略表格检测错误
                    
                    progress.update(task, advance=1)
                
                self.metrics.char_count = total_chars
                self.metrics.word_count = total_words
            
            doc.close()
            
        except Exception as e:
            console.print(f"[red]❌ PDF 分析失败: {e}[/red]")
        
        return self.metrics


class ConversionValidator:
    """转换质量验证器"""
    
    def __init__(self, pdf_path: Path, md_path: Path):
        self.pdf_path = pdf_path
        self.md_path = md_path
    
    def validate(self):
        """执行完整验证流程"""
        console.print("\n" + "=" * 70)
        console.print("  🔍 PDF 转 Markdown 质量验证", style="bold cyan")
        console.print("=" * 70)
        
        # 分析 PDF
        pdf_analyzer = PDFAnalyzer(self.pdf_path)
        pdf_metrics = pdf_analyzer.analyze()
        
        # 验证 Markdown
        md_validator = MarkdownValidator(self.md_path)
        md_metrics = md_validator.validate()
        
        # 生成报告
        self._generate_report(pdf_metrics, md_metrics)
    
    def _generate_report(self, pdf: ValidationMetrics, md: ValidationMetrics):
        """生成验证报告"""
        console.print("\n" + "=" * 70)
        console.print("  📊 验证报告", style="bold green")
        console.print("=" * 70)
        
        # 1. 基础统计对比
        self._print_basic_stats(pdf, md)
        
        # 2. 结构分析
        self._print_structure_analysis(md)
        
        # 3. 问题汇总
        self._print_issues(md)
        
        # 4. 整体评分
        self._print_score(pdf, md)
    
    def _print_basic_stats(self, pdf: ValidationMetrics, md: ValidationMetrics):
        """打印基础统计对比"""
        table = Table(title="📈 基础统计对比", show_header=True)
        table.add_column("指标", style="cyan", width=20)
        table.add_column("PDF", style="yellow", justify="right", width=15)
        table.add_column("Markdown", style="green", justify="right", width=15)
        table.add_column("保留率", style="magenta", justify="right", width=15)
        
        # 页数/行数
        table.add_row(
            "页数/行数",
            f"{pdf.page_count:,} 页",
            f"{md.line_count:,} 行",
            "N/A"
        )
        
        # 字符数
        char_rate = (md.char_count / pdf.char_count * 100) if pdf.char_count > 0 else 0
        table.add_row(
            "字符数",
            f"{pdf.char_count:,}",
            f"{md.char_count:,}",
            f"{char_rate:.1f}%"
        )
        
        # 单词数
        word_rate = (md.word_count / pdf.word_count * 100) if pdf.word_count > 0 else 0
        table.add_row(
            "单词数",
            f"{pdf.word_count:,}",
            f"{md.word_count:,}",
            f"{word_rate:.1f}%"
        )
        
        # 表格数
        if pdf.pdf_tables > 0:
            table_rate = (md.table_count / pdf.pdf_tables * 100)
            table.add_row(
                "表格数",
                f"{pdf.pdf_tables:,}",
                f"{md.table_count:,}",
                f"{table_rate:.1f}%"
            )
        
        console.print(table)
    
    def _print_structure_analysis(self, md: ValidationMetrics):
        """打印结构分析"""
        console.print("\n[bold cyan]📚 文档结构分析[/bold cyan]")
        
        table = Table(show_header=True)
        table.add_column("元素类型", style="cyan", width=20)
        table.add_column("数量", style="green", justify="right", width=15)
        
        # 标题层级
        if md.heading_counts:
            for level in sorted(md.heading_counts.keys()):
                table.add_row(
                    f"{'#' * level} 标题 (H{level})",
                    f"{md.heading_counts[level]:,}"
                )
        
        # 表格
        table.add_row("表格", f"{md.table_count:,}")
        
        # 列表
        table.add_row("列表项", f"{md.list_count:,}")
        
        console.print(table)
    
    def _print_issues(self, md: ValidationMetrics):
        """打印问题汇总"""
        console.print("\n[bold yellow]⚠️  问题汇总[/bold yellow]")
        
        total_issues = (
            len(md.broken_tables) + 
            len(md.malformed_headings) + 
            len(md.encoding_issues)
        )
        
        if total_issues == 0:
            console.print("[green]✅ 未发现明显问题[/green]")
            return
        
        # 表格问题
        if md.broken_tables:
            console.print(f"\n[red]❌ 表格问题 ({len(md.broken_tables)} 个):[/red]")
            for issue in md.broken_tables[:5]:
                console.print(f"   • {issue}")
            if len(md.broken_tables) > 5:
                console.print(f"   ... 还有 {len(md.broken_tables) - 5} 个问题")
        
        # 标题问题
        if md.malformed_headings:
            console.print(f"\n[yellow]⚠️  标题问题 ({len(md.malformed_headings)} 个):[/yellow]")
            for issue in md.malformed_headings[:5]:
                console.print(f"   • {issue}")
            if len(md.malformed_headings) > 5:
                console.print(f"   ... 还有 {len(md.malformed_headings) - 5} 个问题")
        
        # 编码问题
        if md.encoding_issues:
            console.print(f"\n[red]❌ 编码问题 ({len(md.encoding_issues)} 个):[/red]")
            for issue in md.encoding_issues[:3]:
                console.print(f"   • {issue}")
            if len(md.encoding_issues) > 3:
                console.print(f"   ... 还有 {len(md.encoding_issues) - 3} 个问题")
    
    def _print_score(self, pdf: ValidationMetrics, md: ValidationMetrics):
        """打印整体评分"""
        console.print("\n[bold magenta]🎯 质量评分[/bold magenta]")
        
        # 计算各项得分
        scores = {}
        
        # 1. 内容完整性 (40分)
        char_rate = (md.char_count / pdf.char_count) if pdf.char_count > 0 else 0
        content_score = min(40, char_rate * 40)
        scores['内容完整性'] = (content_score, 40, char_rate * 100)
        
        # 2. 结构完整性 (30分)
        structure_score = 30
        if md.heading_counts:
            structure_score = 30
        elif md.table_count == 0:
            structure_score = 15
        scores['结构完整性'] = (structure_score, 30, structure_score / 30 * 100)
        
        # 3. 格式规范性 (30分)
        issue_count = (
            len(md.broken_tables) + 
            len(md.malformed_headings) + 
            len(md.encoding_issues)
        )
        format_score = max(0, 30 - issue_count * 2)
        scores['格式规范性'] = (format_score, 30, format_score / 30 * 100)
        
        # 打印得分表
        table = Table(show_header=True)
        table.add_column("评分项", style="cyan", width=20)
        table.add_column("得分", style="green", justify="right", width=10)
        table.add_column("满分", style="yellow", justify="right", width=10)
        table.add_column("百分比", style="magenta", justify="right", width=15)
        
        for name, (score, max_score, percentage) in scores.items():
            table.add_row(
                name,
                f"{score:.1f}",
                f"{max_score}",
                f"{percentage:.1f}%"
            )
        
        total_score = sum(s[0] for s in scores.values())
        total_max = sum(s[1] for s in scores.values())
        
        table.add_row(
            "[bold]总分[/bold]",
            f"[bold]{total_score:.1f}[/bold]",
            f"[bold]{total_max}[/bold]",
            f"[bold]{total_score / total_max * 100:.1f}%[/bold]"
        )
        
        console.print(table)
        
        # 评级
        if total_score >= 90:
            grade = "优秀 🎉"
            style = "bold green"
        elif total_score >= 75:
            grade = "良好 👍"
            style = "bold cyan"
        elif total_score >= 60:
            grade = "及格 ✓"
            style = "bold yellow"
        else:
            grade = "需改进 ⚠️"
            style = "bold red"
        
        console.print(f"\n[{style}]质量评级: {grade}[/{style}]")


def main():
    """主函数"""
    import sys
    
    # 使用示例
    output_dir = Path(__file__).parent / "output"
    files_dir = Path(__file__).parent / "files"
    
    # 查找所有 PDF 和对应的 MD 文件
    pdf_files = list(files_dir.glob("*.pdf"))
    
    if not pdf_files:
        console.print("[red]❌ 未找到 PDF 文件[/red]")
        sys.exit(1)
    
    for pdf_path in pdf_files:
        md_path = output_dir / f"{pdf_path.stem}.md"
        
        if not md_path.exists():
            console.print(f"[yellow]⚠️  跳过 {pdf_path.name}: 未找到对应的 MD 文件[/yellow]")
            continue
        
        validator = ConversionValidator(pdf_path, md_path)
        validator.validate()
        
        console.print("\n")


def validate_cache_quality(pdf_path: Path, md_path: Path, threshold: float = 90.0, verbose: bool = False) -> bool:
    """
    验证缓存文件质量
    
    Args:
        pdf_path: PDF 文件路径
        md_path: Markdown 文件路径
        threshold: 质量阈值（默认 90 分）
        verbose: 是否输出详细信息
        
    Returns:
        bool: True 表示缓存质量合格，False 表示需要重新生成
    """
    if not md_path.exists():
        return False
    
    try:
        # 分析 PDF
        pdf_analyzer = PDFAnalyzer(pdf_path)
        pdf_metrics = pdf_analyzer.analyze() if verbose else _analyze_pdf_silent(pdf_path)
        
        # 验证 Markdown
        md_validator = MarkdownValidator(md_path)
        md_metrics = md_validator.validate() if verbose else _validate_markdown_silent(md_path)
        
        # 计算总分
        score = _calculate_quality_score(pdf_metrics, md_metrics)
        
        if verbose:
            if score >= threshold:
                console.print(f"[green]✅ 缓存质量合格: {score:.1f}/100[/green]")
            else:
                console.print(f"[yellow]⚠️  缓存质量不足: {score:.1f}/100 (阈值: {threshold})[/yellow]")
        
        return score >= threshold
        
    except Exception as e:
        if verbose:
            console.print(f"[red]❌ 验证失败: {e}[/red]")
        return False


def _analyze_pdf_silent(pdf_path: Path) -> ValidationMetrics:
    """静默分析 PDF（不输出）"""
    metrics = ValidationMetrics()
    try:
        doc = fitz.open(pdf_path)
        metrics.page_count = len(doc)
        
        total_chars = 0
        total_words = 0
        
        for page in doc:
            text = page.get_text()
            total_chars += len(text)
            total_words += len(re.findall(r'\b\w+\b', text))
        
        metrics.char_count = total_chars
        metrics.word_count = total_words
        doc.close()
    except Exception:
        pass
    
    return metrics


def _validate_markdown_silent(md_path: Path) -> ValidationMetrics:
    """静默验证 Markdown（不输出）"""
    metrics = ValidationMetrics()
    try:
        content = md_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # 基础统计
        metrics.char_count = len(content)
        metrics.word_count = len(re.findall(r'\b\w+\b', content))
        metrics.line_count = len(lines)
        
        # 标题统计
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        for line in lines:
            match = heading_pattern.match(line)
            if match:
                level = len(match.group(1))
                metrics.heading_counts[level] = metrics.heading_counts.get(level, 0) + 1
                title = match.group(2).strip()
                if not title or len(title) > 200:
                    metrics.malformed_headings.append("")
        
        # 表格统计
        in_table = False
        table_lines = []
        for line in lines:
            is_table_line = bool(re.match(r'^\|.*\|$', line))
            if is_table_line:
                if not in_table:
                    in_table = True
                    table_lines = [line]
                else:
                    table_lines.append(line)
            elif in_table:
                metrics.table_count += 1
                # 简化验证
                if len(table_lines) >= 2:
                    col_counts = []
                    for tline in table_lines:
                        cols = tline.split('|')
                        if cols and cols[0] == '':
                            cols = cols[1:]
                        if cols and cols[-1] == '':
                            cols = cols[:-1]
                        col_counts.append(len(cols))
                    if max(col_counts) - min(col_counts) > 1:
                        metrics.broken_tables.append("")
                in_table = False
                table_lines = []
        
        if in_table:
            metrics.table_count += 1
        
        # 编码检查
        problematic_chars = ['�', '\ufffd', '\x00']
        for line in lines:
            if any(char in line for char in problematic_chars):
                metrics.encoding_issues.append("")
                break
    
    except Exception:
        pass
    
    return metrics


def _calculate_quality_score(pdf: ValidationMetrics, md: ValidationMetrics) -> float:
    """
    计算质量得分
    
    Args:
        pdf: PDF 指标
        md: Markdown 指标
        
    Returns:
        float: 质量得分 (0-100)
    """
    # 1. 内容完整性 (40分)
    char_rate = (md.char_count / pdf.char_count) if pdf.char_count > 0 else 0
    content_score = min(40, char_rate * 40)
    
    # 2. 结构完整性 (30分)
    structure_score = 30
    if not md.heading_counts:
        structure_score = 15
    elif md.table_count == 0:
        structure_score = 20
    
    # 3. 格式规范性 (30分)
    issue_count = (
        len(md.broken_tables) + 
        len(md.malformed_headings) + 
        len(md.encoding_issues)
    )
    format_score = max(0, 30 - issue_count * 2)
    
    total_score = content_score + structure_score + format_score
    return total_score


if __name__ == "__main__":
    main()
