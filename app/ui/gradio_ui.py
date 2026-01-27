"""Gradio UI for testing document processing pipeline."""
import gradio as gr
import httpx
import asyncio
from pathlib import Path
from typing import Optional, Tuple
import time

# 从 app.config 导入配置（已加载 .env）
from ..config import API_HOST, API_PORT

# Markdown rendering support
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("[WARNING] 'markdown' library not installed. Content will be displayed as plain text.")

# UI 内部调用使用 localhost（UI 和 API 在同一进程内，用 localhost 更安全快速）
# 如果 API_HOST 是 0.0.0.0，则改用 localhost；否则使用配置的地址
_API_HOST_FOR_UI = "localhost" if API_HOST == "0.0.0.0" else API_HOST
API_BASE_URL = f"http://{_API_HOST_FOR_UI}:{API_PORT}"


def get_config_info() -> list:
    """获取配置信息并脱敏."""
    from .. import config
    
    def mask_sensitive(value: str, key: str) -> str:
        """脱敏敏感信息."""
        if value is None:
            return "Not Set"
        
        # 需要脱敏的字段
        sensitive_keys = ["KEY", "SECRET", "PASSWORD", "TOKEN"]
        if any(k in key.upper() for k in sensitive_keys):
            if len(value) <= 8:
                return "***"
            # 显示前4位和后4位
            return f"{value[:4]}...{value[-4:]}"
        
        return str(value)
    
    # 构建配置表格
    config_data = [
        ["Database", "DATABASE_PATH", config.DATABASE_PATH],
        ["Vector Store", "VECTOR_STORE_TYPE", config.VECTOR_STORE_TYPE],
        ["Vector Store", "VECTOR_STORE_HOST", config.VECTOR_STORE_HOST],
        ["Vector Store", "VECTOR_STORE_PORT", str(config.VECTOR_STORE_PORT)],
        ["Vector Store", "DEFAULT_COLLECTION_NAME", config.DEFAULT_COLLECTION_NAME],
        ["Embedding", "EMBEDDING_URI", config.EMBEDDING_URI],
        ["Embedding", "OPENAI_API_KEY", mask_sensitive(config.OPENAI_API_KEY, "OPENAI_API_KEY")],
        ["Embedding", "BAILIAN_API_KEY", mask_sensitive(config.BAILIAN_API_KEY, "BAILIAN_API_KEY")],
        ["Storage", "UPLOAD_DIR", config.UPLOAD_DIR],
        ["Storage", "STORAGE_TYPE", config.STORAGE_TYPE],
        ["Storage (S3)", "S3_BUCKET", config.S3_BUCKET or "Not Set"],
        ["Storage (S3)", "S3_REGION", config.S3_REGION],
        ["Storage (S3)", "S3_ACCESS_KEY", mask_sensitive(config.S3_ACCESS_KEY, "S3_ACCESS_KEY")],
        ["Storage (S3)", "S3_SECRET_KEY", mask_sensitive(config.S3_SECRET_KEY, "S3_SECRET_KEY")],
        ["API Server", "API_HOST", config.API_HOST],
        ["API Server", "API_PORT", str(config.API_PORT)],
    ]
    
    return config_data


async def list_datasets() -> list[str]:
    """获取所有 dataset 列表."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/datasets")
            if response.status_code == 200:
                data = response.json()["data"]
                dataset_names = [ds["name"] for ds in data]
                print(f"[DEBUG] Loaded {len(dataset_names)} datasets: {dataset_names}")
                return dataset_names
            else:
                print(f"[ERROR] Failed to fetch datasets: {response.status_code}")
                return []
    except Exception as e:
        print(f"[ERROR] Failed to list datasets: {e}")
        import traceback
        traceback.print_exc()
        return []


async def create_dataset(name: str, description: str, engine: str) -> Tuple[bool, str, Optional[str]]:
    """创建新的 dataset.
    
    Returns:
        (success, message, dataset_id)
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE_URL}/datasets",
                json={
                    "name": name,
                    "description": description or f"Dataset: {name}",
                    "engine": engine,
                    "config": {}
                }
            )
            if response.status_code == 200:
                data = response.json()["data"]
                dataset_id = data["dataset_id"]
                return True, f"✅ Dataset '{name}' created successfully", dataset_id
            else:
                error = response.json().get("detail", "Unknown error")
                return False, f"❌ Failed to create dataset: {error}", None
    except Exception as e:
        return False, f"❌ Error: {str(e)}", None


async def submit_task(
    dataset_name: Optional[str],
    file_path: str,
    new_dataset_name: Optional[str],
    new_dataset_desc: Optional[str]
) -> Tuple[str, str]:
    """提交文档处理任务."""
    
    dataset_id = None  # Will store the actual dataset ID
    
    # 如果选择创建新 dataset
    if new_dataset_name:
        success, msg, dataset_id = await create_dataset(
            name=new_dataset_name,
            description=new_dataset_desc or "",
            engine="qdrant"
        )
        if not success:
            return "❌ Failed", msg
    else:
        # 如果使用现有 dataset，需要通过 name 查询 ID
        if not dataset_name:
            return "❌ Failed", "Please select or create a dataset"
        
        try:
            async with httpx.AsyncClient() as client:
                # 获取所有 datasets，找到匹配的 name
                response = await client.get(f"{API_BASE_URL}/datasets")
                if response.status_code == 200:
                    datasets = response.json()["data"]
                    matching_dataset = next((d for d in datasets if d["name"] == dataset_name), None)
                    if matching_dataset:
                        dataset_id = matching_dataset["dataset_id"]
                    else:
                        return "❌ Failed", f"Dataset '{dataset_name}' not found"
                else:
                    return "❌ Failed", "Failed to fetch datasets"
        except Exception as e:
            return "❌ Failed", f"Error fetching dataset: {str(e)}"
    
    if not dataset_id:
        return "❌ Failed", "Dataset ID not found"
    
    if not file_path:
        return "❌ Failed", "Please upload a PDF file"
    
    try:
        # Step 1: Upload file (use dataset_id, not name)
        async with httpx.AsyncClient(timeout=30.0) as client:
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f, "application/pdf")}
                response = await client.post(
                    f"{API_BASE_URL}/datasets/{dataset_id}/documents",
                    files=files
                )
            
            if response.status_code != 200:
                error = response.json().get("detail", "Unknown error")
                return "❌ Failed", f"Upload failed: {error}"
            
            doc_info = response.json()["data"]
            doc_id = doc_info["doc_id"]
            task_id = doc_info["task_id"]
        
        return "✅ Submitted", f"Task ID: {task_id}\nDocument ID: {doc_id}\n\nWorker is processing..."
    
    except Exception as e:
        return "❌ Failed", f"Error: {str(e)}"


async def get_task_status(task_id: str) -> Tuple[str, str, str, str, str]:
    """查询任务状态."""
    if not task_id:
        return "No Task", "-", "-", "-", "Please submit a task first"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/tasks/{task_id}")
            
            if response.status_code == 404:
                return "Not Found", "-", "-", "-", f"Task {task_id} not found"
            
            if response.status_code != 200:
                return "Error", "-", "-", "-", f"API Error: {response.status_code}"
            
            task = response.json()["data"]
            
            status = task["status"]
            progress = f"{task.get('progress', 0)}%"
            units = str(task.get("unit_count", "-"))
            error = task.get("error_message", {}).get("error", "-") if task.get("error_message") else "-"
            
            # 计算 Last updated 时间
            from datetime import datetime
            try:
                updated_at = task["updated_at"]
                # 解析 ISO 时间戳
                updated_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                now_time = datetime.now(updated_time.tzinfo)
                elapsed_seconds = int((now_time - updated_time).total_seconds())
                
                if elapsed_seconds < 60:
                    last_update = f"(updated {elapsed_seconds}s ago)"
                elif elapsed_seconds < 3600:
                    last_update = f"(updated {elapsed_seconds // 60}min ago)"
                else:
                    last_update = f"(updated {elapsed_seconds // 3600}h ago)"
                
                progress_with_time = f"{progress} {last_update}"
            except Exception:
                progress_with_time = progress
            
            # 格式化详细信息
            details = f"Task ID: {task_id}\n"
            details += f"Dataset: {task['dataset_id']}\n"
            details += f"Created: {task['created_at']}\n"
            details += f"Updated: {task['updated_at']}\n"
            
            return status, progress_with_time, units, error, details
    
    except Exception as e:
        return "Error", "-", "-", "-", f"Error: {str(e)}"


async def get_recent_tasks(limit: int = 5) -> list:
    """获取最近的任务历史（返回表格数据）."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/tasks", params={"limit": limit})
            
            if response.status_code != 200:
                return [["Error", "Failed to fetch", "-", "-"]]
            
            tasks = response.json()["data"]
            
            if not tasks:
                return [["No tasks yet", "-", "-", "-"]]
            
            # 构建表格数据：[Task ID, Status, Progress, Created At]
            table_data = []
            for task in tasks:
                status = task["status"]
                task_id = task["task_id"][:8]  # 缩短显示
                progress = f"{task['progress']}%"
                created_at = task["created_at"].split("T")[0]  # 只显示日期
                
                table_data.append([task_id, status, progress, created_at])
            
            return table_data
    
    except Exception as e:
        return [["Error", str(e), "-", "-"]]


async def query_documents(dataset_name: str, query: str, top_k: int) -> str:
    """查询文档并返回 HTML 格式的结果."""
    print(f"[DEBUG] query_documents called - Dataset: {dataset_name}, Query: {query[:30]}...")
    
    if not dataset_name:
        print("[DEBUG] Error: No dataset selected")
        return "<div style='color: red;'>❌ Please select a dataset</div>"
    
    if not query or not query.strip():
        print("[DEBUG] Error: Empty query")
        return "<div style='color: red;'>❌ Please enter a question</div>"
    
    try:
        # 获取 dataset_id
        print(f"[DEBUG] Fetching dataset info for: {dataset_name}")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/datasets")
            if response.status_code == 200:
                datasets = response.json()["data"]
                matching_dataset = next((d for d in datasets if d["name"] == dataset_name), None)
                if not matching_dataset:
                    print(f"[DEBUG] Error: Dataset not found: {dataset_name}")
                    return f"<div style='color: red;'>❌ Dataset '{dataset_name}' not found</div>"
                dataset_id = matching_dataset["dataset_id"]
                print(f"[DEBUG] Found dataset_id: {dataset_id}")
            else:
                print(f"[DEBUG] Error: Failed to fetch datasets, status: {response.status_code}")
                return "<div style='color: red;'>❌ Failed to fetch datasets</div>"
        
        # 调用查询 API
        print(f"[DEBUG] Calling /query/web API - Dataset ID: {dataset_id}")
        async with httpx.AsyncClient(timeout=180.0) as client:  # 3分钟超时
            response = await client.post(
                f"{API_BASE_URL}/query/web",
                json={
                    "dataset_id": dataset_id,
                    "query": query,
                    "top_k": top_k
                }
            )
            
            print(f"[DEBUG] API response status: {response.status_code}")
            
            if response.status_code != 200:
                error = response.json().get("detail", "Unknown error")
                print(f"[DEBUG] API error: {error}")
                return f"<div style='color: red;'>❌ Query failed: {error}</div>"
            
            data = response.json()
            print(f"[DEBUG] Query successful, rendering results...")
            return _render_query_results(data, query)
    
    except Exception as e:
        print(f"[DEBUG] Exception in query_documents: {e}")
        import traceback
        traceback.print_exc()
        return f"<div style='color: red;'>❌ Error: {str(e)}</div>"


def _render_query_results(data: dict, query: str) -> str:
    """渲染查询结果为 HTML."""
    results = data.get("results", [])
    timing = data.get("timing", {})
    total_time = data.get("total_time", 0)
    
    if not results:
        return """
        <div style='text-align: center; padding: 40px; color: #888;'>
            <div style='font-size: 48px; margin-bottom: 16px;'>∅</div>
            <div>No results found</div>
        </div>
        """
    
    # 构建 HTML
    html = f"""
    <div style='font-family: Arial, sans-serif; line-height: 1.6;'>
        <div style='background: #f5f5f5; padding: 16px; border-radius: 8px; margin-bottom: 24px;'>
            <div style='font-weight: bold; margin-bottom: 8px;'>Query: {query}</div>
            <div style='font-size: 14px; color: #666;'>
                Found {len(results)} results in {total_time:.2f}s
                (Retrieval: {timing.get('retrieval', 0):.2f}s, 
                Reranking: {timing.get('reranking', 0):.2f}s, 
                Selector: {timing.get('selector', 0):.2f}s, 
                Analysis: {timing.get('analysis', 0):.2f}s)
            </div>
        </div>
    """
    
    for idx, result in enumerate(results, 1):
        analysis = result.get("analysis", {})
        metadata = result.get("metadata", {})
        
        is_relevant = analysis.get("is_relevant", True)
        confidence = analysis.get("confidence", "unknown")
        reason = analysis.get("reason", "No analysis")
        excerpts = analysis.get("relevant_excerpts", [])
        
        relevance_color = "#22c55e" if is_relevant else "#ef4444"
        relevance_text = "✓ Relevant" if is_relevant else "✗ Not Relevant"
        
        confidence_colors = {
            "high": "#22c55e",
            "medium": "#eab308",
            "low": "#ef4444",
            "unknown": "#888"
        }
        confidence_color = confidence_colors.get(confidence.lower(), "#888")
        
        html += f"""
        <div style='background: white; border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin-bottom: 16px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #eee;'>
                <span style='color: #666; font-size: 14px;'>Result {idx}</span>
                <span style='color: #666; font-size: 14px; font-family: monospace;'>Score: {result.get("score", 0):.4f}</span>
            </div>
        """
        
        # Metadata
        if metadata:
            html += "<div style='background: #f9f9f9; padding: 12px; border-radius: 6px; margin-bottom: 16px; font-size: 13px;'>"
            for key, value in metadata.items():
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                html += f"<div><span style='color: #666;'>{key}:</span> {value}</div>"
            html += "</div>"
        
        # Content - 使用 Markdown 渲染
        raw_content = result.get("content", "")
        if MARKDOWN_AVAILABLE:
            # 使用 Markdown 渲染，启用表格和代码高亮扩展
            formatted_content = markdown.markdown(
                raw_content,
                extensions=['tables', 'fenced_code', 'nl2br']
            )
        else:
            # 回退到简单的换行处理
            formatted_content = raw_content.replace("\n", "<br>")
        
        html += f"""
            <div style='background: #f9f9f9; padding: 16px; border-left: 3px solid #3b82f6; border-radius: 4px; margin-bottom: 16px;'>
                <div style='font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 8px;'>📄 Retrieved Content</div>
                <div style='overflow-x: auto;'>{formatted_content}</div>
            </div>
        """
        
        # Analysis
        html += f"""
            <div style='background: #f9f9f9; padding: 16px; border: 1px dashed #ddd; border-radius: 6px;'>
                <div style='font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 12px;'>🤖 GPT Quality Assessment</div>
                <div style='margin-bottom: 8px;'>
                    <span style='color: #666;'>Relevance:</span>
                    <span style='background: {relevance_color}20; color: {relevance_color}; padding: 4px 12px; border-radius: 4px; font-size: 13px; margin-left: 8px;'>{relevance_text}</span>
                </div>
                <div style='margin-bottom: 8px;'>
                    <span style='color: #666;'>Confidence:</span>
                    <span style='background: {confidence_color}20; color: {confidence_color}; padding: 4px 12px; border-radius: 4px; font-size: 13px; margin-left: 8px;'>{confidence.title()}</span>
                </div>
                <div style='margin-bottom: 8px;'>
                    <span style='color: #666;'>Reason:</span> {reason}
                </div>
        """
        
        # Excerpts
        if excerpts:
            html += "<div style='margin-top: 12px;'><div style='color: #666; font-size: 13px; margin-bottom: 6px;'>Key Excerpts:</div>"
            for excerpt in excerpts:
                truncated = excerpt[:150] + "..." if len(excerpt) > 150 else excerpt
                html += f"<div style='padding: 6px 12px; background: white; border-left: 2px solid #ddd; border-radius: 4px; margin-bottom: 4px; font-size: 13px;'>{truncated}</div>"
            html += "</div>"
        
        html += """
                <div style='margin-top: 12px; font-size: 11px; color: #999; font-style: italic;'>
                    Note: This analysis is for reference only and not part of RAG output
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html


# ========== Gradio Interface ==========

def create_gradio_ui():
    """创建 Gradio UI."""
    
    with gr.Blocks(
        title="RAG Document Processor",
        analytics_enabled=False,  # Disable analytics to reduce noise
        delete_cache=(3600, 86400)  # Clean cache: check every 1 hour, delete files older than 24 hours
    ) as demo:
        gr.Markdown("# RAG Document Processing UI")
        gr.Markdown("Upload PDF documents and monitor worker processing status")
        
        # 状态存储
        current_task_id = gr.State(value="")
        
        # ========== Tabs ==========
        with gr.Tabs():
            # Tab 1: Document Processing
            with gr.Tab("Document Processing"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. Select or Create Dataset")
                        
                        # Dataset 选择
                        with gr.Tab("Use Existing"):
                            dataset_dropdown = gr.Dropdown(
                                label="Select Dataset",
                                choices=[],
                                interactive=True
                            )
                            refresh_datasets_btn = gr.Button("Refresh List", size="sm")
                        
                        with gr.Tab("Create New"):
                            new_dataset_name = gr.Textbox(
                                label="Dataset Name",
                                placeholder="e.g., mortgage_guidelines"
                            )
                            new_dataset_desc = gr.Textbox(
                                label="Description (Optional)",
                                placeholder="e.g., Mortgage lending guidelines"
                            )
                        
                        gr.Markdown("### 2. Upload PDF")
                        file_upload = gr.File(
                            label="PDF File",
                            file_types=[".pdf"],
                            type="filepath"
                        )
                        
                        submit_btn = gr.Button("Submit Task", variant="primary", size="lg")
                        submit_status = gr.Textbox(label="Submission Status", interactive=False)
                        submit_details = gr.Textbox(label="Details", interactive=False, lines=3)
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Task Status")
                        
                        task_id_input = gr.Textbox(
                            label="Task ID",
                            placeholder="Automatically filled after submission",
                            interactive=True
                        )
                        
                        with gr.Row():
                            refresh_status_btn = gr.Button("Refresh Status", variant="secondary")
                        
                        status_text = gr.Textbox(label="Status", interactive=False)
                        progress_text = gr.Textbox(label="Progress", interactive=False)
                        units_text = gr.Textbox(label="Units Created", interactive=False)
                        error_text = gr.Textbox(label="Error", interactive=False, lines=2)
                        details_text = gr.Textbox(label="Task Details", interactive=False, lines=4)
                        
                        gr.Markdown("### Recent Tasks")
                        history_table = gr.Dataframe(
                            headers=["Task ID", "Status", "Progress", "Created"],
                            datatype=["str", "str", "str", "str"],
                            interactive=False,
                            wrap=True
                        )
                        refresh_history_btn = gr.Button("Refresh History", size="sm")
            
            # Tab 2: Query
            with gr.Tab("Query"):
                gr.Markdown("### Query Documents")
                gr.Markdown("⚠️ This feature uses LLM-based analysis and may take ~20 seconds")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        query_dataset_dropdown = gr.Dropdown(
                            label="Select Dataset",
                            choices=[],
                            interactive=True
                        )
                        refresh_query_datasets_btn = gr.Button("Refresh Datasets", size="sm")
                        
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="e.g., What are the income verification requirements?",
                            lines=3
                        )
                        
                        query_top_k = gr.Slider(
                            label="Number of Results",
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1
                        )
                        
                        query_btn = gr.Button("Query (may take ~20s)", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        query_result_html = gr.HTML(
                            label="Results",
                            value="<div style='color: #999; padding: 40px; text-align: center;'>Click 'Query' button to search</div>"
                        )
            
            # Tab 3: Configuration
            with gr.Tab("Configuration"):
                gr.Markdown("### Current Configuration (Sensitive values are masked)")
                config_table = gr.Dataframe(
                    headers=["Category", "Key", "Value"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    value=get_config_info(),
                    wrap=True
                )
                gr.Markdown("*API Keys and Secrets show only first 4 and last 4 characters*")
        
        # ========== Event Handlers ==========
        
        # 刷新 dataset 列表
        def refresh_datasets():
            datasets = asyncio.run(list_datasets())
            return gr.Dropdown(choices=datasets)
        
        refresh_datasets_btn.click(
            fn=refresh_datasets,
            outputs=dataset_dropdown
        )
        
        # 页面加载时自动刷新
        demo.load(
            fn=refresh_datasets,
            outputs=dataset_dropdown
        )
        
        # 提交任务
        def handle_submit(dataset_name, file_path, new_name, new_desc):
            # 如果填写了新 dataset 名称，忽略 dropdown 的值
            if new_name and new_name.strip():
                dataset_name = None  # 清空，强制使用 new_name
            
            status, details = asyncio.run(
                submit_task(dataset_name, file_path, new_name, new_desc)
            )
            
            # 提取 task_id
            task_id = ""
            if "Task ID:" in details:
                task_id = details.split("Task ID:")[1].split("\n")[0].strip()
            
            return status, details, task_id
        
        submit_btn.click(
            fn=handle_submit,
            inputs=[dataset_dropdown, file_upload, new_dataset_name, new_dataset_desc],
            outputs=[submit_status, submit_details, task_id_input]
        )
        
        # 刷新任务状态
        def handle_refresh_status(task_id):
            return asyncio.run(get_task_status(task_id))
        
        refresh_status_btn.click(
            fn=handle_refresh_status,
            inputs=task_id_input,
            outputs=[status_text, progress_text, units_text, error_text, details_text]
        )
        
        # 刷新任务历史
        def handle_refresh_history():
            try:
                return asyncio.run(get_recent_tasks(5))
            except Exception as e:
                return [["Error", str(e), "-", "-"]]
        
        refresh_history_btn.click(
            fn=handle_refresh_history,
            outputs=history_table
        )
        
        # 页面加载时自动刷新 dataset 列表和历史
        def on_page_load():
            """页面加载时的初始化."""
            try:
                datasets = asyncio.run(list_datasets())
                history = asyncio.run(get_recent_tasks(5))
                return gr.Dropdown(choices=datasets), history
            except Exception as e:
                # 如果加载失败，返回友好提示
                return gr.Dropdown(choices=[]), [["Error", str(e), "-", "-"]]
        
        demo.load(
            fn=on_page_load,
            outputs=[dataset_dropdown, history_table]
        )
        
        # ========== Query Tab Event Handlers ==========
        
        # 刷新 query 页面的 dataset 列表
        def refresh_query_datasets():
            datasets = asyncio.run(list_datasets())
            return gr.Dropdown(choices=datasets)
        
        refresh_query_datasets_btn.click(
            fn=refresh_query_datasets,
            outputs=query_dataset_dropdown
        )
        
        # Query 页面加载时自动刷新 dataset 列表
        demo.load(
            fn=refresh_query_datasets,
            outputs=query_dataset_dropdown
        )
        
        # 执行查询
        def handle_query(dataset_name, query, top_k):
            print(f"[DEBUG] Query started - Dataset: {dataset_name}, Query: {query[:50]}..., Top-K: {top_k}")
            
            # 先返回 loading 状态（立即显示）
            yield "<div style='text-align: center; padding: 60px; color: #666;'><div style='font-size: 24px; margin-bottom: 12px;'>🔍 Querying...</div><div>Please wait ~20 seconds</div></div>"
            
            # 执行实际查询
            result = asyncio.run(query_documents(dataset_name, query, int(top_k)))
            print(f"[DEBUG] Query completed - Result length: {len(result)}")
            
            # 返回最终结果
            yield result
        
        query_btn.click(
            fn=handle_query,
            inputs=[query_dataset_dropdown, query_input, query_top_k],
            outputs=query_result_html
        )
    
    return demo


# 导出 Gradio app
gradio_app = create_gradio_ui()
