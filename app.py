"""
AI大模型面试模拟Agent - Gradio Web界面
"""
import os
import sys
import gradio as gr

# 确保项目根目录在PATH中
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.agent.interviewer import InterviewerAgent
from src.resume.parser import parse_resume_from_bytes
from src.agent.evaluator import save_interview_record, save_report_markdown, save_report_docx
from src.rag.ingest import ingest_pdfs
from src.rag.retriever import get_retriever

# ===== 启动时初始化知识库 =====
def _init_knowledge_base():
    """启动时检查并构建知识库（HuggingFace Spaces 冷启动用）"""
    import os
    from config import CHROMA_PERSIST_DIR, PDF_DIR

    # 检查是否已有知识库
    db_file = os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3")
    if os.path.exists(db_file) and os.path.getsize(db_file) > 1000:
        print(f"[Init] 知识库已存在，跳过构建")
        return

    print("[Init] 检测到知识库不存在，开始构建...")
    print(f"[Init] PDF 目录: {PDF_DIR}")

    try:
        stats = ingest_pdfs(force_rebuild=False)
        print(f"[Init] 知识库构建完成: {stats['total_chunks']} 个文本块")
    except Exception as e:
        print(f"[Init] 知识库构建失败: {e}")
        import traceback
        traceback.print_exc()


# 会话管理：按 session_id 保存各自的 Agent 实例
_agents: dict = {}
_kb_initialized = False


def _ensure_kb():
    """确保知识库已初始化（延迟初始化，首次访问时构建）"""
    global _kb_initialized
    if not _kb_initialized:
        _init_knowledge_base()
        _kb_initialized = True


def get_agent(session_id: str) -> InterviewerAgent:
    """根据 session_id 获取或创建 Agent 实例（会话隔离）"""
    _ensure_kb()  # 首次访问时初始化知识库
    if session_id not in _agents:
        _agents[session_id] = InterviewerAgent()
    return _agents[session_id]


def remove_agent(session_id: str):
    """移除会话（释放资源）"""
    _agents.pop(session_id, None)

# 面试方向选项
CATEGORY_OPTIONS = ["全部", "RAG", "微调", "分布式训练", "基础理论", "推理优化", "工程实践", "预训练", "评测"]
DIFFICULTY_OPTIONS = ["初级", "中级", "高级"]
COMPANY_LEVEL_OPTIONS = ["大厂", "中厂", "小厂"]


def check_kb_status():
    """检查知识库状态（不加载模型，仅检查文件）"""
    import os
    from config import CHROMA_PERSIST_DIR

    db_file = os.path.join(CHROMA_PERSIST_DIR, "chroma.sqlite3")
    if os.path.exists(db_file):
        size_kb = os.path.getsize(db_file) / 1024
        if size_kb > 100:
            return f"知识库已就绪（{size_kb:.0f} KB），点击开始面试后加载模型"
        return f"知识库文件已存在（{size_kb:.0f} KB）"
    return "知识库未构建，首次开始面试时自动构建（约2-5分钟）"


def build_kb(progress=gr.Progress()):
    """构建知识库"""
    progress(0, desc="开始构建知识库...")
    try:
        stats = ingest_pdfs(force_rebuild=False)
        result_msg = (
            f"构建完成!\n"
            f"总PDF数: {stats['total_pdfs']}\n"
            f"新处理: {stats['new_processed']}\n"
            f"跳过(未变更): {stats['skipped']}\n"
            f"总文本块: {stats['total_chunks']}"
        )
        progress(1, desc="完成")
        return result_msg
    except Exception as e:
        return f"构建失败: {e}"


def _extract_file_bytes(file_input):
    """从Gradio File组件返回值中提取文件字节和文件名"""
    import os

    if file_input is None:
        return None, None

    # 格式1: 字符串路径
    if isinstance(file_input, str):
        if os.path.isfile(file_input):
            filename = os.path.basename(file_input)
            with open(file_input, "rb") as f:
                return f.read(), filename
        return None, None

    # 格式2: 列表（多文件）
    if isinstance(file_input, list):
        if len(file_input) == 0:
            return None, None
        return _extract_file_bytes(file_input[0])

    # 格式3: 字典
    if isinstance(file_input, dict):
        if "path" in file_input and os.path.isfile(file_input["path"]):
            with open(file_input["path"], "rb") as f:
                return f.read(), file_input.get("orig_name", os.path.basename(file_input["path"]))
        if "name" in file_input and "data" in file_input:
            data = file_input["data"]
            if isinstance(data, (bytes, bytearray)):
                return bytes(data), file_input["name"]
        return None, None

    # 格式4: 文件对象
    if hasattr(file_input, 'read'):
        if hasattr(file_input, 'seek'):
            file_input.seek(0)
        data = file_input.read()
        name = getattr(file_input, 'name', 'resume.pdf')
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), name

    return None, None


def start_interview(company_level, difficulty, focus_category, resume_file, request: gr.Request):
    """开始面试（会话隔离）"""
    session_id = request.session_hash
    agent = get_agent(session_id)

    resume_info = None

    if resume_file is not None:
        try:
            file_bytes, filename = _extract_file_bytes(resume_file)
            if file_bytes:
                resume_info = parse_resume_from_bytes(file_bytes, filename)
        except Exception as e:
            print(f"Resume parse failed: {e}")
            import traceback
            traceback.print_exc()

    # 处理分类选择
    category = None if focus_category == "全部" else focus_category

    try:
        greeting = agent.start_interview(
            company_level=company_level,
            difficulty=difficulty,
            focus_category=category,
            resume_info=resume_info,
        )
        progress = agent.get_progress()
        return (
            [{"role": "assistant", "content": greeting}],
            progress,
            gr.DownloadButton(interactive=False),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (
            [{"role": "assistant", "content": f"启动面试失败: {e}"}],
            "启动失败",
            gr.DownloadButton(interactive=False),
        )


def respond(message, chat_history, request: gr.Request):
    """响应用户输入（会话隔离）"""
    session_id = request.session_hash
    agent = get_agent(session_id)

    if not message.strip():
        return "", chat_history, agent.get_progress()

    # 添加用户消息
    chat_history.append({"role": "user", "content": message})

    try:
        bot_message = agent.chat(message)
        chat_history.append({"role": "assistant", "content": bot_message})
        progress = agent.get_progress()
        return "", chat_history, progress
    except Exception as e:
        import traceback
        traceback.print_exc()
        chat_history.append({"role": "assistant", "content": f"[错误] {e}"})
        return "", chat_history, agent.get_progress()


def end_interview(chat_history, request: gr.Request):
    """结束面试并生成报告（会话隔离）"""
    session_id = request.session_hash
    agent = get_agent(session_id)

    docx_path = None
    try:
        closing, report = agent.end_interview()

        # 添加结束语到聊天历史
        chat_history.append({"role": "assistant", "content": closing.strip()})

        # 保存记录和报告
        save_msg = ""
        try:
            save_interview_record(
                agent.state.conversation_history,
                report,
                metadata=agent.get_progress(),
            )
            save_report_markdown(report)
            docx_path = save_report_docx(report)
            save_msg = f"\n\n> 📄 Word版报告已生成，点击下方按钮下载"
        except Exception as e:
            import traceback
            traceback.print_exc()
            save_msg = f"\n\n> (报告保存失败: {e})"

        return (
            chat_history,
            report + save_msg,
            "面试已结束",
            gr.DownloadButton(value=docx_path, interactive=True),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return chat_history, f"生成报告失败: {e}", "错误", gr.DownloadButton(interactive=False)


def reset_interview(request: gr.Request):
    """重置面试（会话隔离）"""
    session_id = request.session_hash
    agent = get_agent(session_id)
    agent.state.reset()
    return [], "面试结束后会在这里显示评估报告", "面试未开始", gr.DownloadButton(interactive=False)


# ============== Gradio界面 ==============

with gr.Blocks(title="AI大模型面试模拟") as demo:
    gr.Markdown("""
    # 🤖 AI大模型面试模拟Agent

    基于RAG的大模型应用开发技术面试模拟系统，面试官由AI扮演，题目来自53份精选面试题资料。

    **使用说明**:
    1. （首次使用）先点击"构建知识库"，将PDF面试题导入向量库
    2. 选择公司级别（大厂/中厂/小厂）和难度
    3. 上传你的简历（可选，面试官会针对性提问）
    4. 点击"开始面试"
    5. 面试结束后自动生成评估报告（含Word版下载）
    """)

    with gr.Row():
        # 左侧控制面板
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 控制面板")

            # 知识库管理
            with gr.Accordion("📚 知识库管理", open=False):
                kb_status = gr.Textbox(label="知识库状态", value=check_kb_status(), interactive=False)
                build_btn = gr.Button("🔄 构建/更新知识库", variant="secondary")
                build_output = gr.Textbox(label="构建结果", lines=5, interactive=False)
                build_btn.click(build_kb, outputs=build_output).then(
                    lambda: check_kb_status(), outputs=kb_status
                )

            # 面试设置
            gr.Markdown("### 🎯 面试设置")

            company_level = gr.Dropdown(
                choices=COMPANY_LEVEL_OPTIONS,
                value="中厂",
                label="公司级别",
                info="大厂3轮/中厂2轮/小厂1轮",
            )
            difficulty = gr.Dropdown(
                choices=DIFFICULTY_OPTIONS,
                value="中级",
                label="面试难度",
            )
            focus_category = gr.Dropdown(
                choices=CATEGORY_OPTIONS,
                value="全部",
                label="考察方向",
            )
            resume_file = gr.File(
                label="上传简历 (PDF/DOCX)",
                file_types=[".pdf", ".docx"],
            )

            # 进度显示
            progress_text = gr.Textbox(
                label="面试进度",
                value="面试未开始",
                interactive=False,
            )

            with gr.Row():
                start_btn = gr.Button("▶️ 开始面试", variant="primary")
                end_btn = gr.Button("⏹️ 结束面试", variant="stop")

            reset_btn = gr.Button("🔄 重新开始", variant="secondary")

        # 右侧聊天区域
        with gr.Column(scale=2):
            gr.Markdown("### 💬 面试对话")
            chatbot = gr.Chatbot(
                height=500,
                label="面试官",
                show_label=True,
            )
            msg = gr.Textbox(
                label="输入你的回答",
                placeholder="在这里输入你的回答，按回车发送...",
                lines=3,
            )

            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("清空输入", scale=1)

    # 评估报告区域
    with gr.Accordion("📊 面试评估报告", open=False):
        report_output = gr.Markdown(value="面试结束后会在这里显示评估报告")
        download_btn = gr.DownloadButton(
            "📥 下载 Word 版报告",
            variant="primary",
            interactive=False,
        )

    # ============== 事件绑定 ==============

    # 开始面试
    start_btn.click(
        start_interview,
        inputs=[company_level, difficulty, focus_category, resume_file],
        outputs=[chatbot, progress_text, download_btn],
    )

    # 发送消息
    submit_btn.click(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, progress_text],
    )

    # 回车发送
    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, progress_text],
    )

    # 清空输入
    clear_btn.click(lambda: "", outputs=msg)

    # 结束面试
    end_btn.click(
        end_interview,
        inputs=[chatbot],
        outputs=[chatbot, report_output, progress_text, download_btn],
    )

    # 重新开始
    reset_btn.click(
        reset_interview,
        outputs=[chatbot, report_output, progress_text, download_btn],
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  AI大模型面试模拟Agent 启动中...")
    print("=" * 60)

    # 排队机制：限制并发，避免免费版 CPU 内存溢出
    demo.queue(
        default_concurrency_limit=int(os.getenv("GRADIO_CONCURRENCY", "10")),
        max_size=int(os.getenv("GRADIO_QUEUE_SIZE", "100")),
    )

    # HuggingFace Spaces 会自动注入 GRADIO_SERVER_NAME / GRADIO_SERVER_PORT
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    # 本地运行自动开浏览器，HF Spaces 上不开
    inbrowser = os.getenv("GRADIO_IN_BROWSER", "0") == "1"

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        inbrowser=inbrowser,
        theme=gr.themes.Soft(),
    )
