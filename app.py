"""
Streamlit 前端应用：基于 RAG 的多模态领事服务与外交礼仪智能助手
优化：@st.cache_resource 全局缓存模型，防止多用户访问内存溢出
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便导入 rag_system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from datetime import datetime
from rag_system import process_query


# ========== 全局模型缓存（@st.cache_resource 防止多用户重复加载爆内存）==========
@st.cache_resource(show_spinner=False, max_entries=1, ttl=None)
def get_cached_ocr_reader():
    """全局缓存 EasyOCR 阅读器，避免多用户重复加载 PyTorch 模型"""
    import easyocr

    ocr_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".easyocr_model"
    )
    os.makedirs(ocr_model_dir, exist_ok=True)

    return easyocr.Reader(
        ['ch_sim', 'en'],
        model_storage_directory=ocr_model_dir,
        user_network_directory=ocr_model_dir,
    )


@st.cache_resource(show_spinner=False, max_entries=1, ttl=None)
def get_cached_embeddings():
    """全局缓存 HuggingFace Embedding 模型"""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
    )


@st.cache_resource(show_spinner=False, max_entries=1, ttl=None)
def get_cached_vector_db(_embeddings):
    """全局缓存 Chroma 向量数据库连接（复用已加载的 embedding）"""
    from langchain_community.vectorstores import Chroma

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=_embeddings,
    )


def generate_report_text(user_question, ocr_result, llm_answer, context_text):
    """生成研判报告文本字符串"""
    lines = []
    lines.append("=" * 60)
    lines.append("外交领事智能研判报告")
    lines.append("=" * 60)
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 用户问题
    lines.append("【用户问题】")
    lines.append("-" * 40)
    lines.append(user_question if user_question else "（无文字提问）")
    lines.append("")

    # OCR 识别结果
    if ocr_result and ocr_result.get("text"):
        lines.append("【OCR 图片识别文字】")
        lines.append("-" * 40)
        lines.append(f"识别状态：{ocr_result.get('message', '')}")
        lines.append("")
        lines.append(ocr_result.get("text", ""))
        lines.append("")

    # 智能解答
    if llm_answer:
        lines.append("【智能解答内容】")
        lines.append("-" * 40)
        lines.append(llm_answer)
        lines.append("")

    # 检索依据
    if context_text:
        lines.append("【官方检索依据】")
        lines.append("-" * 40)
        lines.append(context_text)
        lines.append("")

    lines.append("=" * 60)
    lines.append("本报告由「基于 RAG 的多模态领事服务与外交礼仪智能助手」生成")
    lines.append("仅供参考，具体领事事务请联系当地使领馆或外交部全球热线 12308")
    lines.append("=" * 60)

    return "\n".join(lines)

# ========== 页面配置 ==========
st.set_page_config(
    page_title="全球涉外领事服务与外交礼仪智能研判平台",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "全球涉外领事服务与外交礼仪智能研判平台 - 人工智能导论期末作品"
    }
)

# ========== 自定义 CSS 样式 ==========
def load_custom_css():
    """加载自定义 CSS 样式：高端外交智库主题"""
    st.markdown("""
    <style>
        /* ===== 主色调：外交深蓝 ===== */
        :root {
            --diplo-primary: #1A365D;
            --diplo-secondary: #2A4A7F;
            --diplo-accent: #C9A962;
            --diplo-bg: #F8F9FA;
            --diplo-card-bg: #FFFFFF;
            --diplo-text: #2D3748;
            --diplo-text-light: #718096;
        }

        /* ===== 全局字体与背景 ===== */
        .stApp {
            background-color: var(--diplo-bg);
            font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
            color: var(--diplo-text);
        }

        .main .block-container {
            max-width: 1400px;
            padding-top: 1rem;
        }

        /* ===== Header 区域 ===== */
        .diplo-header {
            background: linear-gradient(135deg, #1A365D 0%, #2A4A7F 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(26, 54, 93, 0.15);
            position: relative;
            overflow: hidden;
        }

        .diplo-header::before {
            content: "";
            position: absolute;
            top: 0; right: 0;
            width: 200px; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(201, 169, 98, 0.08));
        }

        .diplo-header h1 {
            font-size: 2.2rem;
            font-weight: 800;
            margin: 0 0 0.5rem 0;
            letter-spacing: 1px;
            background: linear-gradient(90deg, #93C5FD 0%, #60A5FA 50%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .diplo-header p {
            color: rgba(255, 255, 255, 0.75);
            font-size: 0.95rem;
            margin: 0;
            font-weight: 300;
            letter-spacing: 0.5px;
        }

        .diplo-header .accent-line {
            width: 60px; height: 3px;
            background: var(--diplo-accent);
            border-radius: 2px;
            margin-top: 0.8rem;
        }

        /* ===== 卡片样式 ===== */
        .diplo-card {
            background: var(--diplo-card-bg);
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transition: box-shadow 0.3s ease;
        }

        .diplo-card:hover {
            box-shadow: 0 4px 16px rgba(26, 54, 93, 0.1);
        }

        .diplo-card-title {
            color: var(--diplo-primary);
            font-size: 1.1rem;
            font-weight: 700;
            border-left: 4px solid var(--diplo-accent);
            padding-left: 0.75rem;
            margin-bottom: 1rem;
        }

        /* ===== OCR 卡片 ===== */
        .ocr-card {
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin: 0.8rem 0;
            border-left: 4px solid;
        }

        /* ===== 按钮样式 ===== */
        .stButton > button {
            background: linear-gradient(180deg, #2A4A7F 0%, #1A365D 100%);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 1px;
            box-shadow: 0 2px 8px rgba(26, 54, 93, 0.25);
            transition: all 0.3s ease;
            padding: 0.6rem 1.5rem;
        }

        .stButton > button:hover {
            background: linear-gradient(180deg, #3A5A8F 0%, #2A4A7F 100%);
            box-shadow: 0 4px 16px rgba(26, 54, 93, 0.35);
            transform: translateY(-1px);
        }

        .stButton > button:active {
            transform: translateY(0);
        }

        /* ===== 侧边栏 ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1A365D 0%, #2A4A7F 100%);
        }

        section[data-testid="stSidebar"] * {
            color: rgba(255, 255, 255, 0.9) !important;
        }

        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(201, 169, 98, 0.4);
            color: #FFFFFF;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(201, 169, 98, 0.25);
            border-color: var(--diplo-accent);
        }

        /* ===== 输入框 ===== */
        .stTextArea textarea {
            border: 1px solid #CBD5E0;
            border-radius: 8px;
            font-size: 15px;
            background: #FFFFFF;
            transition: border-color 0.2s ease;
        }

        .stTextArea textarea:focus {
            border-color: var(--diplo-primary);
            box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
        }

        /* ===== 文件上传器 ===== */
        .stFileUploader {
            border: 2px dashed #CBD5E0;
            border-radius: 12px;
            padding: 0.8rem;
            background: #FFFFFF;
        }

        /* ===== Tabs 样式 ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            background: #FFFFFF;
            border-radius: 10px 10px 0 0;
            padding: 0 0.5rem;
            border: 1px solid #E2E8F0;
            border-bottom: none;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            color: var(--diplo-text-light);
            border-radius: 8px 8px 0 0;
        }

        .stTabs [aria-selected="true"] {
            color: var(--diplo-primary) !important;
            border-bottom: 3px solid var(--diplo-accent);
        }

        .stTabs [data-baseweb="tab-border-bottom"] {
            display: none;
        }

        /* ===== 分隔线 ===== */
        hr {
            border-color: #E2E8F0;
            margin: 1rem 0;
        }

        /* ===== 底部说明 ===== */
        .diplo-footer {
            text-align: center;
            color: var(--diplo-text-light);
            font-size: 12px;
            padding: 1.5rem;
            border-top: 1px solid #E2E8F0;
            margin-top: 2rem;
        }

        /* ===== Markdown 内容排版 ===== */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--diplo-primary);
        }

        .stMarkdown blockquote {
            border-left: 3px solid var(--diplo-accent);
            background: #FFFCF5;
            padding: 0.5rem 1rem;
            border-radius: 0 6px 6px 0;
            margin: 0.5rem 0;
        }

        .stMarkdown code {
            background: #EDF2F7;
            color: var(--diplo-primary);
            border-radius: 4px;
            padding: 0.1rem 0.3rem;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """渲染页面 Header"""
    st.markdown("""
    <div class="diplo-header">
        <h1>🌐 全球涉外领事服务与外交礼仪智能研判平台</h1>
        <p>结合多模态 CV 视觉识别与垂直领域 RAG 知识库 · 人工智能导论期末演示作品</p>
        <div class="accent-line"></div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """渲染侧边栏：快捷问题 + 使用说明"""
    with st.sidebar:
        st.markdown("### 🏛️ 快捷测试问题")
        st.markdown("点击下方按钮快速体验：")

        # 预设示例问题
        example_questions = [
            "🇯🇵 在日本遭遇突发地震怎么办？",
            "👔 涉外商务拜访有什么服装礼仪规范？",
            "🛂 海外护照过期如何申请紧急旅行证？"
        ]

        # 点击示例按钮：填充文本框并自动触发研判
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.user_input = question
                st.session_state.trigger_query = True
                st.rerun()

        st.markdown("---")

        # 使用说明
        with st.expander("ℹ️ 使用说明"):
            st.markdown("""
            **操作步骤：**
            1. 在左侧输入区填写您的问题，或在此选择预设问题
            2. 可选：上传证件图片进行 OCR 识别
            3. 点击「🚀 提交研判」按钮
            4. 在右侧标签页查看智能解答与权威依据

            **支持功能：**
            - 🛂 领事保护与协助咨询
            - 📜 外交礼仪与涉外规范
            - 💡 证件 OCR 识别与解读
            """)


def save_uploaded_file(uploaded_file):
    """将上传的文件保存到临时目录"""
    if uploaded_file is None:
        return None

    # 创建临时文件
    temp_dir = tempfile.mkdtemp()
    file_path = Path(temp_dir) / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(file_path)


def render_ocr_result(ocr_result):
    """渲染 OCR 识别结果"""
    if not ocr_result:
        return

    success = ocr_result.get("success", False)
    has_text = ocr_result.get("has_text", False)
    message = ocr_result.get("message", "")
    text = ocr_result.get("text", "")

    # 根据状态选择图标和样式
    if not success:
        icon = "❌"
        border_color = "#E53E3E"
        bg_color = "#FFF5F5"
    elif not has_text:
        icon = "⚠️"
        border_color = "#D69E2E"
        bg_color = "#FFF8E1"
    else:
        icon = "📷"
        border_color = "#C9A962"
        bg_color = "#FEFCF5"

    st.markdown(f"""
    <div class="ocr-card" style="background: {bg_color}; border-left: 4px solid {border_color};">
        <h4 style="color: {border_color}; margin-top: 0;">{icon} OCR 识别结果</h4>
        <p style="color: #555; margin-bottom: 0.5rem;">{message}</p>
    """, unsafe_allow_html=True)

    if text:
        st.markdown(f"""
        <div style="background: white; border: 1px dashed #C9A962; padding: 1rem;
                    border-radius: 6px; font-family: monospace; font-size: 14px;
                    white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto;">
{text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """主函数"""
    load_custom_css()
    render_header()

    # 侧边栏
    render_sidebar()

    # 初始化 session_state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "trigger_query" not in st.session_state:
        st.session_state.trigger_query = False
    if "last_report_data" not in st.session_state:
        st.session_state.last_report_data = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # ========== 左右分栏布局 ==========
    col_left, col_right = st.columns([2, 3], gap="large")

    # ===== 左侧：输入与交互区（40%）=====
    with col_left:
        st.markdown("""
        <div class="diplo-card">
            <div class="diplo-card-title">📝 信息输入与证件上传</div>
        </div>
        """, unsafe_allow_html=True)

        # 证件图片上传
        st.markdown("📷 **上传证件图片（可选）**")
        uploaded_file = st.file_uploader(
            label="上传护照、签证、告示等证件图片",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
            help="支持格式：JPG、PNG、BMP、TIFF、WebP，单张不超过 20MB，建议图片清晰、文字正面朝上",
            label_visibility="collapsed"
        )

        # ===== 新增：上传后即时显示图片预览 + 大小提示 =====
        if uploaded_file is not None:
            try:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                size_color = "#38A169" if file_size_mb <= 5 else "#D69E2E" if file_size_mb <= 15 else "#E53E3E"
                st.markdown(
                    f"<div style='font-size: 12px; color: {size_color}; margin-bottom: 6px;'>"
                    f"📎 已上传：{uploaded_file.name} · {file_size_mb:.2f} MB</div>",
                    unsafe_allow_html=True
                )
                # 显示缩略图预览（固定宽度，不占太多空间）
                st.image(
                    uploaded_file,
                    caption="图片预览（点击放大）",
                    use_column_width="auto",
                    width=280,
                    clamp=True,
                    output_format="JPEG",
                )
                # 重置文件指针，供后续 save_uploaded_file 使用
                uploaded_file.seek(0)
            except Exception as preview_err:  # noqa: BLE001
                st.warning(f"⚠️ 图片预览失败：{preview_err}，但仍可提交识别。")
                uploaded_file.seek(0)

        st.markdown("---")

        # 文字提问框
        st.markdown("💬 **输入您的问题**")
        user_input = st.text_area(
            label="请输入您关于领事保护、外交礼仪、涉外事务等方面的问题：",
            key="user_input",
            height=120,
            placeholder="例如：在日本遇到突发地震，如何寻求领事保护？",
            label_visibility="collapsed"
        )

        # 提交按钮
        st.markdown("")
        submit_button = st.button("🚀 提交研判", type="primary", use_container_width=True)

    # ===== 右侧：智能研判与依据区（60%）=====
    with col_right:
        # 创建 Tabs
        tab_answer, tab_evidence = st.tabs(["🤖 智能研判解答", "🔍 调用的权威法规与依据"])

    # ========== 处理提交逻辑 ==========
    should_run = submit_button or st.session_state.get("trigger_query", False)
    if st.session_state.get("trigger_query"):
        st.session_state.trigger_query = False

    # 结果渲染函数（供新查询和恢复显示共用）
    def render_result(result, question_text):
        """渲染研判结果到右侧 Tabs，并保存到 session_state"""
        # ===== Tab 1: 智能研判解答 =====
        with tab_answer:
            # OCR 结果（如有）
            render_ocr_result(result.get("ocr_result"))

            # 错误处理
            if result.get("error"):
                st.error(f"⚠️ 处理提示\n\n{result['error']}")
                st.session_state.last_report_data = None
                st.session_state.last_result = None
                return

            # LLM 回答
            answer = result.get("llm_answer", "")
            if answer:
                st.markdown(f"""
                <div class="diplo-card">
                    <div class="diplo-card-title">🤖 智能助手解答</div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(answer)
            else:
                st.info("暂无解答内容，请提交问题后查看。")

        # ===== Tab 2: 权威依据 =====
        with tab_evidence:
            context = result.get("context_text", "")
            if context:
                st.markdown("""
                <div style="color: #1A365D; font-weight: 600; margin-bottom: 0.8rem; font-size: 0.95rem;">
                    📚 以下内容包含：<b>🌐 网络搜索依据</b>（主要信息来源）与 <b>📚 知识库校验</b>（纠正幻觉）：
                </div>
                """, unsafe_allow_html=True)
                st.markdown(context)
            else:
                st.info("暂无检索依据，请提交问题后查看。")

        # 生成并保存报告文本（用于下载）
        llm_ans = result.get("llm_answer", "")
        ctx_text = result.get("context_text", "")
        if llm_ans or ctx_text:
            report_text = generate_report_text(
                user_question=question_text,
                ocr_result=result.get("ocr_result"),
                llm_answer=llm_ans,
                context_text=ctx_text,
            )
            st.session_state.last_report_data = report_text
            st.session_state.last_result = result
        else:
            st.session_state.last_report_data = None
            st.session_state.last_result = None

    # 空状态提示（无结果时在右侧 Tabs 显示占位）
    if not should_run and not st.session_state.get("last_result"):
        with tab_answer:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #A0AEC0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🏛️</div>
                <p style="font-size: 1.1rem; font-weight: 500;">智能研判平台已就绪</p>
                <p style="font-size: 0.9rem;">请在左侧输入问题或上传证件图片，点击「🚀 提交研判」开始</p>
            </div>
            """, unsafe_allow_html=True)
        with tab_evidence:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #A0AEC0;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📜</div>
                <p style="font-size: 0.9rem;">检索依据将在提交研判后展示于此</p>
            </div>
            """, unsafe_allow_html=True)

    if should_run:
        if not user_input and not uploaded_file:
            with tab_answer:
                st.warning("⚠️ 请输入提问文本或上传证件图片后再提交。")
        else:
            # ===== 分步骤进度展示容器 =====
            with tab_answer:
                progress_placeholder = st.empty()
                status_box = progress_placeholder.container()
                with status_box:
                    status_1 = st.status("📦 初始化资源与上传处理...", expanded=True)

            try:
                # ===== 步骤 1：保存上传文件 =====
                with status_1:
                    st.write("📥 保存上传图片到临时目录...")
                image_path = save_uploaded_file(uploaded_file)

                # ===== 步骤 2：预加载全局缓存的模型 =====
                with status_1:
                    st.write("🧠 加载 AI 模型（首次使用需等待几秒钟）...")
                cached_resources = {}
                try:
                    # 顺序加载，避免同时加载爆内存
                    cached_resources["ocr_reader"] = get_cached_ocr_reader()
                    cached_resources["embeddings"] = get_cached_embeddings()
                    cached_resources["vector_db"] = get_cached_vector_db(cached_resources["embeddings"])
                except Exception as init_err:  # noqa: F841
                    print(f"[INIT] 资源预加载警告（将使用懒加载回退）: {init_err}")

                # ===== 步骤 3：OCR 识别 =====
                if image_path:
                    with status_1:
                        status_1.update(label="📷 正在进行图片 OCR 识别（约 5-30 秒）...", expanded=True)
                        st.write("🔍 图片文字提取中，大图片会自动压缩处理...")

                # ===== 步骤 4：调用 RAG 系统（OCR + 搜索 + 检索 + LLM）=====
                with status_1:
                    if image_path:
                        st.write("🌐 网络搜索 + 📚 知识库匹配 + 🤖 智能研判中...")
                    else:
                        st.write("🌐 网络搜索 + 📚 知识库匹配 + 🤖 智能研判中...")

                try:
                    result = process_query(
                        user_text=user_input or "",
                        image_path=image_path,
                        cached_resources=cached_resources,
                    )
                except Exception as inner_err:  # noqa: BLE001
                    print(f"[CRITICAL] process_query 异常: {type(inner_err).__name__}: {inner_err}")
                    traceback.print_exc()
                    result = {
                        "ocr_result": None,
                        "llm_answer": "",
                        "context_text": "",
                        "user_text": user_input or "",
                        "combined_query": user_input or "",
                        "error": "⚠️ 服务器繁忙或检索超时，请稍后再试或简化输入问题。",
                    }

                # ===== 步骤 5：完成，清除进度状态并渲染结果 =====
                progress_placeholder.empty()
                render_result(result, user_input or "")

            except Exception as e:  # noqa: BLE001
                print(f"[CRITICAL] 外层兜底异常: {type(e).__name__}: {e}")
                traceback.print_exc()
                progress_placeholder.empty()
                with tab_answer:
                    st.error("⚠️ 服务器繁忙或检索超时，请稍后再试或简化输入问题。")
                st.session_state.last_report_data = None
                st.session_state.last_result = None

            finally:
                if image_path and os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        os.rmdir(os.path.dirname(image_path))
                    except Exception:  # noqa: BLE001
                        pass
    elif st.session_state.get("last_result"):
        # 页面重新运行时恢复显示上一次的研判结果
        render_result(
            st.session_state.last_result,
            st.session_state.last_result.get("user_text", "")
        )

    # ========== 导出研判报告按钮 ==========
    if st.session_state.get("last_report_data"):
        st.markdown("---")
        st.download_button(
            label="📥 一键导出研判报告",
            data=st.session_state.last_report_data.encode("utf-8"),
            file_name="外交领事智能研判报告.txt",
            mime="text/plain",
            use_container_width=False,
            help="将本次研判的问题、解答和依据导出为文本文件",
        )

    # 底部说明
    st.markdown("""
    <div class="diplo-footer">
        本系统基于 RAG（检索增强生成）技术，融合官方领事知识库与 EasyOCR 视觉识别能力，
        为用户提供专业、严谨、符合外交礼仪规范的智能问答服务。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()