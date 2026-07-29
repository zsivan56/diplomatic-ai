"""
Streamlit 前端应用：基于 RAG 的多模态领事服务与外交礼仪智能助手
"""
import os
import sys
import tempfile
from pathlib import Path

# 确保项目根目录在 sys.path 中，以便导入 rag_system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag_system import process_query

# ========== 页面配置 ==========
st.set_page_config(
    page_title="基于 RAG 的多模态领事服务与外交礼仪智能助手",
    page_icon="🛂",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "基于 RAG 的多模态领事服务与外交礼仪智能助手 - 人工智能导论期末作品"
    }
)

# ========== 自定义 CSS 样式 ==========
def load_custom_css():
    """加载自定义 CSS 样式：外交蓝主题"""
    st.markdown("""
    <style>
        /* 主色调：外交蓝 */
        :root {
            --diplo-primary: #0F2C59;
            --diplo-secondary: #1a3a5c;
            --diplo-accent: #c9a962;
            --diplo-bg: #F8F9FA;
        }
        
        /* 全局字体 */
        .main, .block-container {
            font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", Arial, sans-serif;
        }
        
        /* 标题样式 */
        .main-title {
            color: var(--diplo-primary);
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            padding: 1.5rem 0;
        }
        
        .subtitle {
            color: #555555;
            font-size: 1rem;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* 卡片样式 */
        .result-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        }
        
        .ocr-card {
            background: #fefcf5;
            border-left: 4px solid var(--diplo-accent);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .ocr-card h3 {
            color: #9a6700;
            margin-top: 0;
        }
        
        .answer-card h3 {
            color: var(--diplo-primary);
            border-left: 4px solid var(--diplo-accent);
            padding-left: 0.75rem;
            margin-top: 0;
        }
        
        /* 按钮样式 */
        .stButton > button {
            background: linear-gradient(180deg, #1a3a5c 0%, #0F2C59 100%);
            color: white;
            border: 1px solid #c9a962;
            border-radius: 8px;
            font-weight: 600;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(180deg, #2c5282 0%, #1a3a5c 100%);
            box-shadow: 0 4px 12px rgba(10, 37, 64, 0.4);
            transform: translateY(-1px);
        }
        
        /* 侧边栏样式 */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8f9fa 0%, #e8eef5 100%);
        }
        
        section[data-testid="stSidebar"] .element-container {
            margin-bottom: 1rem;
        }
        
        /* 示例问题按钮 */
        .example-btn {
            background: white !important;
            color: #0F2C59 !important;
            border: 1px solid #c9a962 !important;
            text-align: left !important;
            padding: 0.75rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 8px !important;
            width: 100% !important;
            transition: all 0.2s ease !important;
        }
        
        .example-btn:hover {
            background: #c9a962 !important;
            color: white !important;
            border-color: #0F2C59 !important;
        }
        
        /* 输入框样式 */
        .stTextArea textarea {
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            font-size: 15px;
        }
        
        .stTextArea textarea:focus {
            border-color: #0F2C59;
            box-shadow: 0 0 0 2px rgba(15, 44, 89, 0.1);
        }
        
        /* 文件上传器样式 */
        .stFileUploader {
            border: 2px dashed #d0d0d0;
            border-radius: 12px;
            padding: 1rem;
            background: white;
        }
        
        /* 加载动画 */
        .loading {
            text-align: center;
            color: #0F2C59;
            font-weight: 600;
            padding: 2rem;
        }
        
        /* 分隔线 */
        hr {
            border-color: #c9a962;
            margin: 1.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """渲染页面标题"""
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 class="main-title">🛂 基于 RAG 的多模态领事服务与外交礼仪智能助手</h1>
        <p class="subtitle">
            结合计算机视觉（CV）与 RAG 垂直知识库 | 人工智能导论期末作品展示
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """渲染侧边栏：图片上传 + 示例问题"""
    with st.sidebar:
        st.markdown("### 📷 图片上传")
        st.markdown("上传护照、签证、告示等证件图片（可选）")
        
        uploaded_file = st.file_uploader(
            label="上传证件图片",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'],
            help="支持格式：JPG、PNG、BMP、TIFF、WebP，建议图片清晰、文字正面朝上",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("### 💡 快捷测试问题")
        st.markdown("点击下方按钮快速体验：")
        
        # 预设示例问题
        example_questions = [
            "🇯🇵 在日本遇到突发地震，领事保护电话是多少？有什么社交禁忌？",
            "🍽️ 涉外商务宴请中有哪些礼仪规范？座次如何安排？",
            "🛂 中国公民在海外遗失护照应该如何补办？需要哪些材料？"
        ]
        
        # 使用 session_state 存储选中的示例问题
        for i, question in enumerate(example_questions):
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.selected_question = question
        
        st.markdown("---")
        
        # 使用说明
        with st.expander("ℹ️ 使用说明"):
            st.markdown("""
            **操作步骤：**
            1. 在主界面输入您的问题，或在侧边栏选择预设问题
            2. 可选：上传证件图片进行 OCR 识别
            3. 点击「提交智能研判」按钮
            4. 查看智能解答与知识库依据
            
            **支持功能：**
            - 领事保护与协助咨询
            - 外交礼仪与涉外规范
            - 证件 OCR 识别与解读
            """)
    
    return uploaded_file


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
        border_color = "#c53030"
        bg_color = "#fff5f5"
    elif not has_text:
        icon = "⚠️"
        border_color = "#d69e2e"
        bg_color = "#fff8e1"
    else:
        icon = "📷"
        border_color = "#c9a962"
        bg_color = "#fefcf5"
    
    st.markdown(f"""
    <div class="ocr-card" style="background: {bg_color}; border-left: 4px solid {border_color};">
        <h3>{icon} 【OCR 识别文字】</h3>
        <p style="color: #555; margin-bottom: 0.5rem;">{message}</p>
    """, unsafe_allow_html=True)
    
    if text:
        st.markdown(f"""
        <div style="background: white; border: 1px dashed #c9a962; padding: 1rem; 
                    border-radius: 4px; font-family: monospace; font-size: 14px; 
                    white-space: pre-wrap; word-break: break-word; max-height: 300px; overflow-y: auto;">
{text}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_llm_answer(answer, context):
    """渲染 LLM 回答和检索依据"""
    if not answer:
        return
    
    st.markdown("""
    <div class="result-card answer-card">
        <h3>🤖 智能助手解答</h3>
    """, unsafe_allow_html=True)
    
    st.markdown(answer)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if context:
        st.markdown("""
        <hr style="border-color: #c9a962; margin: 1.5rem 0;">
        <div class="result-card">
            <h3 style="color: #0F2C59; border-left: 4px solid #c9a962; padding-left: 0.75rem;">
                📚 检索调用的参考依据
            </h3>
        """, unsafe_allow_html=True)
        
        st.markdown(context)
        
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    """主函数"""
    load_custom_css()
    render_header()
    
    # 侧边栏
    uploaded_file = render_sidebar()
    
    # 主界面
    st.markdown("### 📝 信息输入")
    
    # 文本输入框（从 session_state 获取选中的示例问题）
    default_text = st.session_state.get("selected_question", "")
    user_input = st.text_area(
        label="请输入您关于领事保护、外交礼仪、涉外事务等方面的问题：",
        value=default_text,
        height=150,
        placeholder="例如：在日本遇到突发地震，如何寻求领事保护？\n\n也可以上传护照、签证等证件图片，系统会自动识别图中文字并研判。",
        label_visibility="collapsed"
    )
    
    # 清除 session_state 中的选中问题（避免重复填充）
    if "selected_question" in st.session_state:
        del st.session_state.selected_question
    
    # 提交按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        submit_button = st.button("🚀 提交智能研判", type="primary", use_container_width=True)
    
    # 处理提交
    if submit_button:
        if not user_input and not uploaded_file:
            st.warning("⚠️ 请输入提问文本或上传证件图片后再提交。")
        else:
            # 显示加载状态
            with st.spinner("🔍 正在智能研判中..."):
                # 保存上传的文件
                image_path = save_uploaded_file(uploaded_file)
                
                try:
                    # 调用 RAG 系统
                    result = process_query(user_text=user_input or "", image_path=image_path)
                    
                    # 渲染结果
                    st.markdown("---")
                    
                    # OCR 结果
                    render_ocr_result(result.get("ocr_result"))
                    
                    # 错误处理
                    if result.get("error"):
                        st.error(f"⚠️ 处理提示\n\n{result['error']}")
                    else:
                        # LLM 回答
                        render_llm_answer(
                            result.get("llm_answer", ""),
                            result.get("context_text", "")
                        )
                
                except Exception as e:
                    st.error(f"❌ 系统处理时发生错误：\n\n**错误类型**：{type(e).__name__}\n\n**错误信息**：{str(e)}")
                
                finally:
                    # 清理临时文件
                    if image_path and os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            os.rmdir(os.path.dirname(image_path))
                        except:  # noqa: E722
                            pass
    
    # 底部说明
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px; padding: 1rem;">
        本系统基于 RAG（检索增强生成）技术，融合官方领事知识库与 EasyOCR 视觉识别能力，
        为用户提供专业、严谨、符合外交礼仪规范的智能问答服务。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()