"""
RAG 核心系统：多模态领事服务与外交礼仪智能助手
优化版本：支持 Streamlit Cloud 部署
"""
import os
import re
from typing import Optional, Dict, Any

# 可选：设置国内镜像源（国内开发时手动在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com）
# 不硬编码，让 Streamlit Cloud 使用默认的 huggingface.co
pass

# 从 .env 文件加载环境变量（本地开发用）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ========== 全局变量（懒加载）==========
_llm = None
_retriever = None
_ocr_reader = None
_init_error = None


def _get_llm():
    """懒加载 LLM 实例"""
    global _llm, _init_error
    
    if _llm is not None:
        return _llm
    
    try:
        from langchain_openai import ChatOpenAI
        
        # 优先从环境变量读取 API Key
        api_key = os.getenv("DEEPSEEK_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        
        if not api_key:
            raise EnvironmentError(
                "未检测到 DEEPSEEK_API_KEY，请在 Streamlit Cloud 的 Secrets 中设置该变量。"
            )
        
        _llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=api_key,
            base_url=base_url,
            temperature=0.3
        )
        return _llm
        
    except Exception as e:
        _init_error = f"LLM 初始化失败: {str(e)}"
        raise


def _get_retriever():
    """懒加载向量数据库检索器"""
    global _retriever, _init_error
    
    if _retriever is not None:
        return _retriever
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        
        print("正在加载外交领事知识库...")
        
        # Streamlit Cloud 上不需要 local_files_only
        embeddings = HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
        )
        
        vector_db = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings
        )
        
        _retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        return _retriever
        
    except Exception as e:
        _init_error = f"知识库加载失败: {str(e)}"
        raise


def _get_ocr_reader():
    """懒加载 OCR 识别器"""
    global _ocr_reader, _init_error
    
    if _ocr_reader is not None:
        return _ocr_reader
    
    try:
        import easyocr
        
        print("正在初始化 OCR 视觉识别引擎...")
        
        # 使用项目内目录存放 OCR 模型
        ocr_model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            ".easyocr_model"
        )
        os.makedirs(ocr_model_dir, exist_ok=True)
        
        _ocr_reader = easyocr.Reader(
            ['ch_sim', 'en'],
            model_storage_directory=ocr_model_dir,
            user_network_directory=ocr_model_dir,
        )
        return _ocr_reader
        
    except Exception as e:
        _init_error = f"OCR 引擎初始化失败: {str(e)}"
        raise


# 支持的图片格式扩展名
SUPPORTED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _clean_ocr_text(raw_lines):
    """
    清洗 OCR 识别出的原始文本：去除空白、过滤无意义的单字符碎片、合并行
    """
    if not raw_lines:
        return ""

    cleaned = []
    for line in raw_lines:
        line = str(line).strip()
        if not line:
            continue
        # 去除过长的全角/半角空格串
        line = re.sub(r'\s{2,}', ' ', line)
        # 过滤仅含无意义字符的行（如纯标点、纯符号、仅 1 个非字母数字的符号）
        if re.fullmatch(r'[\W_]{1,2}', line) and not re.search(r'[A-Za-z0-9\u4e00-\u9fa5]', line):
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def extract_text_from_image(image_path: Optional[str]) -> Dict[str, Any]:
    """
    从图片中使用 EasyOCR 提取文字。

    参数:
        image_path: 图片文件路径

    返回:
        dict: {
            "success": bool,
            "text": str,
            "message": str,
            "has_text": bool
        }
    """
    result = {
        "success": False,
        "text": "",
        "message": "",
        "has_text": False,
    }

    if not image_path:
        result["message"] = "未提供图片路径。"
        return result

    if not os.path.exists(image_path):
        result["message"] = f"图片文件不存在：{os.path.basename(image_path)}"
        return result

    try:
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > 20:
            result["message"] = f"图片文件过大（{file_size_mb:.1f} MB），请上传小于 20MB 的清晰图片。"
            return result
    except OSError as e:
        result["message"] = f"无法读取图片文件：{e}"
        return result

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXT:
        result["message"] = (
            f"不支持的图片格式「{ext}」。\n"
            f"支持的格式：{', '.join(sorted(SUPPORTED_IMAGE_EXT))}"
        )
        return result

    try:
        print(f"[OCR] 正在识别图片: {image_path} ({file_size_mb:.2f} MB)")

        # 懒加载 OCR 识别器
        ocr_reader = _get_ocr_reader()

        raw_results = ocr_reader.readtext(
            image_path,
            detail=1,
            paragraph=False,
            contrast_ths=0.1,
            adjust_contrast=0.5,
        )

        confident_lines = []
        low_conf_count = 0
        for bbox, text, prob in raw_results:
            if prob >= 0.4:
                confident_lines.append(text)
            else:
                low_conf_count += 1

        cleaned_text = _clean_ocr_text(confident_lines)
        cleaned_text = cleaned_text.strip()

        result["success"] = True
        result["text"] = cleaned_text

        if cleaned_text:
            result["has_text"] = True
            char_count = len(cleaned_text.replace("\n", "").replace(" ", ""))
            hint = ""
            if low_conf_count > 0:
                hint = f"（已过滤 {low_conf_count} 条低置信度噪点）"
            result["message"] = (
                f"✅ 图片识别成功，共提取 {len(cleaned_text.splitlines())} 行，"
                f"{char_count} 个字符{hint}。"
            )
        else:
            result["has_text"] = False
            result["message"] = (
                "⚠️ 图片识别完成，但未检测到可识别的文字内容。\n"
                "建议：上传清晰度更高、文字较大的证件或告示图片，并确保文字正面朝上。"
            )

        print(f"[OCR] 提取到 {len(cleaned_text)} 字符文本（has_text={result['has_text']}）")
        return result

    except Exception as e:  # noqa: BLE001
        result["message"] = (
            f"❌ OCR 识别过程中出现错误：{type(e).__name__}: {e}\n"
            "请检查图片格式与清晰度，或稍后再试。"
        )
        return result


def format_context_with_metadata(relevant_docs):
    """
    将检索到的文档片段格式化为结构化的 Markdown 输出
    格式：
    ---
    📌 **依据 [序号]**
    • **权威来源**：[来源文件名]
    • **发布/发布机构**：[authority]
    • **调取的原条文节选**：
      > "[具体检索到的文本片段...]"
    ---
    """
    formatted_parts = []

    for i, doc in enumerate(relevant_docs, 1):
        # 提取元数据
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        source_title = metadata.get("source_title", "未知来源")
        authority = metadata.get("authority", "未知机构")
        page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)

        # 格式化单个依据
        formatted_part = f"""---
📌 **依据 [{i}]**
• **权威来源**：{source_title}
• **发布/发布机构**：{authority}
• **调取的原条文节选**：
  > "{page_content.strip()}"
---"""
        formatted_parts.append(formatted_part)

    return "\n\n".join(formatted_parts)


def process_query(user_text: str = "", image_path: Optional[str] = None) -> Dict[str, Any]:
    """
    核心处理函数：融合 CV 提取结果与 RAG 知识库检索。

    返回一个字典，供前端分区域展示：
    {
        "ocr_result": dict,
        "llm_answer": str,
        "context_text": str,
        "user_text": str,
        "combined_query": str,
        "error": str
    }
    """
    output = {
        "ocr_result": None,
        "llm_answer": "",
        "context_text": "",
        "user_text": user_text or "",
        "combined_query": "",
        "error": "",
    }

    # ========== 模块 A: OCR 图片文字提取 ==========
    extracted_text = ""

    if image_path:
        try:
            ocr_result_dict = extract_text_from_image(image_path)
            ocr_result_dict["input_image"] = image_path
            output["ocr_result"] = ocr_result_dict

            if ocr_result_dict["success"]:
                extracted_text = ocr_result_dict["text"]
        except Exception as e:  # noqa: BLE001
            output["ocr_result"] = {
                "success": False,
                "text": "",
                "message": f"OCR 识别异常：{str(e)}",
                "has_text": False,
                "input_image": image_path
            }

    # ========== 模块 B: 融合文本与图片内容构建查询 ==========
    user_text_clean = (user_text or "").strip()
    combined_parts = [p for p in [user_text_clean, extracted_text] if p]
    combined_query = " ".join(combined_parts).strip()
    output["combined_query"] = combined_query

    if not combined_query:
        if image_path and output.get("ocr_result") and not output["ocr_result"].get("success"):
            output["error"] = "未能获取到有效输入：" + output["ocr_result"]["message"]
        else:
            output["error"] = "请提供文字提问或上传包含可识别文字的证件/告示图片！"
        return output

    # ========== 模块 C: 检索增强生成 (RAG) ==========
    try:
        # 懒加载检索器
        retriever = _get_retriever()

        print(f"[RAG] 正在检索知识库中的相关条款，查询长度={len(combined_query)}...")
        relevant_docs = retriever.invoke(combined_query)

        # 使用结构化格式化函数
        context_text = format_context_with_metadata(relevant_docs)
        output["context_text"] = context_text

    except Exception as e:  # noqa: BLE001
        output["error"] = f"知识库检索失败：{type(e).__name__}: {e}"
        return output

    # ========== 模块 D: 构建 Prompt 并调用 LLM ==========
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        
        # 懒加载 LLM
        llm = _get_llm()
        
        system_content = f"""你是一名专业的外交领事与涉外礼仪智能助手。
请结合以下从官方知识库中检索到的权威依据，回答用户的问题或对证件信息给出提醒。

要求：
1. 态度严谨、礼貌，符合外交礼仪规范。
2. 答案必须严格优先依据提供的参考资料，切勿捏造条款。
3. 标注回答所参考的依据编号。

【官方检索依据】：
{context_text}"""

        if extracted_text:
            user_content = (
                f"【用户提问】\n{user_text_clean or '（未填写文字提问，以下为图片识别内容）'}\n\n"
                f"【OCR 图片识别提取到的文字】\n{extracted_text}"
            )
        else:
            user_content = f"【用户提问】\n{user_text_clean}"

        response = llm.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ])
        output["llm_answer"] = response.content
        
    except Exception as e:  # noqa: BLE001
        output["error"] = f"大模型调用失败：{type(e).__name__}: {e}"
        return output

    return output


def render_output_to_markdown(output: Dict[str, Any]) -> str:
    """
    把 process_query 返回的字典渲染为单一 Markdown 字符串（用于 CLI 测试）。
    """
    parts = []

    if output.get("ocr_result"):
        ocr = output["ocr_result"]
        parts.append("### 📷 【OCR 识别文字】")
        parts.append("")
        parts.append(f"> {ocr['message']}")
        parts.append("")
        if ocr.get("text"):
            parts.append("```")
            parts.append(ocr["text"])
            parts.append("```")
        parts.append("")
        parts.append("---")
        parts.append("")

    if output.get("error"):
        parts.append("### ⚠️ 处理提示")
        parts.append("")
        parts.append(output["error"])
        parts.append("")
        return "\n".join(parts)

    if output.get("llm_answer"):
        parts.append("### 🤖 智能助手解答：")
        parts.append("")
        parts.append(output["llm_answer"])
        parts.append("")

    if output.get("context_text"):
        parts.append("---")
        parts.append("")
        parts.append("### 📚 检索调用的参考依据：")
        parts.append("")
        parts.append(output["context_text"])

    return "\n".join(parts)


# 测试运行
if __name__ == "__main__":
    print("\n" + "=" * 50)
    test_question = "如果在日本遇到突发地震，领事保护电话是多少？有什么社交禁忌？"
    print(f"测试提问: {test_question}\n")
    result_dict = process_query(user_text=test_question)
    print(render_output_to_markdown(result_dict))