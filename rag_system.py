import os
import re

# 设置国内镜像源，避免直连 huggingface.co 超时
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 修复 macOS Python SSL 证书验证问题（EasyOCR 下载模型需要）
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import easyocr
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# 从 .env 文件加载环境变量
load_dotenv()

# 1. 配置大语言模型 API (以 DeepSeek 为例)
# 优先从环境变量读取 API Key，避免硬编码
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
BASE_URL = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")

if not API_KEY:
    raise EnvironmentError(
        "未检测到 DEEPSEEK_API_KEY，请在 Hugging Face Spaces 的 Repository secrets 中设置该变量，"
        "或在本地创建 .env 文件写入 DEEPSEEK_API_KEY=your_key_here"
    )

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.3  # 较低的温度保证回答客观严谨，减少幻觉
)

# 2. 加载之前构建好的向量数据库
print("正在加载外交领事知识库...")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"local_files_only": True},
)
vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})  # 检索最相关的 3 条依据

# 3. 初始化计算机视觉 (CV) OCR 识别器 (支持中文和英文)
print("正在初始化 OCR 视觉识别引擎...")
# 使用项目内目录存放 OCR 模型，避免 macOS 权限问题
ocr_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".easyocr_model")
os.makedirs(ocr_model_dir, exist_ok=True)
ocr_reader = easyocr.Reader(
    ['ch_sim', 'en'],
    model_storage_directory=ocr_model_dir,
    user_network_directory=ocr_model_dir,
)

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


def extract_text_from_image(image_path):
    """
    【模块 A】从图片中使用 EasyOCR 提取文字。

    参数:
        image_path: 图片文件路径

    返回:
        dict: {
            "success": bool,          # 识别是否成功（无异常、格式支持）
            "text": str,              # 提取到的纯文本（若失败则为 ""）
            "message": str,           # 前端展示的友好提示
            "has_text": bool          # 实际是否提取到了有效文字内容
        }
    """
    result = {
        "success": False,
        "text": "",
        "message": "",
        "has_text": False,
    }

    # 1. 校验路径是否存在
    if not image_path:
        result["message"] = "未提供图片路径。"
        return result

    if not os.path.exists(image_path):
        result["message"] = f"图片文件不存在：{os.path.basename(image_path)}"
        return result

    # 2. 校验文件大小（避免超大文件卡住 OCR）
    try:
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > 20:
            result["message"] = f"图片文件过大（{file_size_mb:.1f} MB），请上传小于 20MB 的清晰图片。"
            return result
    except OSError as e:
        result["message"] = f"无法读取图片文件：{e}"
        return result

    # 3. 校验扩展名
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXT:
        result["message"] = (
            f"不支持的图片格式「{ext}」。\n"
            f"支持的格式：{', '.join(sorted(SUPPORTED_IMAGE_EXT))}"
        )
        return result

    # 4. 调用 EasyOCR 进行识别（外层全面捕获异常）
    try:
        print(f"[OCR] 正在识别图片: {image_path} ({file_size_mb:.2f} MB)")

        # detail=1 同时拿到坐标/置信度，便于后续过滤低置信度行
        raw_results = ocr_reader.readtext(
            image_path,
            detail=1,        # 返回 (bbox, text, prob) 三元组
            paragraph=False, # 保留细粒度行
            contrast_ths=0.1,
            adjust_contrast=0.5,
        )

        # 过滤置信度过低的识别结果（<0.4 很可能是噪点）
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

    except (FileNotFoundError, IsADirectoryError) as e:
        result["message"] = f"图片读取失败，文件路径无效：{e}"
        return result
    except (ValueError, TypeError) as e:
        result["message"] = f"图片数据格式异常，OCR 无法解析：{e}\n请确认上传的是有效的图片文件。"
        return result
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda" in msg or "gpu" in msg:
            result["message"] = (
                "OCR 引擎在调用 GPU 时出错，已切换为 CPU 模式。\n"
                "如果问题持续，请更换更小或更清晰的图片后重试。"
            )
        elif "out of memory" in msg:
            result["message"] = "OCR 内存不足，请上传更小尺寸的图片或压缩后重试。"
        else:
            result["message"] = f"OCR 运行时错误：{e}\n请稍后重试，或更换其他图片。"
        return result
    except Exception as e:  # noqa: BLE001 - 前端需要兜底兜底所有异常
        result["message"] = (
            f"❌ OCR 识别过程中出现未知错误：{type(e).__name__}: {e}\n"
            "请检查图片格式与清晰度，或稍后再试。"
        )
        return result


def process_query(user_text="", image_path=None):
    """
    核心处理函数：融合 CV 提取结果与 RAG 知识库检索。

    返回一个字典（而不是单一字符串），供前端分区域展示：
    {
        "ocr_result": {
            "success": bool,
            "text": str,
            "message": str,
            "has_text": bool,
            "input_image": str  # 原始图片路径
        },
        "llm_answer": str,
        "context_text": str,
        "user_text": str,
        "combined_query": str,
        "error": str  # 若整个流程异常，则在这里记录
    }
    """
    # 输出结构模板
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
    ocr_result_dict = None

    if image_path:
        ocr_result_dict = extract_text_from_image(image_path)
        ocr_result_dict["input_image"] = image_path
        output["ocr_result"] = ocr_result_dict

        if ocr_result_dict["success"]:
            extracted_text = ocr_result_dict["text"]
        # 若 OCR 失败（非崩掉、而是友好提示），仍然继续，让用户纯文字问题也能被回答

    # ========== 模块 B: 融合文本与图片内容构建查询 ==========
    user_text_clean = (user_text or "").strip()
    combined_parts = [p for p in [user_text_clean, extracted_text] if p]
    combined_query = " ".join(combined_parts).strip()
    output["combined_query"] = combined_query

    if not combined_query:
        # 没有任何文字输入（也没有 OCR 文本）
        if image_path and ocr_result_dict and not ocr_result_dict["success"]:
            # 是因为 OCR 失败导致没有文字，优先返回 OCR 错误提示
            output["error"] = (
                "未能获取到有效输入：" + ocr_result_dict["message"]
            )
        else:
            output["error"] = "请提供文字提问或上传包含可识别文字的证件/告示图片！"
        return output

    # ========== 模块 C: 检索增强生成 (RAG) 知识库匹配 ==========
    try:
        print(f"[RAG] 正在检索知识库中的相关条款，查询长度={len(combined_query)}...")
        relevant_docs = retriever.invoke(combined_query)
    except Exception as e:  # noqa: BLE001
        output["error"] = f"知识库检索失败：{type(e).__name__}: {e}"
        return output

    # 拼接检索到的官方资料片段作为上下文 (Context)
    context_text = "\n\n".join(
        [f"[依据 {i + 1}]: {doc.page_content}" for i, doc in enumerate(relevant_docs)]
    )
    output["context_text"] = context_text

    # ========== 模块 D: 构建 Prompt 并送入大模型 ==========
    system_content = f"""你是一名专业的外交领事与涉外礼仪智能助手。
请结合以下从官方知识库中检索到的权威依据，回答用户的问题或对证件信息给出提醒。

要求：
1. 态度严谨、礼貌，符合外交礼仪规范。
2. 答案必须严格优先依据提供的参考资料，切勿捏造条款。
3. 标注回答所参考的依据编号。

【官方检索依据】：
{context_text}"""

    # 如果有 OCR 结果，在 user_content 中明确区分"用户提问"和"图片提取内容"
    if extracted_text:
        user_content = (
            f"【用户提问】\n{user_text_clean or '（未填写文字提问，以下为图片识别内容）'}\n\n"
            f"【OCR 图片识别提取到的文字】\n{extracted_text}"
        )
    else:
        user_content = f"【用户提问】\n{user_text_clean}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_content),
            HumanMessage(content=user_content),
        ])
        output["llm_answer"] = response.content
    except Exception as e:  # noqa: BLE001
        output["error"] = f"大模型调用失败：{type(e).__name__}: {e}"
        return output

    return output


# ================================
# 为保持命令行运行（__main__ 测试）仍能直接看到结果，
# 增加一个将结构化输出渲染为 Markdown 字符串的辅助函数。
# ================================
def render_output_to_markdown(output):
    """
    把 process_query 返回的字典渲染为单一 Markdown 字符串（用于 CLI 测试或兼容旧调用）。
    """
    parts = []

    # OCR 识别区
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

    # 异常兜底
    if output.get("error"):
        parts.append(f"### ⚠️ 处理提示")
        parts.append("")
        parts.append(output["error"])
        parts.append("")
        return "\n".join(parts)

    # LLM 回答
    if output.get("llm_answer"):
        parts.append("### 🤖 智能助手解答：")
        parts.append("")
        parts.append(output["llm_answer"])
        parts.append("")

    # 检索依据
    if output.get("context_text"):
        parts.append("---")
        parts.append("")
        parts.append("### 📚 检索调用的参考依据：")
        parts.append("")
        parts.append(output["context_text"])

    return "\n".join(parts)


# 测试运行（命令行测试）
if __name__ == "__main__":
    print("\n" + "=" * 50)
    test_question = "如果在日本遇到突发地震，领事保护电话是多少？有什么社交禁忌？"
    print(f"测试提问: {test_question}\n")
    result_dict = process_query(user_text=test_question)
    print(render_output_to_markdown(result_dict))
