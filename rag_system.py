"""
RAG 核心系统：多模态领事服务与外交礼仪智能助手
优化版本：支持 Streamlit Cloud 部署
- 网络搜索（DuckDuckGo）作为主要信息来源
- 本地知识库作为校验参考，纠正幻觉
"""
import os
import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

# 可选：设置国内镜像源（本地开发时手动在 .env 中设置 HF_ENDPOINT=https://hf-mirror.com）
# 不硬编码，让 Streamlit Cloud 使用默认的 huggingface.co

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


def _get_llm():
    """懒加载 LLM 实例"""
    global _llm

    if _llm is not None:
        return _llm

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
        temperature=0.3,
        timeout=30,
    )
    return _llm


def _get_retriever(cached_resources: Optional[Dict[str, Any]] = None):
    """懒加载向量数据库检索器，支持从外部注入缓存的 embedding / vector_db"""
    global _retriever

    # 优先使用外部注入的缓存（Streamlit @st.cache_resource 提供的全局单例）
    if _retriever is None and cached_resources:
        cached_db = cached_resources.get("vector_db")
        if cached_db is not None:
            try:
                _retriever = cached_db.as_retriever(search_kwargs={"k": 3})
                return _retriever
            except Exception as e:  # noqa: BLE001
                print(f"[RAG] 使用缓存 vector_db 失败，回退懒加载: {e}")

    if _retriever is not None:
        return _retriever

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


def _get_ocr_reader(cached_resources: Optional[Dict[str, Any]] = None):
    """懒加载 OCR 识别器，支持从外部注入缓存的 Reader"""
    global _ocr_reader

    # 优先使用外部注入的缓存（Streamlit @st.cache_resource 提供的全局单例）
    if _ocr_reader is None and cached_resources:
        cached = cached_resources.get("ocr_reader")
        if cached is not None:
            _ocr_reader = cached
            return _ocr_reader

    if _ocr_reader is not None:
        return _ocr_reader

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


# 支持的图片格式扩展名
SUPPORTED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def _preprocess_image_for_ocr(image_path: str, max_side: int = 1568) -> str:
    """
    OCR 前预处理:如果图片最大边超过 max_side,等比缩小后存为临时 PNG。
    返回处理后的图片路径(未处理则返回原路径)。
    目的:加速 CPU 端 EasyOCR,避免大图导致云端超时。
    """
    try:
        from PIL import Image
    except ImportError:
        # 无 Pillow 则跳过预处理,直接用原图
        return image_path

    try:
        with Image.open(image_path) as img:
            w, h = img.size
            longest = max(w, h)
            if longest <= max_side:
                # 图片不大,无需预处理
                return image_path

            # 等比缩小
            ratio = max_side / longest
            new_size = (int(w * ratio), int(h * ratio))
            resized = img.convert("RGB").resize(new_size, Image.LANCZOS)

            # 保存为临时 PNG
            import tempfile
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="ocr_pre_")
            os.close(fd)
            resized.save(tmp_path, format="PNG")
            print(f"[OCR] 图片预处理: {w}x{h} → {new_size[0]}x{new_size[1]}")
            return tmp_path
    except Exception as e:
        print(f"[OCR] 图片预处理失败,使用原图: {e}")
        return image_path


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


def extract_text_from_image(
    image_path: Optional[str],
    cached_resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    从图片中使用 EasyOCR 提取文字。

    参数:
        image_path: 图片文件路径
        cached_resources: 外部注入的缓存模型（来自 Streamlit @st.cache_resource）

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

        # 懒加载 OCR 识别器（优先使用缓存模型，避免重复加载爆内存）
        ocr_reader = _get_ocr_reader(cached_resources)

        # 图片预处理：大图自动缩小,加速 CPU 端 OCR(最大边 1568px)
        processed_path = _preprocess_image_for_ocr(image_path)
        try:
            raw_results = ocr_reader.readtext(
                processed_path,
                detail=1,
                paragraph=False,
                contrast_ths=0.1,
                adjust_contrast=0.5,
            )
        finally:
            # 清理预处理产生的临时文件(若与原路径不同)
            if processed_path != image_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except OSError:
                    pass

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


def _humanize_title(title: str, url: str) -> str:
    """
    当搜索引擎返回的标题是 URL 形式时,尝试从 URL 路径提取可读片段作为标题。
    否则返回原标题。
    """
    if not title:
        return title
    # 判断是否是 URL 形式的标题(以 http 开头或包含 / 且无中文空格分隔词)
    if title.startswith("http") or (".gov.cn/" in title and len(title) > 30):
        try:
            path = urlparse(url).path if url else ""
            # 取最后一段路径,去除扩展名和数字后缀
            last_segment = path.rstrip("/").split("/")[-1] if path else ""
            # 去除 .html/.htm 后缀和 _数字 后缀
            last_segment = re.sub(r'\.(html?|shtml)$', '', last_segment)
            last_segment = re.sub(r'_\d+$', '', last_segment)
            if last_segment:
                return f"外交部官方页面({last_segment})"
        except Exception:
            pass
    return title


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    使用 DuckDuckGo 搜索网络信息，返回结构化搜索结果。
    策略：普通关键词搜索（中文 + 领域强化词），然后按域名过滤，
    把官方结果（cs.mfa.gov.cn / mfa.gov.cn）排到前面，
    不使用 site: 语法（海外服务器上 site: 常返回 0 条）。
    每条结果包含：title, source(域名), snippet, url, is_official
    """
    try:
        from ddgs import DDGS

        print(f"[Web Search] 正在搜索：{query[:60]}...")

        # 去除多余空白字符
        clean_query = re.sub(r'\s+', ' ', query).strip()

        # 官方域名白名单（用于搜索后过滤标记）
        OFFICIAL_DOMAINS = {
            "cs.mfa.gov.cn", "mfa.gov.cn", "www.mfa.gov.cn",
            "www.gov.cn", "gov.cn",
        }

        # 只要官方域名后缀命中（含子域）即算官方
        def _is_official(domain: str) -> bool:
            return any(
                domain == d or domain.endswith("." + d)
                for d in OFFICIAL_DOMAINS
            )

        formatted = []
        seen_urls = set()

        # 第一轮：普通搜索（带领域强化词）
        # 海外服务器上 site: cs.mfa.gov.cn 几乎搜不到结果，
        # 用关键词强化（外交部 领事 礼仪 出国 安全 签证）+ 后过滤更可靠
        boosted_query = f"{clean_query} 外交部 领事"
        try:
            raw_results = list(DDGS().text(boosted_query, max_results=max_results + 3))
        except Exception as e:  # noqa: BLE001
            print(f"[Web Search] 第一轮(boosted)搜索失败: {e}")
            raw_results = []

        # 第一轮没结果时，第二轮：不加领域词的纯用户问题
        if not raw_results:
            try:
                raw_results = list(DDGS().text(clean_query, max_results=max_results + 3))
            except Exception as e:  # noqa: BLE001
                print(f"[Web Search] 第二轮(plain)搜索失败: {e}")
                raw_results = []

        for r in raw_results:
            if len(formatted) >= max_results + 3:  # 多取 3 条用于后过滤排序
                break
            url = r.get("href", r.get("link", ""))
            if not url or url in seen_urls:
                continue
            title = r.get("title", "")
            snippet = r.get("body", r.get("snippet", ""))
            if not (title and snippet):
                continue
            seen_urls.add(url)
            domain = urlparse(url).netloc or "未知来源"
            formatted.append({
                "title": _humanize_title(title, url),
                "source": domain,
                "snippet": snippet,
                "url": url,
                "is_official": _is_official(domain),
            })

        # 排序：官方结果优先，然后按原顺序
        formatted.sort(key=lambda x: (not x["is_official"]))

        # 只取前 max_results 条
        formatted = formatted[:max_results]

        print(f"[Web Search] 获取到 {len(formatted)} 条网络搜索结果"
              f"（官方 {sum(1 for r in formatted if r.get('is_official'))} 条）")
        return formatted

    except Exception as e:
        print(f"[Web Search] 搜索失败: {type(e).__name__}: {e}")
        return []


def format_web_evidence(web_results: List[Dict[str, str]]) -> str:
    """
    将网络搜索结果格式化为结构化的「依据」卡片。
    注意:网络搜索结果为搜索引擎返回的网页摘要,并非法规原文,
    标注须如实反映其性质,避免误导用户。
    """
    if not web_results:
        return ""

    formatted_parts = []
    for i, result in enumerate(web_results, 1):
        title = result.get("title", "未知标题")
        source = result.get("source", "未知来源")  # 实为域名
        snippet = result.get("snippet", "无摘要")
        url = result.get("url", "")
        is_official = result.get("is_official", False)

        trust_label = "🏛 官方来源" if is_official else "📰 第三方网页"

        formatted_part = f"""---
🌐 **网络搜索结果 [{i}]**  `{trust_label}`
• **网页标题**：{title}
• **来源域名**：{source}
• **网页摘要**（搜索引擎返回，非法规原文，仅供参考）：
  > "{snippet.strip()}"
{f"• **链接**：{url}" if url else ""}
---"""
        formatted_parts.append(formatted_part)

    return "\n\n".join(formatted_parts)


def format_local_db_reference(relevant_docs) -> str:
    """
    将本地知识库检索结果格式化为「知识库校验」卡片。
    知识库源文件为官方指导性文档（如《中国领事保护与协助指南》），
    非严格意义上的法规条文，标注须如实反映。
    """
    if not relevant_docs:
        return ""

    formatted_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}
        source_title = metadata.get("source_title", "未知来源")
        authority = metadata.get("authority", "未知机构")
        page_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)

        formatted_part = f"""---
📚 **知识库校验 [{i}]**  `🏛 官方指导文档`
• **文件名称**：{source_title}
• **发布机构**：{authority}
• **知识库文档节选**：
  > "{page_content.strip()}"
---"""
        formatted_parts.append(formatted_part)

    return "\n\n".join(formatted_parts)


def process_query(
    user_text: str = "",
    image_path: Optional[str] = None,
    cached_resources: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    核心处理函数：融合 CV 提取结果与 RAG 知识库检索。
    - cached_resources：由 Streamlit @st.cache_resource 注入的全局单例模型，
      避免多用户访问时重复加载模型导致内存溢出。

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
    # ========== 最外层兜底：任何未预料的异常都不崩溃 ==========
    _fallback = {
        "ocr_result": None,
        "llm_answer": "",
        "context_text": "",
        "user_text": user_text or "",
        "combined_query": user_text or "",
        "error": "⚠️ 服务器繁忙或检索超时，请稍后再试或简化输入问题。",
    }
    try:
        return _process_query_inner(user_text, image_path, cached_resources)
    except Exception as e:  # noqa: BLE001
        print(f"[CRITICAL] process_query 全局异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return _fallback


def _process_query_inner(
    user_text: str,
    image_path: Optional[str],
    cached_resources: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """process_query 的内部实现（被外层 try-except 兜底包装）"""
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
            # 将 cached_resources 注入到 OCR 初始化链路中（避免重复加载爆内存）
            ocr_result_dict = extract_text_from_image(image_path, cached_resources=cached_resources)
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

    # ========== 模块 C: 网络搜索（主要信息来源）==========
    web_results = []
    web_evidence_text = ""
    web_search_diag = ""  # 诊断:失败原因,用于兜底提示
    try:
        web_results = search_web(combined_query, max_results=3)
        web_evidence_text = format_web_evidence(web_results)
        if not web_results:
            web_search_diag = "搜索引擎未返回有效结果（可能因服务器网络区域限制）"
    except Exception as e:  # noqa: BLE001
        print(f"[Web Search] 异常: {e}")
        web_search_diag = f"搜索异常：{type(e).__name__}: {e}"

    # ========== 模块 D: 本地知识库检索（纠正幻觉）==========
    local_db_text = ""
    local_db_diag = ""
    try:
        retriever = _get_retriever(cached_resources)
        print(f"[RAG] 正在检索本地知识库校验条款...")

        # Sanity check: 确认向量库里有数据(防止连接到空 collection)
        try:
            underlying_db = retriever.vectorstore
            collection_count = underlying_db._collection.count()
            print(f"[RAG] 知识库文档数: {collection_count}")
            if collection_count == 0:
                local_db_diag = "知识库连接成功但 collection 为空（数据库可能未正确加载）"
                # 不 return,继续尝试检索
        except Exception as count_e:  # noqa: BLE001
            print(f"[RAG] 无法获取知识库文档数: {count_e}")

        relevant_docs = retriever.invoke(combined_query)
        local_db_text = format_local_db_reference(relevant_docs)
        if not relevant_docs:
            local_db_diag = "知识库未检索到匹配片段（可能数据库为空或 embedding 维度不匹配）"
    except Exception as e:  # noqa: BLE001
        local_db_err = f"{type(e).__name__}: {e}"
        print(f"[RAG] 本地知识库检索失败: {local_db_err}")
        local_db_diag = local_db_err

    # 合并展示：网络依据 + 知识库校验
    context_parts = []
    if web_evidence_text:
        context_parts.append(web_evidence_text)
    if local_db_text:
        context_parts.append(local_db_text)

    context_text = "\n\n".join(context_parts) if context_parts else ""
    output["context_text"] = context_text

    # 诊断信息,用于 LLM fallback 时显示给用户
    retrieval_hint_parts = []
    if web_search_diag:
        retrieval_hint_parts.append(f"• 🌐 网络搜索：{web_search_diag}")
    if local_db_diag:
        retrieval_hint_parts.append(f"• 📚 本地知识库：{local_db_diag}")
    retrieval_hint = "\n".join(retrieval_hint_parts)

    # 任何一边有结果就正常继续(不再报错)
    # 两边都空时也继续走 LLM fallback,不直接报错,但在 LLM 回答前加免责声明
    fallback_mode = (not context_text)

    # ========== 模块 E: 构建 Prompt 并调用 LLM ==========
    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        # 懒加载 LLM
        llm = _get_llm()

        if fallback_mode:
            # 检索全空:fallback 模式,LLM 仅能基于自身训练知识回答,
            # 必须严格免责 + 严禁捏造法规/电话等事实信息
            system_content = f"""你是一名专业的外交领事与涉外礼仪智能助手。

⚠️ 【重要警示】: 本次查询网络搜索与本地知识库均未获取到有效参考依据。
你只能基于自己的训练知识给出**一般性的建议或指引**,不可当作最终权威结论。

【回答必须遵守】:
1. 开头必须先声明:「⚠️ 提示：本次未检索到实时官方资料，以下仅为一般性参考指引，请以中华人民共和国外交部 12308 领事保护热线或当地使领馆官方答复为准。」
2. 严禁捏造或杜撰任何具体的法律法规条文、电话号码、签证要求或具体国别政策。
3. 涉及领事应急/证件办理/法规效力类问题,必须明确引导用户拨打 12308 或查询 cs.mfa.gov.cn 官方渠道。
4. 回答应礼貌、严谨,不给出绝对性承诺。

【用户问题】: {combined_query}"""
        else:
            system_content = f"""你是一名专业的外交领事与涉外礼仪智能助手。
请结合以下参考依据，回答用户的问题或对证件信息给出提醒。

【依据说明】
- 🌐 网络搜索结果：搜索引擎返回的网页摘要，非法规原文。其中「官方来源」可信度高，「第三方网页」仅供参考。
- 📚 知识库校验：官方指导性文档节选，可信度最高，作为核实与纠错依据。

【回答要求】
1. 态度严谨、礼貌，符合外交礼仪规范。
2. 优先采信「🏛 官方来源」网络结果与「📚 知识库校验」；对「📰 第三方网页」内容需谨慎，不可作为事实依据。
3. 切勿捏造条款、电话号码、法规条文。若依据中无相关信息，明确告知用户建议查询官方渠道（外交部 12308 热线）。
4. 标注回答所参考的依据编号（如「网络搜索结果 [1]」「知识库校验 [2]」）。

【参考依据】：
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
        llm_text = response.content

        # Fallback 模式:在回答前置诊断提示(便于用户直观看到哪一步失败)
        if fallback_mode:
            diag_lines = ["### ⚠️ 检索状态说明"]
            diag_lines.append("")
            diag_lines.append("本次未能获取到实时或本地参考依据,已切换为通用知识答复模式:")
            diag_lines.append("")
            diag_lines.append(retrieval_hint or "• 未知原因")
            diag_lines.append("")
            diag_lines.append("如需获得权威答案,建议稍后重试,或直接拨打外交部 12308 领事保护热线。")
            diag_lines.append("")
            diag_lines.append("---")
            diag_lines.append("")
            llm_text = "\n".join(diag_lines) + llm_text

        output["llm_answer"] = llm_text

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