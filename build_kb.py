"""
外交与领事知识库构建脚本
功能：加载 PDF/TXT 文档，添加官方来源元数据，构建向量数据库

环境变量(可选,本地开发用):
    HF_ENDPOINT: HuggingFace 镜像地址,如 https://hf-mirror.com
    (Streamlit Cloud 部署时不要设置,使用默认的 huggingface.co)
"""
import os

# 从 .env 文件加载环境变量(本地开发用),不硬编码镜像源
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ========== 官方文件名称映射 ==========
# 根据文件名自动映射到规范的官方文件标题
SOURCE_TITLE_MAP = {
    "中国领事保护与协助指南.txt": "《中国领事保护与协助指南》",
    "中国领事保护与协助指南.pdf": "《中国领事保护与协助指南》",
    "外交相关知识.txt": "《外交礼仪与涉外事务规范》",
    "外交相关知识.pdf": "《外交礼仪与涉外事务规范》",
    "日本.txt": "《赴日领事服务与注意事项》",
    "日本.pdf": "《赴日领事服务与注意事项》",
}

# 统一权威来源标注
DEFAULT_AUTHORITY = "中华人民共和国外交部 / 中国领事服务网 (cs.mfa.gov.cn)"

# 排除非知识库文件
EXCLUDE_FILES = {"requirements.txt", "packages.txt", ".env"}


def add_metadata_to_documents(documents, file_name):
    """
    为文档添加元数据：source_title 和 authority
    """
    # 获取规范的官方文件名称
    source_title = SOURCE_TITLE_MAP.get(file_name, f"《{os.path.splitext(file_name)[0]}》")

    for doc in documents:
        doc.metadata["source_title"] = source_title
        doc.metadata["authority"] = DEFAULT_AUTHORITY
        doc.metadata["original_file"] = file_name

    return documents


def build_knowledge_base():
    """读取文档并构建向量数据库"""
    print("--- 开始读取知识库文档 ---")

    documents = []
    success_count = 0
    fail_count = 0

    for file in os.listdir("."):
        if file in EXCLUDE_FILES:
            continue
        if file.endswith(".pdf"):
            print(f"正在尝试加载 PDF: {file}")
            try:
                loader = PyPDFLoader(file)
                docs = loader.load()
                # 添加元数据
                docs = add_metadata_to_documents(docs, file)
                documents.extend(docs)
                success_count += 1
                print(f"  ✓ 已添加元数据：{SOURCE_TITLE_MAP.get(file, file)}")
            except Exception as e:
                print(f"加载 PDF 失败（若一直报错，建议将 PDF 另存为 TXT 后再试）: {e}")
                fail_count += 1
        elif file.endswith(".txt"):
            print(f"正在加载 TXT: {file}")
            try:
                loader = TextLoader(file, encoding="utf-8")
                docs = loader.load()
                # 添加元数据
                docs = add_metadata_to_documents(docs, file)
                documents.extend(docs)
                success_count += 1
                print(f"  ✓ 已添加元数据：{SOURCE_TITLE_MAP.get(file, file)}")
            except Exception as e:
                print(f"加载 TXT 失败: {e}")
                fail_count += 1

    print(f"\n文档读取统计：成功 {success_count} 个，失败 {fail_count} 个")

    if not documents:
        print("❌ 错误：没有任何文档被成功读取！")
        return

    # 文档切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    print(f"✅ 文档切分完成，共生成 {len(docs)} 个知识片段。")

    # 统计元数据
    sources = set()
    for doc in docs:
        sources.add(doc.metadata.get("source_title", "未知来源"))
    print(f"✅ 包含官方文件：{', '.join(sorted(sources))}")

    print("--- 正在下载文本向量模型（首次运行约 1-2 分钟）---")
    # 不使用 local_files_only,允许 Streamlit Cloud 首次部署时下载模型
    embeddings = HuggingFaceEmbeddings(
        model_name="shibing624/text2vec-base-chinese",
    )

    print("--- 正在构建向量数据库 ---")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("\n🎉 成功！你的外交与领事知识库已构建完成，保存在 ./chroma_db 文件夹中！")
    print(f"📊 知识库统计：")
    print(f"  - 总文档片段：{len(docs)} 个")
    print(f"  - 官方来源：{len(sources)} 个")
    print(f"  - 权威机构：{DEFAULT_AUTHORITY}")


if __name__ == "__main__":
    build_knowledge_base()
