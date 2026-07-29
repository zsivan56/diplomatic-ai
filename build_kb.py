import os

# 【关键修复】设置国内镜像源，URL 必须是纯字符串，不能带反引号
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("--- 开始读取知识库文档 ---")

documents = []
success_count = 0
fail_count = 0

for file in os.listdir("."):
    if file.endswith(".pdf"):
        print(f"正在尝试加载 PDF: {file}")
        try:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
            success_count += 1
        except Exception as e:
            print(f"加载 PDF 失败（若一直报错，建议将 PDF 另存为 TXT 后再试）: {e}")
            fail_count += 1
    elif file.endswith(".txt"):
        print(f"正在加载 TXT: {file}")
        try:
            loader = TextLoader(file, encoding="utf-8")
            documents.extend(loader.load())
            success_count += 1
        except Exception as e:
            print(f"加载 TXT 失败: {e}")
            fail_count += 1

print(f"\n文档读取统计：成功 {success_count} 个，失败 {fail_count} 个")

if not documents:
    print("❌ 错误：没有任何文档被成功读取！")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)
print(f"✅ 文档切分完成，共生成 {len(docs)} 个知识片段。")

print("--- 正在连接国内镜像，下载文本向量模型（大约需要1-2分钟）---")
embeddings = HuggingFaceEmbeddings(
    model_name="shibing624/text2vec-base-chinese",
    model_kwargs={"local_files_only": True},
)

print("--- 正在构建向量数据库 ---")
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("\n🎉 成功！你的外交与领事知识库已构建完成，保存在 ./chroma_db 文件夹中！")