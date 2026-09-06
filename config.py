import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 加载 .env 文件
load_dotenv(os.path.join(BASE_DIR, ".env"))

# DeepSeek LLM配置
LLM_CONFIG = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
}

# Embedding配置（本地BGE中文模型）
EMBEDDING_CONFIG = {
    "model_name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
    "device": os.getenv("EMBEDDING_DEVICE", "cpu"),
    "normalize_embeddings": os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
}

# 面试题PDF目录（项目内）
PDF_DIR = os.path.join(BASE_DIR, "data", "interview_pdfs")

# Chroma向量库路径
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
CHROMA_COLLECTION_NAME = "llm_interview_kb"

# 文本分块配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索配置
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "10"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))

# 面试记录保存目录
INTERVIEW_RECORD_DIR = os.path.join(BASE_DIR, "interviews")
