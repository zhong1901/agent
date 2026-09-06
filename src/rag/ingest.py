import os
import hashlib
import json
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

from config import (
    PDF_DIR,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_CONFIG,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from src.rag.utils import extract_text_from_pdf, categorize_pdf, list_pdf_files, clean_text


STATE_FILE = os.path.join(CHROMA_PERSIST_DIR, "ingest_state.json")


def load_ingest_state() -> Dict:
    """加载已处理文件的状态（用于增量更新）"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_files": {}}


def save_ingest_state(state: Dict):
    """保存处理状态"""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_file_hash(filepath: str) -> str:
    """计算文件MD5，用于判断是否变更"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def get_embedding_model():
    """加载BGE中文embedding模型"""
    model_kwargs = {'device': EMBEDDING_CONFIG['device']}
    encode_kwargs = {'normalize_embeddings': EMBEDDING_CONFIG['normalize_embeddings']}
    return HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_CONFIG['model_name'],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


def get_vector_store(embedding_model):
    """获取Chroma向量存储实例"""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def split_documents(full_text: str, filename: str, categories: List[str],
                    pages_info: List[Dict]) -> List[Dict]:
    """
    将文档分块，返回带metadata的块列表
    每个块: {"page_content": "...", "metadata": {...}}
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    # 按页分块，保留页码信息
    chunks = []
    for page_info in pages_info:
        page_text = page_info["text"]
        if not page_text.strip():
            continue

        page_chunks = text_splitter.split_text(page_text)
        for i, chunk_text in enumerate(page_chunks):
            chunk_text = clean_text(chunk_text)
            if len(chunk_text) < 50:  # 跳过过短的块
                continue
            chunks.append({
                "page_content": chunk_text,
                "metadata": {
                    "source": filename,
                    "page": page_info["page_num"],
                    "categories": ",".join(categories),
                    "chunk_index": i,
                    "char_count": len(chunk_text),
                }
            })

    return chunks


def ingest_pdfs(force_rebuild: bool = False) -> Dict:
    """
    摄取所有PDF到向量库
    force_rebuild: True则清空重建
    返回统计信息
    """
    state = load_ingest_state()
    embedding_model = get_embedding_model()
    vector_store = get_vector_store(embedding_model)

    if force_rebuild:
        # 清空重建
        try:
            vector_store.delete_collection()
        except Exception:
            pass
        vector_store = get_vector_store(embedding_model)
        state = {"processed_files": {}}

    pdf_files = list_pdf_files(PDF_DIR)
    stats = {
        "total_pdfs": len(pdf_files),
        "new_processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks": 0,
        "failed_files": [],
    }

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        file_hash = get_file_hash(pdf_path)

        # 检查是否已处理且未变更
        if filename in state["processed_files"] and state["processed_files"][filename] == file_hash:
            stats["skipped"] += 1
            continue

        try:
            print(f"处理中: {filename}")
            full_text, pages_info = extract_text_from_pdf(pdf_path)
            categories = categorize_pdf(filename)
            chunks = split_documents(full_text, filename, categories, pages_info)

            if not chunks:
                print(f"  警告: 无有效文本块，跳过")
                stats["failed"] += 1
                stats["failed_files"].append(filename)
                continue

            # 写入向量库
            texts = [c["page_content"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

            state["processed_files"][filename] = file_hash
            stats["new_processed"] += 1
            stats["total_chunks"] += len(chunks)
            print(f"  完成: {len(chunks)} 个文本块, 分类: {categories}")

        except Exception as e:
            print(f"  失败: {e}")
            stats["failed"] += 1
            stats["failed_files"].append(filename)

    save_ingest_state(state)
    return stats


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    print("=" * 50)
    print("开始构建RAG知识库...")
    print(f"PDF目录: {PDF_DIR}")
    print(f"向量库路径: {CHROMA_PERSIST_DIR}")
    print(f"强制重建: {force}")
    print("=" * 50)

    stats = ingest_pdfs(force_rebuild=force)

    print("\n" + "=" * 50)
    print("构建完成!")
    print(f"  总PDF数: {stats['total_pdfs']}")
    print(f"  新处理: {stats['new_processed']}")
    print(f"  跳过(未变更): {stats['skipped']}")
    print(f"  失败: {stats['failed']}")
    print(f"  总文本块数: {stats['total_chunks']}")
    if stats["failed_files"]:
        print(f"  失败文件: {stats['failed_files']}")
    print("=" * 50)
