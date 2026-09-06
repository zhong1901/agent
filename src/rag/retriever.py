from typing import List, Dict, Optional
from langchain_chroma import Chroma

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    RETRIEVER_TOP_K,
    RETRIEVER_K,
    MMR_LAMBDA,
)
from src.rag.ingest import get_embedding_model, get_vector_store


class InterviewRetriever:
    """面试知识库检索器封装"""

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store: Chroma = get_vector_store(self.embedding_model)

    def similarity_search(
        self,
        query: str,
        k: int = RETRIEVER_TOP_K,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        语义相似度检索
        query: 用户问题
        k: 返回top-k结果
        category: 可选，按分类过滤（如"RAG"、"微调"）
        返回: [{"content": "...", "source": "...", "page": 1, "categories": "...", "score": 0.9}, ...]
        """
        filter_dict = {}
        if category:
            filter_dict = {"categories": {"$contains": category}}

        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict if filter_dict else None,
        )

        formatted = []
        for doc, score in results:
            formatted.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
                "categories": doc.metadata.get("categories", ""),
                "score": float(score),
            })

        return formatted

    def mmr_search(
        self,
        query: str,
        k: int = RETRIEVER_TOP_K,
        fetch_k: int = RETRIEVER_K,
        lambda_mult: float = MMR_LAMBDA,
        category: Optional[str] = None,
    ) -> List[Dict]:
        """
        MMR检索（最大边际相关性）- 在相关性和多样性之间平衡
        lambda_mult=1 等价于普通相似度检索，=0 最大化多样性
        """
        filter_dict = {}
        if category:
            filter_dict = {"categories": {"$contains": category}}

        results = self.vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter_dict if filter_dict else None,
        )

        formatted = []
        for doc in results:
            formatted.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", 0),
                "categories": doc.metadata.get("categories", ""),
                "score": 0.0,  # MMR不直接返回score
            })

        return formatted

    def search(
        self,
        query: str,
        k: int = RETRIEVER_TOP_K,
        category: Optional[str] = None,
        use_mmr: bool = True,
    ) -> List[Dict]:
        """统一检索入口，默认使用MMR"""
        if use_mmr:
            return self.mmr_search(query, k=k, category=category)
        else:
            return self.similarity_search(query, k=k, category=category)

    def format_context(self, results: List[Dict]) -> str:
        """将检索结果格式化为LLM可用的上下文字符串"""
        if not results:
            return "（无相关参考资料）"

        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"【参考资料{i}】来源: {r['source']} (第{r['page']}页)\n"
                f"分类: {r['categories']}\n"
                f"内容:\n{r['content']}\n"
            )

        return "\n".join(context_parts)

    def get_collection_stats(self) -> Dict:
        """获取向量库统计信息"""
        collection = self.vector_store._collection
        count = collection.count()
        return {
            "total_chunks": count,
            "collection_name": CHROMA_COLLECTION_NAME,
        }


# 单例
_retriever_instance: Optional[InterviewRetriever] = None


def get_retriever() -> InterviewRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = InterviewRetriever()
    return _retriever_instance


if __name__ == "__main__":
    # 测试检索
    retriever = get_retriever()
    stats = retriever.get_collection_stats()
    print(f"向量库统计: {stats}")

    test_queries = [
        "什么是RAG？",
        "LoRA的原理是什么？",
        "大模型分布式训练有哪些方式？",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"查询: {q}")
        print(f"{'='*60}")
        results = retriever.search(q, k=3)
        for i, r in enumerate(results, 1):
            print(f"\n--- 结果{i}: {r['source']} p.{r['page']} [{r['categories']}] ---")
            print(r['content'][:200] + "..." if len(r['content']) > 200 else r['content'])
