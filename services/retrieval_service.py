"""
Retrieval service for GraphSeek application.
Handles document retrieval with hybrid search, reranking, and GraphRAG.

2026 演进（白皮书演进 2/3/4）：
- 三路召回：FAISS/BM25（基座，保留降级） + ColBERT Token 级 + 图谱（Low/High 双层）
- RRF（Reciprocal Rank Fusion）融合替代固定权重拼接
- 多目标重排：语义 + 图中心性 + 时效 + MMR 多样性
- Agentic Query Planning 查询分解（替代 HyDE；HyDE 保留为可选项）
"""
from typing import List, Optional, Dict, Any, Sequence, Tuple
from langchain_core.documents import Document

from core.cache import RetrievalCache
from services.colbert_retriever import ColbertRetriever
from services.query_planner import QueryPlanner
from services.reranker import MultiObjectiveReranker
from utils.monitoring import Monitor
from utils.logger import get_logger


logger = get_logger(__name__)


def rrf_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = 60,
) -> List[Document]:
    """
    Reciprocal Rank Fusion：融合多路召回结果。

    Args:
        ranked_lists: 多路按相关度降序的文档列表
        k: RRF 平滑常数（通常 60）

    Returns:
        融合后的文档列表（按 RRF 分数降序，内容去重）
    """
    score_map: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            score_map[key] = score_map.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map.setdefault(key, doc)
    ordered_keys = sorted(score_map, key=lambda x: score_map[x], reverse=True)
    return [doc_map[key] for key in ordered_keys]


class RetrievalService:
    """Service for retrieving documents using multiple strategies."""

    def __init__(
        self,
        ensemble_retriever,
        reranker=None,
        knowledge_graph=None,
        graph_service=None,
        cache_enabled: bool = True,
        cache_ttl: float = 3600.0,
        colbert_retriever: Optional[ColbertRetriever] = None,
        candidate_documents: Optional[Sequence[Document]] = None,
        llm_service=None,
    ) -> None:
        self.ensemble_retriever = ensemble_retriever
        self.reranker = reranker
        self.knowledge_graph = knowledge_graph
        self.graph_service = graph_service
        self.colbert_retriever = colbert_retriever
        self.candidate_documents = list(candidate_documents or [])
        self.planner = QueryPlanner(llm_service=llm_service)

        # 多目标重排器（保留 CrossEncoder 作为 S_rel 信号源）
        self.multi_reranker = MultiObjectiveReranker(
            cross_encoder=reranker, graph_service=graph_service
        )

        # Initialize cache
        self.cache_enabled = cache_enabled
        self.cache = RetrievalCache(max_size=1000, default_ttl=cache_ttl) if cache_enabled else None

        # Initialize monitoring
        self.monitor = Monitor()

        # 最近一次重排明细（供 UI 溯源面板展示）
        self.last_rerank_details: List[Dict[str, Any]] = []

    def retrieve(
        self,
        query: str,
        chat_history: str = "",
        enable_hyde: bool = True,
        enable_graph_rag: bool = True,
        enable_reranking: bool = True,
        max_contexts: int = 3,
        llm_service=None,
        use_cache: bool = True,
        enable_query_planning: bool = True,
        enable_colbert: bool = True,
        enable_community: bool = True,
        enable_mmr: bool = True,
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search and optional enhancements.

        Returns:
            List of retrieved documents
        """
        cache_key_params = {
            "chat_history": chat_history,
            "enable_hyde": enable_hyde,
            "enable_graph_rag": enable_graph_rag,
            "enable_reranking": enable_reranking,
            "max_contexts": max_contexts,
            "enable_query_planning": enable_query_planning,
            "enable_colbert": enable_colbert,
            "enable_community": enable_community,
            "enable_mmr": enable_mmr,
        }

        if use_cache and self.cache_enabled and self.cache:
            cached_result = self.cache.get(query, **cache_key_params)
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result

        with self.monitor.measure("retrieval.retrieve"):
            # Agentic 查询规划（Plan-and-Solve）
            sub_queries = [query]
            if enable_query_planning:
                sub_queries = self.planner.plan(query)

            fused_docs: List[Document] = []
            for sub_query in sub_queries:
                expanded = self._expand_query(
                    sub_query, chat_history, enable_hyde, llm_service
                )
                pool = self._multi_path_recall(
                    expanded,
                    query,
                    enable_graph_rag=enable_graph_rag,
                    enable_colbert=enable_colbert,
                    enable_community=enable_community,
                )
                fused_docs.extend(pool)

            # 跨子查询去重（按内容）
            seen: set = set()
            deduped = []
            for doc in fused_docs:
                key = doc.page_content
                if key not in seen:
                    seen.add(key)
                    deduped.append(doc)

            # 多目标重排
            if enable_reranking:
                ranked, details = self.multi_reranker.rerank(
                    query, deduped, enable_mmr=enable_mmr
                )
                self.last_rerank_details = details
                docs = ranked
            else:
                docs = deduped
                self.last_rerank_details = []

            result_docs = docs[:max_contexts]

            if use_cache and self.cache_enabled and self.cache:
                self.cache.set(
                    query, result_docs,
                    metadata={"doc_count": len(result_docs)},
                    **cache_key_params
                )

            return result_docs

    # -- 多路召回 ----------------------------------------------------------

    def _multi_path_recall(
        self,
        expanded_query: str,
        original_query: str,
        enable_graph_rag: bool,
        enable_colbert: bool,
        enable_community: bool,
    ) -> List[Document]:
        """基座(FAISS+BM25) + ColBERT + 图谱三路召回。"""
        paths: List[List[Document]] = []

        # 通路 1：FAISS + BM25 基座（白皮书：保留为基础召回）
        try:
            base_docs = self.ensemble_retriever.invoke(expanded_query)
            paths.append(base_docs or [])
        except Exception as e:
            logger.warning(f"基座召回失败: {e}")

        # 通路 2：ColBERT Token 级（模型不可用时自动返回空）
        if enable_colbert and self.colbert_retriever is not None:
            colbert_docs = self.colbert_retriever.retrieve(
                expanded_query, documents=self.candidate_documents
            )
            if colbert_docs:
                paths.append(colbert_docs)

        # 通路 3：图谱（Low-level 实体路径 + High-level 社区摘要）
        graph_docs = self._retrieve_from_graph(
            original_query, enable_community=enable_community
        )
        if graph_docs:
            paths.append(graph_docs)

        # RRF 融合
        if len(paths) == 1:
            return paths[0]
        return rrf_fusion(paths)

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return None

    def clear_cache(self) -> bool:
        """Clear the retrieval cache."""
        if self.cache:
            self.cache.clear()
            return True
        return False

    def _expand_query(
        self,
        query: str,
        chat_history: str,
        enable_hyde: bool,
        llm_service,
    ) -> str:
        """
        Expand query using HyDE if enabled（保留为可选能力）。
        """
        if not enable_hyde or not llm_service:
            return f"{chat_history}\n{query}" if chat_history else query

        combined_query = f"{chat_history}\n{query}" if chat_history else query
        hypothetical = llm_service.generate_hypothetical_answer(combined_query)
        return f"{combined_query}\n{hypothetical}"

    def _retrieve_from_graph(self, query: str, enable_community: bool = True) -> List[Document]:
        """
        图谱召回：Low-level 实体路径（PageRank）+ High-level 社区摘要。

        Returns:
            List of documents from graph retrieval
        """
        if not self.graph_service:
            return []

        graph_docs: List[Document] = []

        # Low-level：实体多跳/PageRank 相关节点（微观查询）
        related_nodes = self.graph_service.query_graph(query)
        graph_docs.extend(Document(page_content=node) for node in related_nodes)

        # High-level：社区摘要（宏观查询）
        if enable_community:
            communities = self.graph_service.query_community(query, top_k=2)
            for c in communities:
                text = (
                    f"[社区 {c['community_id']}] {c['summary']}\n"
                    f"核心实体: {', '.join(c['key_entities'])}"
                )
                graph_docs.append(Document(page_content=text))

        return graph_docs
