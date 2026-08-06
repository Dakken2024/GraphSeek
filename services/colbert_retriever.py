"""
ColBERT Token 级检索器（白皮书演进 2：文档级向量 -> Token 级多路召回）。

轻量接入方案：基于已安装的 sentence-transformers 加载 ColBERT 系列模型
（如 colbert-ir/colbertv2.0），实现 Query×Document Token 延迟交互 MaxSim 打分。

设计约束：
- 模型懒加载，加载失败自动降级（available=False），不影响 FAISS+BM25 基座通路；
- 纯 numpy/torch 实现 MaxSim，不依赖 colbert-ai/ragatouille 重型库。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from utils.logger import get_logger


logger = get_logger(__name__)


class ColbertRetriever:
    """基于 sentence-transformers 的轻量 ColBERT 检索器。"""

    DEFAULT_MODEL = "colbert-ir/colbertv2.0"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
        top_k: int = 10,
        max_tokens_per_doc: int = 512,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.top_k = top_k
        self.max_tokens_per_doc = max_tokens_per_doc
        self._model: Any = None
        self._load_attempted = False
        self._available = False

    # -- 生命周期 ----------------------------------------------------------

    @property
    def available(self) -> bool:
        """模型是否已成功加载。"""
        if not self._load_attempted:
            self._load()
        return self._available

    def _load(self) -> None:
        """懒加载 ColBERT 模型（失败时置 available=False 不抛出）。"""
        self._load_attempted = True
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(self.model_name, device=self.device)
            # ColBERT 模型输出 token 级嵌入（output_value="token_embeddings"）
            if hasattr(model, "max_seq_length"):
                model.max_seq_length = self.max_tokens_per_doc
            self._model = model
            self._available = True
            logger.info(f"ColBERT 模型加载成功: {self.model_name} @ {self.device}")
        except Exception as e:
            self._available = False
            logger.warning(
                f"ColBERT 模型加载失败（自动降级为 FAISS+BM25 双路）: {e}"
            )

    def unload(self) -> None:
        """释放模型内存。"""
        self._model = None
        self._load_attempted = False
        self._available = False

    # -- 打分与检索 --------------------------------------------------------

    def _token_embeddings(self, text: str):
        """获取文本的 token 级嵌入（list of (d,) tensors）。"""
        return self._model.encode(
            text, output_value="token_embeddings", convert_to_numpy=False
        )

    def score_pair(self, query: str, document: str) -> float:
        """单 Query-Doc 对 MaxSim 打分（ColBERT 延迟交互）。"""
        q_emb = self._token_embeddings(query)
        d_emb = self._token_embeddings(document)
        # 移除 CLS 向量效果由模型决定，直接计算 MaxSim
        sims = q_emb @ d_emb.T            # (nq, nd)
        return float(sims.max(dim=1).values.sum().item())

    def score_pairs(self, query: str, documents: Sequence[str]) -> List[float]:
        """批量打分。"""
        if not self._available:
            return [0.0] * len(documents)
        q_emb = self._token_embeddings(query)
        scores: List[float] = []
        for doc in documents:
            d_emb = self._token_embeddings(doc)
            sims = q_emb @ d_emb.T
            scores.append(float(sims.max(dim=1).values.sum().item()))
        return scores

    def retrieve(
        self,
        query: str,
        documents: Optional[Sequence[Document]] = None,
        texts: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """
        从文档集合中按 Token 级相关性召回。

        Args:
            query: 查询
            documents: Document 列表（与 texts 二选一）
            texts: 纯文本列表（无 metadata）
            top_k: 返回数量

        Returns:
            List[Document]（降序）；模型不可用时返回空列表
        """
        if not self.available:
            return []
        docs = documents or [Document(page_content=t) for t in (texts or [])]
        if not docs:
            return []
        scores = self.score_pairs(query, [d.page_content for d in docs])
        ranked = [d for _, d in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
        return ranked[: top_k or self.top_k]
