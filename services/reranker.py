"""
多目标重排器（白皮书演进 3：多目标重排序 Multi-Objective Reranking）。

综合评分模型（白皮书附录 B）:
    Score_final = α·S_rel + β·S_graph + γ·S_div + δ·S_time

- S_rel  : CrossEncoder 语义相关性（无重排器时用召回分数归一化）
- S_graph: 文档中命中实体的图中心性（PageRank/度中心性）
- S_div  : MMR 多样性（惩罚与已选结果的语义重复）
- S_time : 文档时效性指数衰减 e^(-λΔt)

所有目标归一化到 [0,1]，权重可配置，缺失信号自动取中性值，保证可降级。
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

from utils.logger import get_logger
from utils.monitoring import Monitor


logger = get_logger(__name__)


DEFAULT_WEIGHTS = {
    "rel": 0.5,
    "graph": 0.2,
    "div": 0.2,
    "time": 0.1,
}


def _minmax(scores: List[float]) -> List[float]:
    """Min-Max 归一化到 [0,1]（空或全等时返回 0.5）。"""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _jaccard(a: str, b: str) -> float:
    """基于词集合的轻量文本相似度（无嵌入依赖）。"""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    return inter / (len(wa) + len(wb) - inter)


def _doc_age_days(doc: Document) -> Optional[float]:
    """从 metadata 提取文档年龄（天）。支持 created/updated/modified/date。"""
    import datetime as dt
    for key in ("created", "updated", "modified", "date", "timestamp"):
        val = doc.metadata.get(key)
        if val is None:
            continue
        try:
            if isinstance(val, (int, float)):
                ts = float(val)
            else:
                ts = dt.datetime.fromisoformat(str(val)).timestamp()
            return max(0.0, (dt.datetime.now().timestamp() - ts) / 86400.0)
        except (ValueError, TypeError, OverflowError):
            continue
    return None


class MultiObjectiveReranker:
    """多目标重排器。"""

    def __init__(
        self,
        cross_encoder=None,
        graph_service=None,
        weights: Optional[Dict[str, float]] = None,
        mmr_lambda: float = 0.7,
        time_decay_rate: float = 0.01,
    ) -> None:
        self.cross_encoder = cross_encoder
        self.graph_service = graph_service
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.mmr_lambda = mmr_lambda
        self.time_decay_rate = time_decay_rate
        self.monitor = Monitor()
        self._centrality: Dict[str, float] = {}

    # -- 各目标打分 --------------------------------------------------------

    def _score_rel(
        self, query: str, docs: List[Document], base_scores: Optional[Sequence[float]]
    ) -> List[float]:
        if self.cross_encoder is not None and docs:
            try:
                pairs = [[query, d.page_content] for d in docs]
                raw = self.cross_encoder.predict(pairs).tolist()
                return _minmax(raw)
            except Exception as e:
                logger.warning(f"CrossEncoder 打分失败，降级: {e}")
        if base_scores is not None and len(base_scores) == len(docs):
            return _minmax(list(base_scores))
        return [0.5] * len(docs)

    def _score_graph(self, docs: List[Document]) -> List[float]:
        if self.graph_service is None:
            return [0.5] * len(docs)
        graph = getattr(self.graph_service, "graph", None)
        if graph is None or len(graph) == 0:
            return [0.5] * len(docs)
        if not self._centrality:
            try:
                self._centrality = graph.degree_centrality()
            except Exception:
                self._centrality = {n: 1.0 for n in graph.nodes}
        max_c = max(self._centrality.values()) if self._centrality else 1.0
        scores = []
        for doc in docs:
            content_lower = doc.page_content.lower()
            best = 0.0
            for node, cent in self._centrality.items():
                if node.lower() in content_lower:
                    best = max(best, cent)
            scores.append(best / max_c if max_c > 0 else 0.0)
        return scores

    def _score_time(self, docs: List[Document]) -> List[float]:
        scores = []
        for doc in docs:
            age = _doc_age_days(doc)
            if age is None:
                scores.append(0.5)
            else:
                scores.append(math.exp(-self.time_decay_rate * age))
        return scores

    # -- MMR 多样性 --------------------------------------------------------

    def _mmr_select(
        self,
        query: str,
        docs: List[Document],
        base_scores: List[float],
    ) -> Tuple[List[Document], List[float]]:
        """MMR 增量选择：兼顾相关性与多样性。"""
        if len(docs) <= 1 or self.mmr_lambda >= 1.0:
            return docs, base_scores
        selected: List[int] = []
        remaining = list(range(len(docs)))
        max_sim_to_selected = [0.0] * len(docs)

        while remaining and len(selected) < len(docs):
            best_idx = None
            best_val = -1.0
            for i in remaining:
                mmr = self.mmr_lambda * base_scores[i] - (1 - self.mmr_lambda) * max_sim_to_selected[i]
                if mmr > best_val:
                    best_val = mmr
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)
            # 更新其余文档与已选集合的最大相似度
            for j in remaining:
                sim = _jaccard(docs[j].page_content, docs[best_idx].page_content)
                max_sim_to_selected[j] = max(max_sim_to_selected[j], sim)

        ordered = [docs[i] for i in selected]
        ordered_scores = [base_scores[i] for i in selected]
        return ordered, ordered_scores

    # -- 主入口 ------------------------------------------------------------

    def rerank(
        self,
        query: str,
        docs: List[Document],
        base_scores: Optional[Sequence[float]] = None,
        enable_mmr: bool = True,
    ) -> Tuple[List[Document], List[Dict[str, Any]]]:
        """
        多目标重排。

        Args:
            query: 查询
            docs: 候选文档（保序）
            base_scores: 召回阶段分数（可选，用于无 CrossEncoder 时）
            enable_mmr: 是否启用多样性惩罚

        Returns:
            (ranked_docs, details) details 为逐文档目标分，供溯源面板展示
        """
        with self.monitor.measure("rerank.multi_objective"):
            if not docs:
                return [], []

            s_rel = self._score_rel(query, docs, base_scores)
            s_graph = self._score_graph(docs)
            s_time = self._score_time(docs)

            composite = [
                self.weights["rel"] * r
                + self.weights["graph"] * g
                + self.weights["time"] * t
                for r, g, t in zip(s_rel, s_graph, s_time)
            ]

            # 按文档身份记录各目标分，避免重排后 index 错位
            metrics: Dict[int, Dict[str, float]] = {}
            for i, doc in enumerate(docs):
                metrics[id(doc)] = {
                    "s_rel": s_rel[i],
                    "s_graph": s_graph[i],
                    "s_time": s_time[i],
                }

            if enable_mmr:
                ranked, comp = self._mmr_select(query, docs, composite)
                div_scores = [1.0 - _max_self_sim(ranked, idx) for idx in range(len(ranked))]
            else:
                order = sorted(range(len(docs)), key=lambda i: composite[i], reverse=True)
                ranked = [docs[i] for i in order]
                comp = [composite[i] for i in order]
                div_scores = [0.5] * len(ranked)

            details = []
            for i, doc in enumerate(ranked):
                m = metrics.get(id(doc), {})
                details.append({
                    "content": doc.page_content[:80],
                    "source": doc.metadata.get("source", "unknown"),
                    "s_rel": round(m.get("s_rel", 0.5), 4),
                    "s_graph": round(m.get("s_graph", 0.5), 4),
                    "s_time": round(m.get("s_time", 0.5), 4),
                    "s_div": round(div_scores[i], 4),
                    "final": round(comp[i], 4),
                })
            return ranked, details


def _max_self_sim(ranked: List[Document], idx: int) -> float:
    """与已排序前序文档的最大相似度（用于近似多样性）。"""
    if idx == 0:
        return 0.0
    return max(_jaccard(ranked[idx].page_content, ranked[j].page_content) for j in range(idx))
