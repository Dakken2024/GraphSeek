"""
上下文压缩（白皮书：上下文压缩 Contextual Compression）。

目的：27B 模型支持长上下文但冗余 Token 稀释注意力并增加 TTFT。
在检索结果送入 LLM 前进行"信息密度提纯"。

实现：
- LLM 可用：结构化抽取关键证据句（保留核心实体与关系）
- LLM 不可用：启发式压缩（保留含查询词/实体特征词的句子），保守降级
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from pydantic import BaseModel, Field

from utils.logger import get_logger
from utils.monitoring import Monitor
from utils.text_utils import term_tokens, split_sentences


logger = get_logger(__name__)


class CompressedChunk(BaseModel):
    """压缩后的文档块。"""
    key_sentences: List[str] = Field(
        default_factory=list, description="保留的关键证据句（按原文顺序）"
    )


class ContextCompressor:
    """上下文压缩器。"""

    def __init__(
        self,
        llm_service=None,
        max_sentences_per_chunk: int = 3,
        max_chars_per_chunk: int = 800,
    ) -> None:
        self.llm_service = llm_service
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.max_chars_per_chunk = max_chars_per_chunk
        self.monitor = Monitor()

    def compress(
        self,
        query: str,
        chunks: Sequence[str],
        llm_service=None,
    ) -> List[str]:
        """
        压缩文档块列表。

        Args:
            query: 用户查询（用于关键词保留）
            chunks: 原始文档块

        Returns:
            压缩后的文本列表（与输入顺序一致）
        """
        with self.monitor.measure("context.compress"):
            llm = llm_service or self.llm_service
            query_terms = term_tokens(query)
            result = []
            for chunk in chunks:
                sentences = split_sentences(chunk)
                if len(sentences) <= self.max_sentences_per_chunk:
                    result.append(chunk[: self.max_chars_per_chunk])
                    continue
                result.append(
                    self._compress_chunk(query, chunk, sentences, query_terms, llm)
                )
            return result

    def _compress_chunk(
        self,
        query: str,
        chunk: str,
        sentences: List[str],
        query_terms: set,
        llm,
    ) -> str:
        """单块压缩（优先 LLM，失败/不可用走启发式）。"""
        if llm is not None:
            try:
                prompt = (
                    "你是检索上下文压缩专家。请从下面的文档片段中，选出最可能支撑回答该问题的"
                    f"关键证据句（最多 {self.max_sentences_per_chunk} 句），保持原句不变。\n"
                    "规则：优先保留含核心实体、关系与数值证据的句子。\n\n"
                    f"问题: {query}\n\n文档片段:\n{chunk}"
                )
                c = llm.generate_structured(
                    prompt=prompt, schema=CompressedChunk,
                    temperature=0.1, max_tokens=1024, max_attempts=2,
                )
                kept = [s for s in c.key_sentences if s.strip()]
                if kept:
                    return "\n".join(kept)[: self.max_chars_per_chunk]
            except Exception as e:
                logger.warning(f"LLM 压缩失败，使用启发式: {e}")
        return self._heuristic_compress(sentences, query_terms)

    def _heuristic_compress(self, sentences: List[str], query_terms: set) -> str:
        """启发式：优先保留含查询词或实体特征词的句子。"""
        def score(s: str) -> int:
            return len(term_tokens(s) & query_terms)

        ranked = sorted(range(len(sentences)), key=lambda i: score(sentences[i]), reverse=True)
        kept_idx = sorted(ranked[: self.max_sentences_per_chunk])
        kept = [sentences[i] for i in kept_idx]
        return "\n".join(kept)[: self.max_chars_per_chunk]

    @staticmethod
    def _query_terms(query: str) -> set:
        """（保留兼容别名）查询术语。"""
        return term_tokens(query)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """（保留兼容别名）句子切分。"""
        return split_sentences(text)
